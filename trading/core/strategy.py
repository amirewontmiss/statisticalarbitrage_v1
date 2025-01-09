import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import statsmodels.api as sm
from datetime import datetime, timedelta
import time
import logging
import warnings
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

@dataclass
class PairStats:
    hedge_ratio: float
    half_life: float
    correlation: float
    volatility: float
    spread_mean: float
    spread_std: float
    current_zscore: float
    adf_pvalue: float
    hurst_exponent: float
    var_95: float
    es_95: float
    sharpe_ratio: float

class RiskManager:
    def __init__(self, max_portfolio_var: float = 0.02):
        self.max_portfolio_var = max_portfolio_var
        self.position_limits = {}
        self.risk_metrics = {}
        
    def calculate_position_limits(self, 
                                pair_stats: Dict[Tuple[str, str], PairStats],
                                current_positions: Dict) -> Dict:
        """Calculate position limits based on risk metrics"""
        for pair, stats in pair_stats.items():
            # Calculate VaR-based position limit
            var_limit = self.max_portfolio_var / stats.var_95
            
            # Adjust for correlation with existing positions
            if current_positions:
                correlation_penalty = self._calculate_correlation_penalty(pair, current_positions)
                var_limit *= (1 - correlation_penalty)
            
            self.position_limits[pair] = var_limit
            
        return self.position_limits
    
    def _calculate_correlation_penalty(self, 
                                     pair: Tuple[str, str], 
                                     current_positions: Dict) -> float:
        # Implement correlation-based position sizing
        return 0.2  # Simplified version

class StatArbStrategy:
    def __init__(self, 
                 pairs: List[Tuple[str, str]], 
                 lookback_period: int = 252,
                 zscore_threshold: float = 2.0,
                 position_size: float = 100000,
                 risk_manager: Optional[RiskManager] = None):
        
        self.pairs = pairs
        self.lookback_period = lookback_period
        self.zscore_threshold = zscore_threshold
        self.position_size = position_size
        self.risk_manager = risk_manager or RiskManager()
        
        self.current_positions = {}
        self.historical_data = {}
        self.pair_metrics = {}
        self.performance_metrics = {}
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize async session
        self.session = None
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def _fetch_data_async(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data asynchronously using aiohttp"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            'period1': int(start_date.timestamp()),
            'period2': int(end_date.timestamp()),
            'interval': '1d'
        }
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            # Process raw data into DataFrame
            # ... (implementation details)
            return pd.DataFrame()  # Simplified for brevity

    def calculate_hurst_exponent(self, prices: pd.Series) -> float:
        """Calculate Hurst exponent for mean reversion strength"""
        lags = range(2, 100)
        tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate mean reversion half-life"""
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        spread_lag = spread_lag[1:]
        spread_diff = spread_diff[1:]
        
        model = sm.OLS(spread_diff, spread_lag)
        result = model.fit()
        half_life = -np.log(2) / result.params[0]
        return half_life

    async def fetch_historical_data(self) -> None:
        """Fetch historical price data for all pairs asynchronously"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_period * 2)
        
        tasks = []
        for pair in self.pairs:
            tasks.extend([
                self._fetch_data_async(pair[0], start_date, end_date),
                self._fetch_data_async(pair[1], start_date, end_date)
            ])
            
        results = await asyncio.gather(*tasks)
        
        # Process results into historical_data dictionary
        for i, pair in enumerate(self.pairs):
            self.historical_data[pair] = pd.DataFrame({
                pair[0]: results[i*2]['Adj Close'],
                pair[1]: results[i*2 + 1]['Adj Close']
            }).dropna()

    def calculate_pair_metrics(self) -> None:
        """Calculate comprehensive pair metrics"""
        for pair in self.pairs:
            try:
                data = self.historical_data[pair].tail(self.lookback_period)
                
                # Calculate hedge ratio using Ledoit-Wolf shrinkage
                lw = LedoitWolf().fit(data)
                hedge_ratio = lw.covariance_[0, 1] / lw.covariance_[0, 0]
                
                # Calculate spread
                spread = data[pair[1]] - hedge_ratio * data[pair[0]]
                
                # Calculate additional metrics
                hurst = self.calculate_hurst_exponent(spread)
                half_life = self.calculate_half_life(spread)
                correlation = data[pair[0]].corr(data[pair[1]])
                
                # Calculate risk metrics
                returns = spread.pct_change().dropna()
                var_95 = np.percentile(returns, 5)
                es_95 = returns[returns <= var_95].mean()
                
                # Calculate Sharpe ratio
                sharpe = np.sqrt(252) * returns.mean() / returns.std()
                
                self.pair_metrics[pair] = PairStats(
                    hedge_ratio=hedge_ratio,
                    half_life=half_life,
                    correlation=correlation,
                    volatility=returns.std() * np.sqrt(252),
                    spread_mean=spread.mean(),
                    spread_std=spread.std(),
                    current_zscore=(spread.iloc[-1] - spread.mean()) / spread.std(),
                    adf_pvalue=sm.tsa.stattools.adfuller(spread)[1],
                    hurst_exponent=hurst,
                    var_95=var_95,
                    es_95=es_95,
                    sharpe_ratio=sharpe
                )
                
            except Exception as e:
                self.logger.error(f"Error calculating metrics for pair {pair}: {str(e)}")
