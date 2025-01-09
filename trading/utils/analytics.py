import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import norm
import talib

class PerformanceAnalytics:
    @staticmethod
    def calculate_returns_metrics(returns: pd.Series) -> Dict:
        """Calculate comprehensive return metrics"""
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'cagr': (1 + returns).prod() ** (252/len(returns)) - 1,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std(),
            'sortino_ratio': np.sqrt(252) * returns.mean() / returns[returns < 0].std(),
            'max_drawdown': (returns.cumsum() - returns.cumsum().cummax()).min(),
            'win_rate': len(returns[returns > 0]) / len(returns),
            'avg_win': returns[returns > 0].mean(),
            'avg_loss': returns[returns < 0].mean(),
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum())
        }
        return metrics

class TechnicalAnalysis:
    @staticmethod
    def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for analysis"""
        df = data.copy()
        
        # Add momentum indicators
        df['rsi'] = talib.RSI(df['close'].values)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'].values)
        
        # Add volatility indicators
        df['bbands_upper'], df['bbands_middle'], df['bbands_lower'] = talib.BBANDS(df['close'].values)
        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values)
        
        # Add trend indicators
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values)
        df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values)
        
        return df
