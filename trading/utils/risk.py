class RiskMetrics:
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_es(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall"""
        var = RiskMetrics.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_position_size(volatility: float, 
                              max_loss: float,
                              account_size: float) -> float:
        """Calculate position size based on volatility and maximum loss"""
        return (max_loss * account_size) / (volatility * 2.58) 

