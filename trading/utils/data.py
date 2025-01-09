class DataProcessor:
    @staticmethod
    def clean_and_validate(data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate price data"""
        # Remove outliers
        z_scores = np.abs(stats.zscore(data))
        data = data[(z_scores < 3).all(axis=1)]
        
        # Forward fill missing values
        data = data.ffill()
        
        # Add data quality metrics
        data['quality_score'] = 1 - data.isnull().mean()
        
        return data
