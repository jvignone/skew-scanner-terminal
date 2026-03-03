"""
Options calculations: Black-Scholes delta, realized volatility, Greeks
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Tuple, Optional


class OptionsCalculator:
    """Calculate option metrics and Greeks"""
    
    def __init__(self, risk_free_rate=0.04):
        self.risk_free_rate = risk_free_rate
        self.stats = {
            'invalid_iv_rows': 0,
            'delta_failures': 0,
            'calculations_completed': 0
        }
    
    def black_scholes_delta(self, S: float, K: float, T: float, 
                           sigma: float, option_type: str) -> Optional[float]:
        """
        Calculate Black-Scholes delta
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility (decimal, e.g., 0.25 for 25%)
            option_type: 'call' or 'put'
        
        Returns:
            Delta value or None if calculation fails
        """
        try:
            # Validation
            if sigma is None or np.isnan(sigma) or sigma < 0.01:
                self.stats['invalid_iv_rows'] += 1
                return None
            
            if T <= 0:
                self.stats['invalid_iv_rows'] += 1
                return None
            
            if S <= 0 or K <= 0:
                self.stats['invalid_iv_rows'] += 1
                return None
            
            # Calculate d1
            d1 = (np.log(S / K) + (self.risk_free_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            
            # Calculate delta based on option type
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
            elif option_type.lower() == 'put':
                delta = norm.cdf(d1) - 1.0
            else:
                self.stats['delta_failures'] += 1
                return None
            
            self.stats['calculations_completed'] += 1
            return delta
            
        except Exception as e:
            self.stats['delta_failures'] += 1
            return None
    
    def calculate_realized_volatility(self, prices: pd.Series, window: int = 20) -> Optional[float]:
        """
        Calculate annualized realized volatility from price series
        
        Args:
            prices: Series of closing prices
            window: Lookback window (default 20 days)
        
        Returns:
            Annualized realized volatility (decimal) or None
        """
        try:
            if len(prices) < window + 1:
                return None
            
            # Calculate log returns
            log_returns = np.log(prices / prices.shift(1))
            
            # Get last 'window' returns
            recent_returns = log_returns.iloc[-(window+1):-1]
            
            if len(recent_returns) < window:
                return None
            
            # Annualized standard deviation
            rv = recent_returns.std() * np.sqrt(252)
            
            return rv if not np.isnan(rv) else None
            
        except Exception as e:
            return None
    
    def calculate_sma(self, prices: pd.Series, window: int) -> Optional[float]:
        """Calculate simple moving average"""
        try:
            if len(prices) < window:
                return None
            return prices.iloc[-window:].mean()
        except:
            return None
    
    def determine_regime(self, prices: pd.Series, current_price: float) -> str:
        """
        Determine market regime: bull, bear, or neutral
        
        Args:
            prices: Historical price series
            current_price: Current spot price
        
        Returns:
            'bull', 'bear', or 'neutral'
        """
        try:
            sma20 = self.calculate_sma(prices, 20)
            sma50 = self.calculate_sma(prices, 50)
            
            if sma20 is None or sma50 is None:
                return 'neutral'
            
            # Bull: price > SMA50 AND SMA20 > SMA50
            if current_price > sma50 and sma20 > sma50:
                return 'bull'
            
            # Bear: price < SMA50 AND SMA20 < SMA50
            if current_price < sma50 and sma20 < sma50:
                return 'bear'
            
            return 'neutral'
            
        except:
            return 'neutral'
    
    def enrich_options_data(self, options_df: pd.DataFrame, spot: float) -> pd.DataFrame:
        """
        Add calculated fields to options dataframe
        
        Args:
            options_df: Raw options data
            spot: Current spot price
        
        Returns:
            Enriched dataframe with delta, mid, etc.
        """
        df = options_df.copy()
        
        # Calculate mid price
        df['mid'] = (df['bid'] + df['ask']) / 2.0
        
        # Calculate time to expiration in years
        df['T'] = df['dte'] / 365.0
        
        # Calculate delta
        deltas = []
        for idx, row in df.iterrows():
            delta = self.black_scholes_delta(
                S=spot,
                K=row['strike'],
                T=row['T'],
                sigma=row.get('impliedVolatility', np.nan),
                option_type=row['optionType']
            )
            deltas.append(delta)
        
        df['delta'] = deltas
        df['abs_delta'] = df['delta'].abs()
        
        # Calculate bid-ask spread percentage
        df['ba_pct'] = np.where(
            df['mid'] > 0,
            ((df['ask'] - df['bid']) / df['mid']) * 100,
            np.nan
        )
        
        return df
