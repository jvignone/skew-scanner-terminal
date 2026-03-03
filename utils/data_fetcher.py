"""
Data fetching module with retry logic and error handling
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


class DataFetcher:
    """Fetch market data from Yahoo Finance with robust error handling"""
    
    def __init__(self):
        self.errors = []
        self.max_errors = 20
        
    def add_error(self, error_msg: str):
        """Add error to the error log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.errors.append(f"[{timestamp}] {error_msg}")
        if len(self.errors) > self.max_errors:
            self.errors.pop(0)
    
    def fetch_with_retry(self, func, max_retries=3, initial_delay=1.0):
        """Execute function with exponential backoff retry logic"""
        for attempt in range(max_retries):
            try:
                result = func()
                return result, None
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    return None, error_msg
        return None, "Max retries exceeded"
    
    def fetch_ticker_data(self, ticker: str) -> Dict:
        """
        Fetch all required data for a single ticker
        Returns dict with: spot, history, options_data, error
        """
        result = {
            'ticker': ticker,
            'spot': None,
            'history': None,
            'options_data': [],
            'error': None,
            'status': 'pending'
        }
        
        try:
            # Initialize ticker object
            stock = yf.Ticker(ticker)
            
            # Fetch current price with retry
            def get_info():
                info = stock.info
                if not info or 'currentPrice' not in info and 'regularMarketPrice' not in info:
                    # Fallback to history
                    hist = stock.history(period='1d')
                    if hist.empty:
                        raise ValueError(f"No price data for {ticker}")
                    return hist['Close'].iloc[-1]
                return info.get('currentPrice', info.get('regularMarketPrice'))
            
            spot, error = self.fetch_with_retry(get_info)
            if error:
                result['error'] = f"Failed to fetch spot price: {error}"
                result['status'] = 'failed'
                self.add_error(f"{ticker}: {result['error']}")
                return result
            
            result['spot'] = float(spot)
            
            # Fetch historical data with retry
            def get_history():
                end_date = datetime.now()
                start_date = end_date - timedelta(days=400)  # Extra buffer
                hist = stock.history(start=start_date, end=end_date)
                if hist.empty or len(hist) < 252:
                    raise ValueError(f"Insufficient history for {ticker}: {len(hist)} days")
                return hist
            
            history, error = self.fetch_with_retry(get_history)
            if error:
                result['error'] = f"Failed to fetch history: {error}"
                result['status'] = 'failed'
                self.add_error(f"{ticker}: {result['error']}")
                return result
            
            result['history'] = history
            
            # Fetch options chains with retry
            def get_options():
                expirations = stock.options
                if not expirations:
                    raise ValueError(f"No options data for {ticker}")
                return expirations
            
            expirations, error = self.fetch_with_retry(get_options)
            if error:
                result['error'] = f"No options chains available: {error}"
                result['status'] = 'no_options'
                self.add_error(f"{ticker}: {result['error']}")
                return result
            
            # Filter expirations by DTE (60-180 days)
            today = datetime.now().date()
            valid_expirations = []
            
            for exp_str in expirations:
                try:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                    dte = (exp_date - today).days
                    if 60 <= dte <= 180:
                        valid_expirations.append((exp_str, exp_date, dte))
                except:
                    continue
            
            if not valid_expirations:
                result['error'] = f"No expirations in 60-180 DTE range"
                result['status'] = 'no_valid_exp'
                self.add_error(f"{ticker}: {result['error']}")
                return result
            
            # Fetch option chains for valid expirations
            options_data = []
            for exp_str, exp_date, dte in valid_expirations:
                try:
                    opt_chain = stock.option_chain(exp_str)
                    
                    # Process calls
                    if opt_chain.calls is not None and not opt_chain.calls.empty:
                        calls_df = opt_chain.calls.copy()
                        calls_df['optionType'] = 'call'
                        calls_df['expiration'] = exp_str
                        calls_df['expirationDate'] = exp_date
                        calls_df['dte'] = dte
                        options_data.append(calls_df)
                    
                    # Process puts
                    if opt_chain.puts is not None and not opt_chain.puts.empty:
                        puts_df = opt_chain.puts.copy()
                        puts_df['optionType'] = 'put'
                        puts_df['expiration'] = exp_str
                        puts_df['expirationDate'] = exp_date
                        puts_df['dte'] = dte
                        options_data.append(puts_df)
                    
                    # Small delay between chain fetches
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.add_error(f"{ticker} {exp_str}: Failed to fetch chain - {str(e)}")
                    continue
            
            if not options_data:
                result['error'] = "Failed to fetch any option chains"
                result['status'] = 'no_chains'
                self.add_error(f"{ticker}: {result['error']}")
                return result
            
            result['options_data'] = options_data
            result['status'] = 'success'
            return result
            
        except Exception as e:
            error_trace = traceback.format_exc()
            result['error'] = f"Unexpected error: {str(e)}"
            result['status'] = 'failed'
            self.add_error(f"{ticker}: {result['error']}\n{error_trace}")
            return result
    
    def fetch_multiple_tickers(self, tickers: List[str], progress_callback=None) -> Dict[str, Dict]:
        """
        Fetch data for multiple tickers with progress reporting
        """
        results = {}
        total = len(tickers)
        
        for idx, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(idx, total, ticker)
            
            results[ticker] = self.fetch_ticker_data(ticker)
            
            # Rate limiting: small delay between tickers
            if idx < total - 1:
                time.sleep(0.2)
        
        if progress_callback:
            progress_callback(total, total, "Complete")
        
        return results
