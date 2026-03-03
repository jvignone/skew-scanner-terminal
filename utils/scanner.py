"""
Main scanning logic: find delta buckets, compute skew, score candidates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from .options_calculator import OptionsCalculator


class SkewScanner:
    """Scan options for volatility skew opportunities"""
    
    # Ticker tier definitions
    TIERS = {
        1: ['SPY', 'QQQ', 'IWM'],
        2: ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA'],
        3: ['AMD', 'INTC', 'NFLX', 'CRM', 'ORCL', 'ADBE', 'UBER', 'SHOP', 'SQ', 'PYPL'],
        4: ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'XLF'],
        5: ['SMH', 'SOXX', 'AVGO', 'QCOM', 'TXN', 'MU'],
        6: ['XLE', 'CVX', 'XOM', 'SLB'],
        7: ['XLV', 'JNJ', 'UNH', 'PFE', 'MRK'],
        8: ['WMT', 'COST', 'HD', 'MCD', 'NKE']
    }
    
    TIER_ADJUSTMENTS = {
        1: 5,
        2: 3,
        3: 0,
        4: 0,
        5: -2,
        6: -2,
        7: -5,
        8: -5
    }
    
    def __init__(self):
        self.calculator = OptionsCalculator()
        self.diagnostics = {
            'tickers_selected': 0,
            'tickers_fetched': 0,
            'tickers_scanned': 0,
            'fetch_errors': 0,
            'no_options_data': 0,
            'invalid_iv_rows': 0,
            'delta_failures': 0,
            'no_delta_bucket': 0,
            'liquidity_filtered': 0,
            'hold_filtered': 0,
            'regime_filtered': 0,
            'candidates_strict': 0,
            'candidates_relaxed': 0,
            'final_candidates': 0
        }
    
    def get_ticker_tier(self, ticker: str) -> int:
        """Get tier number for a ticker"""
        for tier, tickers in self.TIERS.items():
            if ticker.upper() in tickers:
                return tier
        return 3  # Default to tier 3
    
    def find_delta_option(self, df: pd.DataFrame, target_delta: float, 
                         primary_range: Tuple[float, float],
                         fallback_range: Tuple[float, float]) -> Optional[pd.Series]:
        """
        Find option closest to target delta
        
        Args:
            df: Options dataframe (already filtered by expiration and type)
            target_delta: Target delta (e.g., 0.50 or 0.25)
            primary_range: Primary search range (e.g., [0.45, 0.55])
            fallback_range: Fallback range if primary fails
        
        Returns:
            Best matching option row or None
        """
        # Filter valid deltas
        valid = df[df['abs_delta'].notna()].copy()
        if valid.empty:
            return None
        
        # Try primary range first
        primary_mask = (valid['abs_delta'] >= primary_range[0]) & (valid['abs_delta'] <= primary_range[1])
        primary_options = valid[primary_mask]
        
        if not primary_options.empty:
            # Find closest to target
            primary_options['delta_diff'] = (primary_options['abs_delta'] - target_delta).abs()
            best = primary_options.nsmallest(1, 'delta_diff').iloc[0]
            return best
        
        # Try fallback range
        fallback_mask = (valid['abs_delta'] >= fallback_range[0]) & (valid['abs_delta'] <= fallback_range[1])
        fallback_options = valid[fallback_mask]
        
        if not fallback_options.empty:
            fallback_options['delta_diff'] = (fallback_options['abs_delta'] - target_delta).abs()
            best = fallback_options.nsmallest(1, 'delta_diff').iloc[0]
            return best
        
        return None
    
    def compute_skew_for_expiration(self, df: pd.DataFrame, option_type: str) -> Optional[Dict]:
        """
        Compute skew for a single expiration and option type
        
        Returns:
            Dict with delta_50, delta_25, skew, and option details
        """
        # Find 50-delta option
        opt_50 = self.find_delta_option(
            df, 
            target_delta=0.50,
            primary_range=(0.45, 0.55),
            fallback_range=(0.40, 0.60)
        )
        
        # Find 25-delta option
        opt_25 = self.find_delta_option(
            df,
            target_delta=0.25,
            primary_range=(0.20, 0.30),
            fallback_range=(0.15, 0.35)
        )
        
        if opt_50 is None or opt_25 is None:
            self.diagnostics['no_delta_bucket'] += 1
            return None
        
        # Calculate skew
        iv_50 = opt_50.get('impliedVolatility', np.nan)
        iv_25 = opt_25.get('impliedVolatility', np.nan)
        
        if np.isnan(iv_50) or np.isnan(iv_25):
            return None
        
        skew = iv_25 - iv_50
        
        return {
            'option_type': option_type,
            'opt_50': opt_50,
            'opt_25': opt_25,
            'iv_50': iv_50,
            'iv_25': iv_25,
            'skew': skew,
            'expiration': opt_50['expiration'],
            'dte': opt_50['dte']
        }
    
    def check_liquidity(self, option: pd.Series, min_oi: int, max_ba_pct: float) -> Tuple[bool, str]:
        """
        Check if option meets liquidity requirements
        
        Returns:
            (passed, reason)
        """
        # Check open interest
        oi = option.get('openInterest', 0)
        if pd.isna(oi) or oi < min_oi:
            return False, f"OI={oi} < {min_oi}"
        
        # Check mid price
        mid = option.get('mid', 0)
        if pd.isna(mid) or mid <= 0.10:
            return False, f"mid={mid:.2f} <= 0.10"
        
        # Check bid-ask spread
        ba_pct = option.get('ba_pct', 999)
        if pd.isna(ba_pct) or ba_pct > max_ba_pct:
            return False, f"BA%={ba_pct:.1f} > {max_ba_pct}"
        
        return True, "OK"
    
    def create_spread_candidate(self, skew_data: Dict, ticker: str, spot: float,
                               regime: str, rv20: float, history: pd.DataFrame,
                               liquidity_mode: str = 'strict') -> Optional[Dict]:
        """
        Create a spread candidate from skew data
        
        Args:
            skew_data: Output from compute_skew_for_expiration
            ticker: Ticker symbol
            spot: Current spot price
            regime: Market regime
            rv20: Realized volatility (20-day)
            history: Price history
            liquidity_mode: 'strict' or 'relaxed'
        
        Returns:
            Candidate dict or None
        """
        opt_type = skew_data['option_type']
        opt_long = skew_data['opt_25']  # Buy the 25-delta (OTM)
        opt_short = skew_data['opt_50']  # Sell the 50-delta (closer to ATM)
        dte = skew_data['dte']
        
        # Check 30-day hold constraint
        if dte < 60 or (dte - 30) < 7:
            self.diagnostics['hold_filtered'] += 1
            return None
        
        # Set liquidity parameters
        if liquidity_mode == 'strict':
            min_oi = 200
            max_ba_pct = 10.0
        else:
            min_oi = 50
            max_ba_pct = 25.0
        
        # Check liquidity for both legs
        long_ok, long_reason = self.check_liquidity(opt_long, min_oi, max_ba_pct)
        short_ok, short_reason = self.check_liquidity(opt_short, min_oi, max_ba_pct)
        
        if not long_ok or not short_ok:
            self.diagnostics['liquidity_filtered'] += 1
            return None
        
        # Calculate spread pricing
        mid_long = opt_long['mid']
        mid_short = opt_short['mid']
        debit = mid_long - mid_short
        
        if debit <= 0:
            return None
        
        k_long = opt_long['strike']
        k_short = opt_short['strike']
        width = abs(k_short - k_long)
        
        max_profit = width - debit
        max_loss = debit
        roi = (max_profit / max_loss) * 100 if max_loss > 0 else 0
        
        # Calculate breakeven
        if opt_type == 'call':
            breakeven = k_long + debit
        else:
            breakeven = k_long - debit
        
        # Calculate exit DTE
        exit_date = datetime.now().date() + timedelta(days=30)
        exit_dte = dte - 30
        
        # Build candidate
        candidate = {
            'ticker': ticker,
            'tier': self.get_ticker_tier(ticker),
            'spot': spot,
            'regime': regime,
            'option_type': opt_type,
            'expiration': skew_data['expiration'],
            'dte': dte,
            'exit_dte': exit_dte,
            'exit_date': exit_date.strftime('%Y-%m-%d'),
            'iv_50': skew_data['iv_50'],
            'iv_25': skew_data['iv_25'],
            'skew': skew_data['skew'],
            'rv20': rv20,
            'iv_rv_25': skew_data['iv_25'] / rv20 if rv20 > 0 else np.nan,
            'iv_rv_50': skew_data['iv_50'] / rv20 if rv20 > 0 else np.nan,
            'k_long': k_long,
            'k_short': k_short,
            'width': width,
            'debit': debit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'roi': roi,
            'breakeven': breakeven,
            'long_oi': opt_long.get('openInterest', 0),
            'short_oi': opt_short.get('openInterest', 0),
            'long_ba_pct': opt_long.get('ba_pct', 0),
            'short_ba_pct': opt_short.get('ba_pct', 0),
            'liquidity_mode': liquidity_mode,
            'history': history  # Store for later charting
        }
        
        return candidate
    
    def score_candidate(self, candidate: Dict, all_skews: List[float],
                       momentum_penalty: bool = True) -> Dict:
        """
        Score a candidate on 0-100 scale with detailed breakdown
        
        Args:
            candidate: Candidate dict
            all_skews: List of all skew values for percentile calculation
            momentum_penalty: Apply penalty if direction conflicts with trend
        
        Returns:
            Dict with score and explanation
        """
        score_breakdown = {}
        
        # 1. Skew magnitude (0-35 points)
        skew = candidate['skew']
        if len(all_skews) > 1:
            skew_percentile = sum(1 for s in all_skews if s < skew) / len(all_skews)
            skew_score = skew_percentile * 35
        else:
            skew_score = 17.5  # Middle score if no comparison
        score_breakdown['skew_magnitude'] = skew_score
        
        # 2. IV/RV setup (0-25 points)
        iv_rv_25 = candidate['iv_rv_25']
        iv_rv_50 = candidate['iv_rv_50']
        
        if not np.isnan(iv_rv_25) and not np.isnan(iv_rv_50):
            # Reward high IV/RV on 25-delta
            setup_score = min(iv_rv_25 * 10, 15)  # Cap at 15
            
            # Penalize if 50-delta IV/RV is also very high
            if iv_rv_50 > 1.20:
                setup_score -= min((iv_rv_50 - 1.20) * 20, 10)
        else:
            setup_score = 12.5  # Default middle score
        
        setup_score = max(0, min(25, setup_score))
        score_breakdown['iv_rv_setup'] = setup_score
        
        # 3. ATM fairness (0-15 points)
        if not np.isnan(iv_rv_50):
            if iv_rv_50 <= 1.10:
                atm_score = 15
            elif iv_rv_50 >= 1.35:
                atm_score = 0
            else:
                # Linear scale from 15 to 0
                atm_score = 15 - ((iv_rv_50 - 1.10) / 0.25) * 15
        else:
            atm_score = 7.5
        score_breakdown['atm_fairness'] = atm_score
        
        # 4. Liquidity (0-15 points)
        avg_oi = (candidate['long_oi'] + candidate['short_oi']) / 2
        avg_ba_pct = (candidate['long_ba_pct'] + candidate['short_ba_pct']) / 2
        
        # OI component (0-8 points)
        if avg_oi >= 500:
            oi_score = 8
        elif avg_oi >= 200:
            oi_score = 5
        else:
            oi_score = 2
        
        # BA spread component (0-7 points)
        if avg_ba_pct <= 5:
            ba_score = 7
        elif avg_ba_pct <= 10:
            ba_score = 5
        elif avg_ba_pct <= 15:
            ba_score = 3
        else:
            ba_score = 1
        
        liquidity_score = oi_score + ba_score
        score_breakdown['liquidity'] = liquidity_score
        
        # 5. Tier adjustment (0-10 points, centered at 5)
        tier = candidate['tier']
        tier_adj = self.TIER_ADJUSTMENTS.get(tier, 0)
        tier_score = 5 + tier_adj
        tier_score = max(0, min(10, tier_score))
        score_breakdown['tier'] = tier_score
        
        # 6. Optional momentum penalty
        momentum_score = 0
        if momentum_penalty:
            regime = candidate['regime']
            opt_type = candidate['option_type']
            
            # Penalize if direction conflicts
            if (regime == 'bull' and opt_type == 'put') or (regime == 'bear' and opt_type == 'call'):
                momentum_score = -5
        score_breakdown['momentum'] = momentum_score
        
        # Total score
        total_score = sum(score_breakdown.values())
        total_score = max(0, min(100, total_score))
        
        return {
            'score': total_score,
            'breakdown': score_breakdown
        }
    
    def generate_explanation(self, candidate: Dict, score_data: Dict) -> Dict:
        """
        Generate trade explanation with good/bad bullets
        
        Returns:
            Dict with why_trade, good_bullets, bad_bullets
        """
        ticker = candidate['ticker']
        opt_type = candidate['option_type'].upper()
        regime = candidate['regime']
        skew = candidate['skew']
        iv_rv_25 = candidate['iv_rv_25']
        roi = candidate['roi']
        dte = candidate['dte']
        
        # Why this trade
        direction = "bullish" if opt_type == 'CALL' else "bearish"
        why_trade = (
            f"{ticker} shows a {skew*100:.1f}% volatility skew between 25Δ and 50Δ {opt_type}s, "
            f"indicating elevated premium on OTM strikes. The {direction} {opt_type} debit spread "
            f"capitalizes on this mispricing while limiting downside to ${candidate['max_loss']:.2f}. "
            f"With {dte} DTE and a planned 30-day hold, the trade has {candidate['exit_dte']} DTE remaining at exit, "
            f"offering a {roi:.1f}% max ROI."
        )
        
        # Good bullets
        good_bullets = []
        
        if skew > 0.05:
            good_bullets.append(f"Strong skew of {skew*100:.1f}% creates favorable spread pricing")
        
        if not np.isnan(iv_rv_25) and iv_rv_25 > 1.15:
            good_bullets.append(f"25Δ IV/RV of {iv_rv_25:.2f} suggests elevated OTM premiums")
        
        if regime == 'bull' and opt_type == 'call':
            good_bullets.append("Trade direction aligns with bullish regime (price > SMA50, SMA20 > SMA50)")
        elif regime == 'bear' and opt_type == 'put':
            good_bullets.append("Trade direction aligns with bearish regime (price < SMA50, SMA20 < SMA50)")
        
        if candidate['long_oi'] > 300 and candidate['short_oi'] > 300:
            good_bullets.append(f"Excellent liquidity: {candidate['long_oi']:.0f}/{candidate['short_oi']:.0f} OI on legs")
        
        if roi > 30:
            good_bullets.append(f"Attractive {roi:.1f}% max ROI for defined-risk spread")
        
        if candidate['tier'] <= 2:
            good_bullets.append(f"Tier {candidate['tier']} ticker with consistent options liquidity")
        
        # Bad bullets
        bad_bullets = []
        
        iv_rv_50 = candidate['iv_rv_50']
        if not np.isnan(iv_rv_50) and iv_rv_50 > 1.15:
            bad_bullets.append(f"50Δ IV/RV of {iv_rv_50:.2f} suggests ATM strikes also expensive")
        
        if regime == 'bull' and opt_type == 'put':
            bad_bullets.append("Bearish trade direction conflicts with bullish trend")
        elif regime == 'bear' and opt_type == 'call':
            bad_bullets.append("Bullish trade direction conflicts with bearish trend")
        
        avg_ba_pct = (candidate['long_ba_pct'] + candidate['short_ba_pct']) / 2
        if avg_ba_pct > 10:
            bad_bullets.append(f"Wide bid-ask spreads averaging {avg_ba_pct:.1f}% may impact entry/exit")
        
        if candidate['liquidity_mode'] == 'relaxed':
            bad_bullets.append("Liquidity below strict thresholds - use limit orders")
        
        if candidate['exit_dte'] < 15:
            bad_bullets.append(f"Only {candidate['exit_dte']} DTE remaining at planned exit - theta decay accelerates")
        
        if candidate['tier'] >= 6:
            bad_bullets.append(f"Tier {candidate['tier']} ticker may have inconsistent options volume")
        
        if roi < 20:
            bad_bullets.append(f"Limited {roi:.1f}% max ROI - risk/reward may not justify position size")
        
        return {
            'why_trade': why_trade,
            'good_bullets': good_bullets,
            'bad_bullets': bad_bullets
        }
    
    def scan_ticker(self, ticker: str, data: Dict, allow_neutral: bool = False,
                   min_oi_strict: int = 200, max_ba_pct_strict: float = 10.0,
                   momentum_penalty: bool = True) -> List[Dict]:
        """
        Scan a single ticker for spread candidates
        
        Returns:
            List of candidate dicts
        """
        if data['status'] != 'success':
            if data['status'] == 'no_options':
                self.diagnostics['no_options_data'] += 1
            else:
                self.diagnostics['fetch_errors'] += 1
            return []
        
        self.diagnostics['tickers_scanned'] += 1
        
        spot = data['spot']
        history = data['history']
        options_data = data['options_data']
        
        # Calculate realized volatility
        rv20 = self.calculator.calculate_realized_volatility(history['Close'], window=20)
        if rv20 is None or rv20 <= 0:
            return []
        
        # Determine regime
        regime = self.calculator.determine_regime(history['Close'], spot)
        
        # Skip neutral regime unless allowed
        if regime == 'neutral' and not allow_neutral:
            self.diagnostics['regime_filtered'] += 1
            return []
        
        # Concatenate all options data
        all_options = pd.concat(options_data, ignore_index=True)
        
        # Enrich with calculations
        all_options = self.calculator.enrich_options_data(all_options, spot)
        
        # Update diagnostics
        self.diagnostics['invalid_iv_rows'] += self.calculator.stats['invalid_iv_rows']
        self.diagnostics['delta_failures'] += self.calculator.stats['delta_failures']
        
        # Determine which option type to scan based on regime
        if regime == 'bull':
            option_types = ['call']
        elif regime == 'bear':
            option_types = ['put']
        else:  # neutral
            option_types = ['call', 'put']
        
        # Scan each expiration
        candidates = []
        expirations = all_options['expiration'].unique()
        
        for expiration in expirations:
            exp_data = all_options[all_options['expiration'] == expiration]
            
            for opt_type in option_types:
                type_data = exp_data[exp_data['optionType'] == opt_type]
                
                if type_data.empty:
                    continue
                
                # Compute skew
                skew_data = self.compute_skew_for_expiration(type_data, opt_type)
                
                if skew_data is None:
                    continue
                
                # Try strict liquidity first
                candidate = self.create_spread_candidate(
                    skew_data, ticker, spot, regime, rv20, history, 
                    liquidity_mode='strict'
                )
                
                if candidate:
                    candidates.append(candidate)
                    self.diagnostics['candidates_strict'] += 1
                else:
                    # Try relaxed liquidity
                    candidate = self.create_spread_candidate(
                        skew_data, ticker, spot, regime, rv20, history,
                        liquidity_mode='relaxed'
                    )
                    if candidate:
                        candidates.append(candidate)
                        self.diagnostics['candidates_relaxed'] += 1
        
        # Score all candidates
        all_skews = [c['skew'] for c in candidates]
        
        for candidate in candidates:
            score_data = self.score_candidate(candidate, all_skews, momentum_penalty)
            candidate['score'] = score_data['score']
            candidate['score_breakdown'] = score_data['breakdown']
            
            explanation = self.generate_explanation(candidate, score_data)
            candidate.update(explanation)
        
        return candidates
    
    def scan_all(self, market_data: Dict[str, Dict], allow_neutral: bool = False,
                min_score: float = 0, momentum_penalty: bool = True) -> List[Dict]:
        """
        Scan all tickers and return sorted candidates
        
        Args:
            market_data: Dict of {ticker: data_dict}
            allow_neutral: Include neutral regime trades
            min_score: Minimum score threshold
            momentum_penalty: Apply momentum penalty
        
        Returns:
            List of candidates sorted by score descending
        """
        all_candidates = []
        
        for ticker, data in market_data.items():
            candidates = self.scan_ticker(
                ticker, data, allow_neutral, 
                momentum_penalty=momentum_penalty
            )
            all_candidates.extend(candidates)
        
        # Filter by min score
        filtered = [c for c in all_candidates if c['score'] >= min_score]
        
        self.diagnostics['final_candidates'] = len(filtered)
        
        # Sort by score descending
        filtered.sort(key=lambda x: x['score'], reverse=True)
        
        return filtered
