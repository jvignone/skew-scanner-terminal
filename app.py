"""
Streamlit Options Volatility Skew Scanner
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Import utilities
from utils.data_fetcher import DataFetcher
from utils.scanner import SkewScanner
from utils.visualizations import (
    create_iv_vs_strike_chart,
    create_price_chart,
    create_payoff_diagram,
    create_score_breakdown_chart
)

# Page config
st.set_page_config(
    page_title="Skew Scanner Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📊 Skew Scanner Terminal (Streamlit / Yahoo Free)")
st.markdown("*Volatility skew scanner with 30-day minimum holding constraint*")

# Initialize session state
if 'market_data' not in st.session_state:
    st.session_state.market_data = {}
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'candidates' not in st.session_state:
    st.session_state.candidates = []
if 'selected_tickers' not in st.session_state:
    # Default to Tier 1-2
    st.session_state.selected_tickers = (
        SkewScanner.TIERS[1] + SkewScanner.TIERS[2]
    )
if 'diagnostics' not in st.session_state:
    st.session_state.diagnostics = {}
if 'errors' not in st.session_state:
    st.session_state.errors = []

# Sidebar - Ticker Selection
st.sidebar.header("🎯 Ticker Universe")

# Quick selection buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Select All", use_container_width=True):
        all_tickers = []
        for tier_tickers in SkewScanner.TIERS.values():
            all_tickers.extend(tier_tickers)
        st.session_state.selected_tickers = all_tickers
        st.rerun()

with col2:
    if st.button("Select None", use_container_width=True):
        st.session_state.selected_tickers = []
        st.rerun()

if st.sidebar.button("Select Tier 1-2 Only", use_container_width=True):
    st.session_state.selected_tickers = (
        SkewScanner.TIERS[1] + SkewScanner.TIERS[2]
    )
    st.rerun()

st.sidebar.markdown("---")

# Tier selection with expanders
scanner = SkewScanner()

for tier_num in sorted(SkewScanner.TIERS.keys()):
    tier_tickers = SkewScanner.TIERS[tier_num]
    tier_label = f"Tier {tier_num}"
    
    # Add tier context
    if tier_num == 1:
        tier_label += " (Major Indices - Bonus +5)"
    elif tier_num == 2:
        tier_label += " (Mega Cap Tech - Bonus +3)"
    elif tier_num in [5, 6]:
        tier_label += " (Penalty -2)"
    elif tier_num in [7, 8]:
        tier_label += " (Penalty -5)"
    
    with st.sidebar.expander(tier_label):
        for ticker in tier_tickers:
            is_selected = ticker in st.session_state.selected_tickers
            
            if st.checkbox(ticker, value=is_selected, key=f"cb_{ticker}"):
                if ticker not in st.session_state.selected_tickers:
                    st.session_state.selected_tickers.append(ticker)
            else:
                if ticker in st.session_state.selected_tickers:
                    st.session_state.selected_tickers.remove(ticker)

# Show selection count
total_tickers = sum(len(tickers) for tickers in SkewScanner.TIERS.values())
selected_count = len(st.session_state.selected_tickers)
st.sidebar.info(f"**Selected: {selected_count}/{total_tickers}**")

st.sidebar.markdown("---")

# Scan parameters
st.sidebar.header("⚙️ Scan Parameters")

min_score = st.sidebar.slider(
    "Minimum Score",
    min_value=0,
    max_value=100,
    value=40,
    step=5,
    help="Filter candidates by minimum score threshold"
)

dte_min, dte_max = st.sidebar.slider(
    "DTE Range",
    min_value=30,
    max_value=365,
    value=(60, 180),
    step=5,
    help="Days to expiration range (30-day hold constraint enforced)"
)

min_oi = st.sidebar.number_input(
    "Min Open Interest (Strict)",
    min_value=50,
    max_value=1000,
    value=200,
    step=50,
    help="Minimum open interest per leg for strict liquidity"
)

max_ba_pct = st.sidebar.slider(
    "Max Bid-Ask % (Strict)",
    min_value=5.0,
    max_value=50.0,
    value=10.0,
    step=2.5,
    help="Maximum bid-ask spread as % of mid price"
)

allow_neutral = st.sidebar.checkbox(
    "Allow Neutral Regime",
    value=False,
    help="Include trades in neutral market regimes (neither bull nor bear)"
)

momentum_penalty = st.sidebar.checkbox(
    "Apply Momentum Penalty",
    value=True,
    help="Penalize trades that conflict with trend direction (-5 pts)"
)

st.sidebar.markdown("---")

# Action buttons
st.sidebar.header("🔄 Actions")

refresh_btn = st.sidebar.button(
    "🔄 Refresh Market Data",
    use_container_width=True,
    type="primary",
    help="Fetch fresh data from Yahoo Finance (may take 1-2 minutes)"
)

scan_btn = st.sidebar.button(
    "▶️ Run Scan (No Fetch)",
    use_container_width=True,
    help="Re-run scan on existing cached data"
)

# Main content area
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.last_refresh:
        refresh_time = st.session_state.last_refresh.strftime("%H:%M:%S")
        st.metric("Last Refresh", refresh_time)
    else:
        st.metric("Last Refresh", "Never")

with col2:
    scanned_count = len(st.session_state.market_data)
    st.metric("Tickers Scanned", scanned_count)

with col3:
    error_count = len([d for d in st.session_state.market_data.values() 
                      if d.get('status') != 'success'])
    st.metric("Fetch Errors", error_count)

st.markdown("---")

# Refresh market data
if refresh_btn:
    if not st.session_state.selected_tickers:
        st.error("⚠️ No tickers selected! Please select tickers from the sidebar.")
    else:
        st.info(f"🔄 Fetching data for {selected_count} tickers... This may take 1-2 minutes.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        fetcher = DataFetcher()
        
        def progress_callback(current, total, ticker):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"Fetching {ticker}... ({current}/{total})")
        
        # Fetch data with caching
        @st.cache_data(ttl=600)  # 10 minute cache
        def fetch_cached(tickers):
            return fetcher.fetch_multiple_tickers(tickers, progress_callback)
        
        with st.spinner("Fetching market data..."):
            market_data = fetch_cached(st.session_state.selected_tickers)
        
        st.session_state.market_data = market_data
        st.session_state.last_refresh = datetime.now()
        st.session_state.errors = fetcher.errors
        
        progress_bar.empty()
        status_text.empty()
        
        # Auto-run scan after refresh
        st.success(f"✅ Data refresh complete! Fetched {len(market_data)} tickers.")
        scan_btn = True  # Trigger scan

# Run scan
if scan_btn:
    if not st.session_state.market_data:
        st.warning("⚠️ No market data available. Click 'Refresh Market Data' first.")
    else:
        st.info("🔍 Running volatility skew scan...")
        
        scanner = SkewScanner()
        
        with st.spinner("Scanning options chains..."):
            candidates = scanner.scan_all(
                st.session_state.market_data,
                allow_neutral=allow_neutral,
                min_score=min_score,
                momentum_penalty=momentum_penalty
            )
        
        st.session_state.candidates = candidates
        st.session_state.diagnostics = scanner.diagnostics
        
        if candidates:
            st.success(f"✅ Scan complete! Found {len(candidates)} candidates.")
        else:
            st.warning("⚠️ No candidates found. Try relaxing parameters or selecting more tickers.")

# Diagnostics Panel
if st.session_state.diagnostics:
    with st.expander("📊 Scan Diagnostics", expanded=False):
        diag = st.session_state.diagnostics
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tickers Selected", diag.get('tickers_selected', selected_count))
            st.metric("Tickers Fetched", diag.get('tickers_fetched', 0))
            st.metric("Tickers Scanned", diag.get('tickers_scanned', 0))
        
        with col2:
            st.metric("Fetch Errors", diag.get('fetch_errors', 0))
            st.metric("No Options Data", diag.get('no_options_data', 0))
            st.metric("Invalid IV Rows", diag.get('invalid_iv_rows', 0))
        
        with col3:
            st.metric("Delta Failures", diag.get('delta_failures', 0))
            st.metric("No Delta Bucket", diag.get('no_delta_bucket', 0))
            st.metric("Liquidity Filtered", diag.get('liquidity_filtered', 0))
        
        with col4:
            st.metric("Hold Filtered", diag.get('hold_filtered', 0))
            st.metric("Regime Filtered", diag.get('regime_filtered', 0))
            st.metric("Candidates (Strict)", diag.get('candidates_strict', 0))
        
        st.markdown("**Final Results:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Candidates (Relaxed)", diag.get('candidates_relaxed', 0))
        with col2:
            st.metric("Final Candidates", diag.get('final_candidates', 0))

# Error Console
if st.session_state.errors:
    with st.expander(f"⚠️ Error Console ({len(st.session_state.errors)} errors)", expanded=False):
        for error in st.session_state.errors[-20:]:  # Last 20 errors
            st.text(error)

st.markdown("---")

# Candidates Table
if st.session_state.candidates:
    st.header("🎯 Top Trade Candidates")
    
    # Prepare table data
    table_data = []
    for idx, c in enumerate(st.session_state.candidates[:10]):  # Top 10
        table_data.append({
            'Rank': idx + 1,
            'Score': f"{c['score']:.1f}",
            'Ticker': c['ticker'],
            'Tier': c['tier'],
            'Type': c['option_type'].upper(),
            'Regime': c['regime'].capitalize(),
            'DTE': c['dte'],
            'Exit DTE': c['exit_dte'],
            'Skew': f"{c['skew']*100:.1f}%",
            'IV/RV 25Δ': f"{c['iv_rv_25']:.2f}",
            'ROI': f"{c['roi']:.1f}%",
            'Debit': f"${c['debit']:.2f}",
            'Max P/L': f"${c['max_profit']:.2f} / ${c['max_loss']:.2f}",
            'Liq': c['liquidity_mode']
        })
    
    df = pd.DataFrame(table_data)
    
    # Display table with selection
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Candidate selection for detail view
    st.markdown("---")
    st.subheader("📈 Detailed Analysis")
    
    candidate_options = [
        f"#{i+1}: {c['ticker']} {c['option_type'].upper()} - Score {c['score']:.1f}"
        for i, c in enumerate(st.session_state.candidates[:10])
    ]
    
    selected_idx = st.selectbox(
        "Select a candidate to analyze:",
        range(len(candidate_options)),
        format_func=lambda x: candidate_options[x]
    )
    
    if selected_idx is not None:
        candidate = st.session_state.candidates[selected_idx]
        
        # Trade Summary
        st.markdown(f"### {candidate['ticker']} {candidate['option_type'].upper()} Debit Spread")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Score", f"{candidate['score']:.1f}/100")
            st.metric("Spot", f"${candidate['spot']:.2f}")
        with col2:
            st.metric("Expiration", candidate['expiration'])
            st.metric("DTE", f"{candidate['dte']} → {candidate['exit_dte']}")
        with col3:
            st.metric("Entry Debit", f"${candidate['debit']:.2f}")
            st.metric("Max Profit", f"${candidate['max_profit']:.2f}")
        with col4:
            st.metric("Max Loss", f"${candidate['max_loss']:.2f}")
            st.metric("ROI", f"{candidate['roi']:.1f}%")
        
        # Spread Details
        st.markdown("#### Spread Construction")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Long (Buy) Leg:**
            - Strike: ${candidate['k_long']:.2f}
            - Delta: ~25Δ
            - IV: {candidate['iv_25']*100:.1f}%
            - OI: {candidate['long_oi']:.0f}
            - BA Spread: {candidate['long_ba_pct']:.1f}%
            """)
        with col2:
            st.markdown(f"""
            **Short (Sell) Leg:**
            - Strike: ${candidate['k_short']:.2f}
            - Delta: ~50Δ
            - IV: {candidate['iv_50']*100:.1f}%
            - OI: {candidate['short_oi']:.0f}
            - BA Spread: {candidate['short_ba_pct']:.1f}%
            """)
        
        st.markdown(f"**Breakeven:** ${candidate['breakeven']:.2f}")
        st.markdown(f"**Planned Exit:** {candidate['exit_date']} ({candidate['exit_dte']} DTE remaining)")
        
        # Trade Rationale
        st.markdown("#### Why This Trade?")
        st.info(candidate['why_trade'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Strengths:**")
            for bullet in candidate['good_bullets']:
                st.markdown(f"- {bullet}")
        with col2:
            st.markdown("**⚠️ Risks:**")
            for bullet in candidate['bad_bullets']:
                st.markdown(f"- {bullet}")
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("#### Charts")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "IV vs Strike",
            "Price & SMAs",
            "Payoff Diagram",
            "Score Breakdown"
        ])
        
        with tab1:
            try:
                fig = create_iv_vs_strike_chart(candidate, st.session_state.market_data)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating IV chart: {str(e)}")
        
        with tab2:
            try:
                fig = create_price_chart(candidate)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating price chart: {str(e)}")
        
        with tab3:
            try:
                fig = create_payoff_diagram(candidate)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating payoff diagram: {str(e)}")
        
        with tab4:
            try:
                fig = create_score_breakdown_chart(candidate)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating score chart: {str(e)}")

else:
    # Welcome message
    st.info("""
    👋 **Welcome to Skew Scanner Terminal!**
    
    **Quick Start:**
    1. Select tickers from the sidebar (Tier 1-2 recommended for reliability)
    2. Click "🔄 Refresh Market Data" to fetch options chains
    3. Review candidates and detailed analysis
    
    **Pro Tips:**
    - Start with Tier 1-2 tickers for best options liquidity
    - Data is cached for 10 minutes to reduce API load
    - Use "Select Tier 1-2 Only" button for quick scans
    - Check diagnostics panel to troubleshoot scan results
    - All spreads enforce a 30-day minimum holding period
    
    **About:**
    This scanner identifies volatility skew opportunities by comparing implied volatility 
    between 25-delta and 50-delta options. It uses Black-Scholes delta calculations, 
    filters by regime and liquidity, and scores candidates on a 0-100 scale.
    
    Data provided by Yahoo Finance (yfinance) - free, no API key required.
    """)

# Footer
st.markdown("---")
st.caption("""
**Disclaimer:** This tool is for educational purposes only. Options trading involves significant risk.
Always conduct your own due diligence and consult with a financial advisor before trading.
Data sourced from Yahoo Finance may be delayed or inaccurate.
""")
