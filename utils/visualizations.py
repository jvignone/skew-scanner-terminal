"""
Plotly visualizations for candidates
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime


def create_iv_vs_strike_chart(candidate: Dict, market_data: Dict) -> go.Figure:
    """
    Create IV vs Strike chart for the expiration
    """
    ticker = candidate['ticker']
    expiration = candidate['expiration']
    opt_type = candidate['option_type']
    
    data = market_data[ticker]
    options_data = data['options_data']
    
    # Concatenate all options
    all_options = pd.concat(options_data, ignore_index=True)
    
    # Filter to this expiration
    exp_options = all_options[all_options['expiration'] == expiration].copy()
    
    # Separate calls and puts
    calls = exp_options[exp_options['optionType'] == 'call'].sort_values('strike')
    puts = exp_options[exp_options['optionType'] == 'put'].sort_values('strike')
    
    fig = go.Figure()
    
    # Plot calls
    if not calls.empty:
        fig.add_trace(go.Scatter(
            x=calls['strike'],
            y=calls['impliedVolatility'] * 100,
            mode='lines+markers',
            name='Calls',
            line=dict(color='green', width=2),
            marker=dict(size=6)
        ))
    
    # Plot puts
    if not puts.empty:
        fig.add_trace(go.Scatter(
            x=puts['strike'],
            y=puts['impliedVolatility'] * 100,
            mode='lines+markers',
            name='Puts',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))
    
    # Mark the spread strikes
    k_long = candidate['k_long']
    k_short = candidate['k_short']
    spot = candidate['spot']
    
    fig.add_vline(x=spot, line_dash="dash", line_color="white", 
                  annotation_text=f"Spot: ${spot:.2f}")
    fig.add_vline(x=k_long, line_dash="dot", line_color="cyan",
                  annotation_text=f"Long: ${k_long:.0f}")
    fig.add_vline(x=k_short, line_dash="dot", line_color="yellow",
                  annotation_text=f"Short: ${k_short:.0f}")
    
    fig.update_layout(
        title=f"{ticker} Volatility Surface - {expiration} ({candidate['dte']} DTE)",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility (%)",
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_price_chart(candidate: Dict) -> go.Figure:
    """
    Create price chart with SMAs
    """
    history = candidate['history']
    ticker = candidate['ticker']
    spot = candidate['spot']
    
    # Get last 90 days
    hist_90 = history.iloc[-90:]
    
    # Calculate SMAs
    hist_90['SMA20'] = hist_90['Close'].rolling(20).mean()
    hist_90['SMA50'] = hist_90['Close'].rolling(50).mean()
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=hist_90.index,
        y=hist_90['Close'],
        mode='lines',
        name='Close',
        line=dict(color='white', width=2)
    ))
    
    # SMA20
    fig.add_trace(go.Scatter(
        x=hist_90.index,
        y=hist_90['SMA20'],
        mode='lines',
        name='SMA20',
        line=dict(color='cyan', width=1.5)
    ))
    
    # SMA50
    fig.add_trace(go.Scatter(
        x=hist_90.index,
        y=hist_90['SMA50'],
        mode='lines',
        name='SMA50',
        line=dict(color='orange', width=1.5)
    ))
    
    # Current spot
    fig.add_hline(y=spot, line_dash="dash", line_color="yellow",
                  annotation_text=f"Current: ${spot:.2f}")
    
    # Breakeven
    fig.add_hline(y=candidate['breakeven'], line_dash="dot", line_color="lime",
                  annotation_text=f"Breakeven: ${candidate['breakeven']:.2f}")
    
    fig.update_layout(
        title=f"{ticker} - 90-Day Price Action & SMAs (Regime: {candidate['regime'].upper()})",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_payoff_diagram(candidate: Dict) -> go.Figure:
    """
    Create P&L payoff diagram at expiration
    """
    k_long = candidate['k_long']
    k_short = candidate['k_short']
    debit = candidate['debit']
    opt_type = candidate['option_type']
    spot = candidate['spot']
    
    # Create price range
    price_min = min(k_long, k_short) * 0.85
    price_max = max(k_long, k_short) * 1.15
    prices = np.linspace(price_min, price_max, 200)
    
    pnl = []
    for price in prices:
        if opt_type == 'call':
            # Long call
            long_payoff = max(0, price - k_long)
            # Short call
            short_payoff = -max(0, price - k_short)
        else:  # put
            # Long put
            long_payoff = max(0, k_long - price)
            # Short put
            short_payoff = -max(0, k_short - price)
        
        spread_pnl = long_payoff + short_payoff - debit
        pnl.append(spread_pnl)
    
    fig = go.Figure()
    
    # P&L line
    fig.add_trace(go.Scatter(
        x=prices,
        y=pnl,
        mode='lines',
        name='P&L',
        line=dict(color='lime', width=3),
        fill='tonexty',
        fillcolor='rgba(0,255,0,0.1)'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    
    # Spot price
    fig.add_vline(x=spot, line_dash="dash", line_color="yellow",
                  annotation_text=f"Spot: ${spot:.2f}")
    
    # Breakeven
    fig.add_vline(x=candidate['breakeven'], line_dash="dot", line_color="cyan",
                  annotation_text=f"BE: ${candidate['breakeven']:.2f}")
    
    # Max profit/loss annotations
    fig.add_annotation(
        x=price_max * 0.95,
        y=candidate['max_profit'],
        text=f"Max Profit: ${candidate['max_profit']:.2f}",
        showarrow=False,
        bgcolor="green",
        font=dict(color="white")
    )
    
    fig.add_annotation(
        x=price_min * 1.05,
        y=-candidate['max_loss'],
        text=f"Max Loss: ${candidate['max_loss']:.2f}",
        showarrow=False,
        bgcolor="red",
        font=dict(color="white")
    )
    
    fig.update_layout(
        title=f"{candidate['ticker']} {opt_type.upper()} Spread Payoff at Expiration",
        xaxis_title="Stock Price at Expiration ($)",
        yaxis_title="P&L ($)",
        template="plotly_dark",
        height=400
    )
    
    return fig


def create_score_breakdown_chart(candidate: Dict) -> go.Figure:
    """
    Create horizontal bar chart of score breakdown
    """
    breakdown = candidate['score_breakdown']
    
    labels = list(breakdown.keys())
    values = list(breakdown.values())
    
    # Color coding
    colors = []
    for val in values:
        if val >= 10:
            colors.append('green')
        elif val >= 5:
            colors.append('yellow')
        elif val >= 0:
            colors.append('orange')
        else:
            colors.append('red')
    
    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:.1f}" for v in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"Score Breakdown (Total: {candidate['score']:.1f}/100)",
        xaxis_title="Points",
        yaxis_title="Component",
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    
    return fig
