import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from fredapi import Fred
from datetime import datetime, timedelta
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="US Rates & Liquidity Dashboard",
    page_icon="💹",
    layout="wide"
)

# Custom CSS for dark mode aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .section-header {
        color: #00ff88;
        font-size: 32px;
        font-weight: bold;
        margin: 25px 0 15px 0;
        padding-bottom: 12px;
        border-bottom: 3px solid #00ff88;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
    }
    .sub-header {
        color: #00d4ff;
        font-size: 20px;
        font-weight: 600;
        margin: 15px 0 10px 0;
    }
    .metric-container {
        background: linear-gradient(135deg, #1a1d29 0%, #252936 100%);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2d3142;
        margin: 8px 0;
    }
    .frequency-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
        margin-left: 8px;
    }
    .daily { background-color: #00ff88; color: #000; }
    .weekly { background-color: #ffd700; color: #000; }
    .monthly { background-color: #ff6b6b; color: #fff; }
    </style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("⚙️ Dashboard Configuration")
st.sidebar.markdown("---")

# Hardcoded API key
api_key = "ec39710b3961ff09572460cb2548361e"

# Optional: Allow override via sidebar (commented out by default)
# api_key = st.sidebar.text_input(
#     "FRED API Key", 
#     value=api_key,
#     type="password", 
#     help="Enter your FRED API key from https://fred.stlouisfed.org"
# )

st.sidebar.markdown("### 📅 Date Range")
default_start = datetime(2000, 1, 1)
default_end = datetime.now()

start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=default_end)

# Convert to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: This dashboard uses high-frequency series (Daily/Weekly) to show the most recent rates, not monthly averages.")

# Remove API key warning section since it's hardcoded
# The dashboard will now load data automatically

# Cached data fetching function
@st.cache_data(ttl=3600)
def get_fred_data(series_id, start, end, _api_key):
    """Fetch data from FRED API with caching"""
    try:
        fred = Fred(api_key=_api_key)
        data = fred.get_series(series_id, observation_start=start, observation_end=end)
        return data.dropna()  # Remove NaN values
    except Exception as e:
        st.error(f"Error fetching {series_id}: {str(e)}")
        return pd.Series(dtype=float)

def identify_recession_periods(recession_data):
    """
    Helper function to identify recession start and end dates.
    
    USREC series: 0 = no recession, 1 = recession
    This function detects transitions:
    - 0 to 1: Recession starts
    - 1 to 0: Recession ends
    
    Returns: List of tuples [(start1, end1), (start2, end2), ...]
    """
    periods = []
    
    if len(recession_data) == 0:
        return periods
    
    # Convert to DataFrame for easier manipulation
    df = recession_data.to_frame('value')
    df['prev_value'] = df['value'].shift(1)
    
    # Find recession starts (0 -> 1 transition)
    starts = df[(df['prev_value'] == 0) & (df['value'] == 1)].index
    
    # Find recession ends (1 -> 0 transition)
    ends = df[(df['prev_value'] == 1) & (df['value'] == 0)].index
    
    # Handle edge cases
    # If starts with recession
    if len(df) > 0 and df.iloc[0]['value'] == 1:
        starts = pd.DatetimeIndex([df.index[0]]).union(starts)
    
    # If ends with recession
    if len(df) > 0 and df.iloc[-1]['value'] == 1:
        ends = ends.union(pd.DatetimeIndex([df.index[-1]]))
    
    # Pair starts and ends
    for i in range(min(len(starts), len(ends))):
        periods.append((starts[i], ends[i]))
    
    return periods

def format_value(value, indicator_name):
    """Format values based on indicator type"""
    if pd.isna(value):
        return "N/A"
    
    # Rates and percentages
    if any(keyword in indicator_name.lower() for keyword in ['rate', 'sofr', 'funds', 'inflation', 'unemployment']):
        return f"{value:.2f}%"
    
    # TGA and large monetary values (in billions)
    if 'TGA' in indicator_name or 'M2' in indicator_name:
        if value >= 1_000_000:  # Trillions
            return f"${value/1_000_000:.2f}T"
        elif value >= 1_000:  # Billions
            return f"${value/1_000:.1f}B"
        else:
            return f"${value:,.0f}M"
    
    # Default formatting
    return f"{value:,.2f}"

def get_frequency_badge(freq):
    """Return HTML badge for frequency"""
    badges = {
        'Daily': '<span class="frequency-badge daily">DAILY</span>',
        'Weekly': '<span class="frequency-badge weekly">WEEKLY</span>',
        'Monthly': '<span class="frequency-badge monthly">MONTHLY</span>'
    }
    return badges.get(freq, '')

# Main Dashboard
st.title("💹 US Rates & Liquidity Dashboard")
st.markdown("*High-frequency monitoring of Fixed Income markets and Macro liquidity conditions*")

# API key is now hardcoded - no warning needed

# ==========================================
# SECTION 1: YIELD CURVE WITH RECESSION SHADING
# ==========================================
st.markdown('<div class="section-header">📈 US Treasury Yield Curve</div>', unsafe_allow_html=True)

# Treasury series - ALL DAILY FREQUENCY
# DGS = Daily Treasury Constant Maturity Rate
treasury_series = {
    '1-Month': 'DGS1MO',    # Daily
    '3-Month': 'DGS3MO',    # Daily
    '6-Month': 'DGS6MO',    # Daily
    '1-Year': 'DGS1',       # Daily
    '2-Year': 'DGS2',       # Daily
    '3-Year': 'DGS3',       # Daily
    '5-Year': 'DGS5',       # Daily
    '7-Year': 'DGS7',       # Daily
    '10-Year': 'DGS10',     # Daily
    '20-Year': 'DGS20',     # Daily
    '30-Year': 'DGS30'      # Daily
}

# Fetch recession data (Monthly)
recession_data = get_fred_data('USREC', start_date, end_date, api_key)
recession_periods = identify_recession_periods(recession_data)

# Create yield curve chart
fig_yield = go.Figure()

# Add yield curves
for name, series_id in treasury_series.items():
    data = get_fred_data(series_id, start_date, end_date, api_key)
    
    # Emphasize 10Y and 2Y (most watched for inversions)
    if name in ['10-Year', '2-Year']:
        line_width = 3.5
        opacity = 1.0
        color = None  # Use default Plotly colors
    # Make shorter maturities more subtle
    elif name in ['1-Month', '3-Month', '6-Month']:
        line_width = 1.5
        opacity = 0.6
        color = None
    else:
        line_width = 2
        opacity = 0.8
        color = None
    
    fig_yield.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        name=name,
        mode='lines',
        line=dict(width=line_width, color=color) if color else dict(width=line_width),
        opacity=opacity,
        hovertemplate=f'{name}: %{{y:.2f}}%<extra></extra>'
    ))

# Add recession shading (map monthly data to daily chart)
for start, end in recession_periods:
    fig_yield.add_vrect(
        x0=start,
        x1=end,
        fillcolor="gray",
        opacity=0.25,
        layer="below",
        line_width=0,
        annotation_text="Recession",
        annotation_position="top left",
        annotation=dict(font_size=10, font_color="white")
    )

fig_yield.update_layout(
    template='plotly_dark',
    title={
        'text': 'Treasury Yields Over Time - All Daily Data (with NBER Recession Periods)',
        'font': {'size': 20, 'color': '#00ff88'}
    },
    xaxis_title='Date',
    yaxis_title='Yield (%)',
    hovermode='x unified',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(14,17,23,0.8)',
    height=600,
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor='rgba(255,255,255,0.2)',
        borderwidth=1
    ),
    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
)

st.plotly_chart(fig_yield, use_container_width=True)

# Show current spread and key rates
col1, col2, col3, col4 = st.columns(4)

with col1:
    three_month = get_fred_data('DGS3MO', start_date, end_date, api_key)
    if len(three_month) > 0:
        st.metric("3-Month Treasury", f"{three_month.iloc[-1]:.2f}%", 
                  delta=f"{three_month.iloc[-1] - three_month.iloc[-2]:.2f}%" if len(three_month) > 1 else None)

with col2:
    two_year = get_fred_data('DGS2', start_date, end_date, api_key)
    if len(two_year) > 0:
        st.metric("2-Year Treasury", f"{two_year.iloc[-1]:.2f}%", 
                  delta=f"{two_year.iloc[-1] - two_year.iloc[-2]:.2f}%" if len(two_year) > 1 else None)

with col3:
    ten_year = get_fred_data('DGS10', start_date, end_date, api_key)
    if len(ten_year) > 0:
        st.metric("10-Year Treasury", f"{ten_year.iloc[-1]:.2f}%",
                  delta=f"{ten_year.iloc[-1] - ten_year.iloc[-2]:.2f}%" if len(ten_year) > 1 else None)

with col4:
    if len(two_year) > 0 and len(ten_year) > 0:
        spread = ten_year.iloc[-1] - two_year.iloc[-1]
        st.metric("10Y-2Y Spread", f"{spread:.2f}%",
                  delta="Inverted" if spread < 0 else "Normal",
                  delta_color="inverse" if spread < 0 else "normal")

# ==========================================
# SECTION 2: LIQUIDITY & RATES MONITOR
# ==========================================
st.markdown('<div class="section-header">💰 Liquidity & Rates Monitor</div>', unsafe_allow_html=True)

# High-frequency series definitions
# KEY: Using specific high-frequency series IDs (NOT monthly averages)
liquidity_indicators = {
    'SOFR (Overnight)': {
        'series_id': 'SOFR',
        'frequency': 'Daily',
        'description': 'Secured Overnight Financing Rate'
    },
    'Effective Fed Funds Rate': {
        'series_id': 'DFF',  # CRITICAL: Using DFF (Daily) NOT FEDFUNDS (Monthly)
        'frequency': 'Daily',
        'description': 'Daily Effective Federal Funds Rate'
    },
    'TGA Balance': {
        'series_id': 'WTREGEN',
        'frequency': 'Weekly',
        'description': 'Treasury General Account at Federal Reserve'
    },
    'CPI Inflation (YoY)': {
        'series_id': 'CPIAUCSL',
        'frequency': 'Monthly',
        'description': 'Consumer Price Index for All Urban Consumers'
    },
    'Unemployment Rate': {
        'series_id': 'UNRATE',
        'frequency': 'Monthly',
        'description': 'Civilian Unemployment Rate'
    }
}

# Fetch all data and build summary table
# CRITICAL LOGIC: Handle mixed frequencies - get latest value for EACH series independently
@st.cache_data(ttl=3600)
def build_summary_table(_api_key, indicators, start, end):
    """Build summary table with latest values from mixed-frequency data"""
    summary_rows = []
    
    for name, info in indicators.items():
        series_id = info['series_id']
        frequency = info['frequency']
        
        # Fetch data
        data = get_fred_data(series_id, start, end, _api_key)
        
        if len(data) > 0:
            # Get the LAST NON-NULL value (critical for mixed frequencies)
            latest_value = data.iloc[-1]
            latest_date = data.index[-1]
            
            # Format the value
            formatted_value = format_value(latest_value, name)
            
            summary_rows.append({
                'Indicator': name,
                'Latest Value': formatted_value,
                'Date': latest_date.strftime('%Y-%m-%d'),
                'Frequency': frequency,
                'Series ID': series_id,
                'Raw Value': latest_value
            })
    
    return pd.DataFrame(summary_rows)

summary_df = build_summary_table(api_key, liquidity_indicators, start_date, end_date)

st.markdown('<div class="sub-header">📊 Latest Market Conditions</div>', unsafe_allow_html=True)
st.markdown("*Click any row to view detailed historical data and analysis*")

# Display summary table with selection
event = st.dataframe(
    summary_df[['Indicator', 'Latest Value', 'Date', 'Frequency']],
    use_container_width=True,
    hide_index=True,
    selection_mode='single-row',
    on_select='rerun',
    height=250
)

# Determine selected indicator
selected_rows = event.selection.rows if event.selection and event.selection.rows else [0]
selected_idx = selected_rows[0] if selected_rows else 0
selected_indicator = summary_df.iloc[selected_idx]['Indicator']
selected_series_id = summary_df.iloc[selected_idx]['Series ID']
selected_frequency = summary_df.iloc[selected_idx]['Frequency']

# ==========================================
# DRILL-DOWN VIEW
# ==========================================
st.markdown("---")
st.markdown(f'<div class="sub-header">🔍 Detailed Analysis: {selected_indicator} {get_frequency_badge(selected_frequency)}</div>', 
            unsafe_allow_html=True)

# Fetch historical data for selected series
historical_data = get_fred_data(selected_series_id, start_date, end_date, api_key)

if len(historical_data) > 0:
    col_chart, col_stats = st.columns([2.5, 1])
    
    with col_chart:
        # Historical chart
        fig_detail = go.Figure()
        
        fig_detail.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data.values,
            mode='lines',
            name=selected_indicator,
            line=dict(color='#00ff88', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)',
            hovertemplate='%{y:.2f}<extra></extra>'
        ))
        
        # Add recession shading
        for start, end in recession_periods:
            fig_detail.add_vrect(
                x0=start,
                x1=end,
                fillcolor="gray",
                opacity=0.2,
                layer="below",
                line_width=0
            )
        
        fig_detail.update_layout(
            template='plotly_dark',
            title=f'{selected_indicator} - Historical Trend ({selected_frequency})',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(14,17,23,0.8)',
            height=450,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        
        st.plotly_chart(fig_detail, use_container_width=True)
    
    with col_stats:
        # Statistics panel
        st.markdown("**📈 Summary Statistics**")
        
        current_val = historical_data.iloc[-1]
        mean_val = historical_data.mean()
        median_val = historical_data.median()
        max_val = historical_data.max()
        min_val = historical_data.min()
        std_val = historical_data.std()
        
        # Calculate recent change
        if len(historical_data) > 1:
            prev_val = historical_data.iloc[-2]
            change = current_val - prev_val
            change_pct = (change / prev_val * 100) if prev_val != 0 else 0
        else:
            change = 0
            change_pct = 0
        
        stats_data = {
            'Metric': ['Current', 'Change', 'Mean', 'Median', 'Max', 'Min', 'Std Dev'],
            'Value': [
                format_value(current_val, selected_indicator),
                f"{change:+.2f} ({change_pct:+.1f}%)",
                format_value(mean_val, selected_indicator),
                format_value(median_val, selected_indicator),
                format_value(max_val, selected_indicator),
                format_value(min_val, selected_indicator),
                format_value(std_val, selected_indicator)
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, hide_index=True, use_container_width=True, height=280)
        
        # Data freshness indicator
        last_update = historical_data.index[-1]
        days_old = (datetime.now() - last_update).days
        
        if days_old == 0:
            freshness = "🟢 Today"
        elif days_old == 1:
            freshness = "🟡 Yesterday"
        elif days_old <= 7:
            freshness = f"🟡 {days_old} days ago"
        else:
            freshness = f"🔴 {days_old} days ago"
        
        st.markdown(f"**Data Freshness:** {freshness}")
        st.caption(f"Last updated: {last_update.strftime('%Y-%m-%d')}")
    
    # Raw data table (scrollable)
    st.markdown("**📋 Historical Data (Most Recent First)**")
    
    raw_df = historical_data.reset_index()
    raw_df.columns = ['Date', 'Value']
    raw_df['Date'] = raw_df['Date'].dt.strftime('%Y-%m-%d')
    raw_df['Formatted Value'] = raw_df['Value'].apply(lambda x: format_value(x, selected_indicator))
    raw_df = raw_df.sort_values('Date', ascending=False)
    
    st.dataframe(
        raw_df[['Date', 'Formatted Value', 'Value']],
        height=300,
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = raw_df.to_csv(index=False)
    st.download_button(
        label=f"📥 Download {selected_indicator} Data (CSV)",
        data=csv,
        file_name=f"{selected_series_id}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
else:
    st.warning(f"No data available for {selected_indicator} in the selected date range.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
col_footer1, col_footer2 = st.columns(2)

with col_footer1:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"*Dashboard last refreshed: {current_time}*")
    st.caption("Data Source: Federal Reserve Economic Data (FRED)")

with col_footer2:
    st.markdown("**🔑 Key Differences:**")
    st.caption("• DFF = Daily Fed Funds (real-time)")
    st.caption("• FEDFUNDS = Monthly average (lagged)")
    st.caption("• WTREGEN = Weekly TGA (updated Wed)")