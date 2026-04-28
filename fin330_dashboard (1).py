# ══════════════════════════════════════════════════════════════════════════════
# FIN 330 Stock Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════
# Structure:
#   1. Imports
#   2. Theme & Styling
#   3. Logic / Calculation Functions
#   4. Chart Functions
#   5. Display / UI Functions
#   6. Main App Entry Point
# ══════════════════════════════════════════════════════════════════════════════


# ─── 1. IMPORTS ────────────────────────────────────────────────────────────────
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ─── 2. THEME & STYLING ────────────────────────────────────────────────────────
# All visual settings live here so they are easy to change in one place.

# Color palette (finance dark theme)
C_BG      = "#0D1117"   # page background
C_SURFACE = "#161B22"   # card / panel background
C_BORDER  = "#30363D"   # subtle borders
C_ACCENT  = "#2EA043"   # green accent (buy / positive)
C_DANGER  = "#DA3633"   # red accent (sell / negative)
C_NEUTRAL = "#388BFD"   # blue accent (neutral / info)
C_TEXT    = "#E6EDF3"   # primary text
C_MUTED   = "#8B949E"   # secondary / muted text
C_GOLD    = "#D4A017"   # gold accent for headers

# Chart line colors
CH_PRICE  = "#388BFD"   # price line
CH_MA20   = "#F0883E"   # 20-day moving average
CH_MA50   = "#D4A017"   # 50-day moving average
CH_RSI    = "#B48EAD"   # RSI line
CH_PORT   = "#2EA043"   # portfolio cumulative return
CH_BENCH  = "#388BFD"   # benchmark cumulative return


def apply_theme():
    """Inject CSS to apply a dark finance theme across the entire app."""
    st.markdown(f"""
    <style>
        /* Page background */
        .stApp {{
            background-color: {C_BG};
            color: {C_TEXT};
            font-family: 'Courier New', monospace;
        }}

        /* Sidebar background and text */
        section[data-testid="stSidebar"] {{
            background-color: {C_SURFACE};
            border-right: 1px solid {C_BORDER};
        }}
        section[data-testid="stSidebar"] * {{
            color: {C_TEXT} !important;
        }}

        /* Main content area padding */
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        /* Section headers */
        h1, h2, h3 {{
            color: {C_TEXT} !important;
            font-family: 'Courier New', monospace !important;
            letter-spacing: 0.05em;
        }}

        /* Metric cards: surface background with a border */
        [data-testid="stMetric"] {{
            background-color: {C_SURFACE};
            border: 1px solid {C_BORDER};
            border-radius: 6px;
            padding: 1rem;
        }}
        [data-testid="stMetricLabel"] {{
            color: {C_MUTED} !important;
            font-size: 0.75rem !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        [data-testid="stMetricValue"] {{
            color: {C_TEXT} !important;
            font-size: 1.4rem !important;
            font-weight: bold;
        }}

        /* Dataframe border */
        [data-testid="stDataFrame"] {{
            border: 1px solid {C_BORDER};
            border-radius: 6px;
        }}

        /* Run buttons: green fill */
        .stButton > button {{
            background-color: {C_ACCENT};
            color: white;
            border: none;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            padding: 0.5rem 1.5rem;
            width: 100%;
        }}
        .stButton > button:hover {{
            background-color: #3DAA52;
        }}

        /* Download button: outlined in blue */
        .stDownloadButton > button {{
            background-color: {C_SURFACE};
            color: {C_NEUTRAL};
            border: 1px solid {C_NEUTRAL};
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            width: 100%;
        }}

        /* Success message: subtle green left border */
        .stSuccess {{
            background-color: rgba(46, 160, 67, 0.12);
            border-left: 3px solid {C_ACCENT};
            border-radius: 4px;
        }}

        /* Error message: subtle red left border */
        .stError {{
            background-color: rgba(218, 54, 51, 0.12);
            border-left: 3px solid {C_DANGER};
        }}

        /* Small step label above each subheader */
        .step-label {{
            font-size: 0.68rem;
            color: {C_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.14em;
            margin-bottom: 0.15rem;
        }}

        /* Recommendation badges */
        .badge-buy {{
            display: inline-block;
            background-color: rgba(46,160,67,0.18);
            color: {C_ACCENT};
            border: 1px solid {C_ACCENT};
            border-radius: 4px;
            padding: 0.35rem 1.4rem;
            font-size: 1.5rem;
            font-weight: bold;
            letter-spacing: 0.18em;
            font-family: 'Courier New', monospace;
        }}
        .badge-sell {{
            display: inline-block;
            background-color: rgba(218,54,51,0.18);
            color: {C_DANGER};
            border: 1px solid {C_DANGER};
            border-radius: 4px;
            padding: 0.35rem 1.4rem;
            font-size: 1.5rem;
            font-weight: bold;
            letter-spacing: 0.18em;
            font-family: 'Courier New', monospace;
        }}
        .badge-hold {{
            display: inline-block;
            background-color: rgba(56,139,253,0.18);
            color: {C_NEUTRAL};
            border: 1px solid {C_NEUTRAL};
            border-radius: 4px;
            padding: 0.35rem 1.4rem;
            font-size: 1.5rem;
            font-weight: bold;
            letter-spacing: 0.18em;
            font-family: 'Courier New', monospace;
        }}
    </style>
    """, unsafe_allow_html=True)


def style_chart(ax, fig, title=""):
    """Apply consistent dark theme to any matplotlib axes object."""
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    ax.set_title(title, color=C_TEXT, fontsize=10, pad=10, fontfamily="monospace")
    ax.set_xlabel(ax.get_xlabel(), color=C_MUTED, fontsize=8, fontfamily="monospace")
    ax.set_ylabel(ax.get_ylabel(), color=C_MUTED, fontsize=8, fontfamily="monospace")
    ax.tick_params(colors=C_MUTED, labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_BORDER)
    ax.spines["bottom"].set_color(C_BORDER)
    ax.grid(True, color=C_BORDER, linewidth=0.5, alpha=0.5)
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_facecolor(C_SURFACE)
        legend.get_frame().set_edgecolor(C_BORDER)
        for text in legend.get_texts():
            text.set_color(C_TEXT)
            text.set_fontsize(7)


# ─── 3. LOGIC / CALCULATION FUNCTIONS ──────────────────────────────────────────
# Pure calculation functions with no Streamlit code.
# Each function takes data in and returns processed data out.

# ── Part 1: Individual Stock ──

def fetch_stock_data(ticker: str):
    """
    DATA: Download 6 months of daily OHLCV data for a single ticker.
    Returns the yfinance Ticker object and the historical DataFrame.
    """
    stock = yf.Ticker(ticker)
    df    = stock.history(period="6mo")
    return stock, df


def calc_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    PROCESSING: Add 20-day and 50-day simple moving averages to the DataFrame.
    Moving averages smooth out daily noise to reveal the price trend.
    """
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    return df


def calc_trend(df: pd.DataFrame) -> str:
    """
    PROCESSING: Determine if the stock is in an uptrend, downtrend, or mixed trend.
    Rule: Price > MA20 > MA50 = uptrend. Price < MA20 < MA50 = downtrend. Else = mixed.
    """
    price = df["Close"].iloc[-1]
    ma20  = df["MA20"].iloc[-1]
    ma50  = df["MA50"].iloc[-1]
    if price > ma20 > ma50:
        return "Strong Uptrend"
    elif price < ma20 < ma50:
        return "Strong Downtrend"
    return "Mixed Trend"


def calc_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    PROCESSING: Calculate the 14-day Relative Strength Index (RSI).
    RSI measures how fast prices are moving up vs down.
    Formula: RSI = 100 - (100 / (1 + average_gain / average_loss))
    Above 70 = overbought. Below 30 = oversold.
    """
    delta    = df["Close"].diff()          # day-over-day price change
    gain     = delta.clip(lower=0)         # keep only positive (up) days
    loss     = -delta.clip(upper=0)        # keep only negative (down) days, flipped positive
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs       = avg_gain / avg_loss         # relative strength ratio
    df["RSI"] = 100 - (100 / (1 + rs))    # convert to 0-100 scale
    return df


def interpret_rsi(rsi_value: float) -> str:
    """PROCESSING: Translate an RSI number into a human-readable signal."""
    if rsi_value > 70:
        return "Overbought (Possible Sell)"
    elif rsi_value < 30:
        return "Oversold (Possible Buy)"
    return "Neutral"


def calc_volatility(df: pd.DataFrame) -> float:
    """
    PROCESSING: Calculate 20-day annualized volatility as a percentage.
    Daily return std * sqrt(252 trading days) gives the annualized figure.
    Higher = more price movement risk.
    """
    daily_returns = df["Close"].pct_change()
    return daily_returns.rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100


def classify_volatility(volatility: float) -> str:
    """PROCESSING: Label volatility level as High (>40%), Medium (25-40%), or Low (<25%)."""
    if volatility > 40:
        return "High"
    elif volatility >= 25:
        return "Medium"
    return "Low"


def build_recommendation(ticker, trend, rsi_value, vol_level, volatility):
    """
    PROCESSING: Combine trend, RSI, and volatility into one BUY / SELL / HOLD call.
    Returns a (recommendation_string, explanation_string) tuple.
    """
    if trend == "Strong Uptrend" and rsi_value < 70:
        rec = "BUY"
        exp = (
            f"{ticker} is in a strong uptrend (Price > 20MA > 50MA) "
            f"and RSI is not overbought ({rsi_value:.1f}). "
            f"Volatility is {vol_level.lower()} ({volatility:.1f}%). "
            "Conditions support a buy."
        )
    elif trend == "Strong Downtrend" or rsi_value > 70:
        rec = "SELL"
        exp = (
            f"{ticker} shows a downtrend or overbought RSI ({rsi_value:.1f}). "
            f"Trend: {trend}. Volatility: {vol_level.lower()} ({volatility:.1f}%). "
            "Consider reducing exposure."
        )
    else:
        rec = "HOLD"
        exp = (
            f"Mixed signals for {ticker}. Trend: {trend}. "
            f"RSI: {rsi_value:.1f} (Neutral). "
            f"Volatility: {vol_level.lower()} ({volatility:.1f}%). "
            "Wait for a clearer signal."
        )
    return rec, exp


# ── Part 2: Portfolio ──

def fetch_portfolio_data(tickers: list, benchmark: str) -> pd.DataFrame:
    """
    DATA: Download 1 year of daily closing prices for all portfolio stocks and the benchmark.
    Returns a DataFrame where each column is one ticker symbol.
    """
    all_tickers = tickers + [benchmark]
    raw = yf.download(all_tickers, period="1y", progress=False)["Close"]
    return raw


def calc_portfolio_returns(raw: pd.DataFrame, tickers: list, weights: list, benchmark: str):
    """
    PROCESSING: Compute daily and cumulative returns for the portfolio and benchmark.
    Portfolio daily return = dot product of stock returns and their weights.
    Cumulative return = (1 + daily_return).cumprod() shows growth of $1 invested.
    """
    returns              = raw.pct_change().dropna()          # daily % changes for all assets
    portfolio_returns    = returns[tickers].dot(weights)      # weighted portfolio daily return
    benchmark_returns    = returns[benchmark]                  # benchmark daily return
    portfolio_cumulative = (1 + portfolio_returns).cumprod()  # how $1 grew over time
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    return returns, portfolio_returns, benchmark_returns, portfolio_cumulative, benchmark_cumulative


def calc_performance_metrics(portfolio_returns, benchmark_returns, portfolio_cumulative, benchmark_cumulative):
    """
    PROCESSING: Calculate total return, annualized volatility, and Sharpe ratio.
    Sharpe ratio = (annualized return - risk-free rate) / annualized volatility.
    A higher Sharpe means more return earned per unit of risk taken.
    Assumes 0% risk-free rate for simplicity.
    """
    total_return           = (portfolio_cumulative.iloc[-1] - 1) * 100
    benchmark_total_return = (benchmark_cumulative.iloc[-1] - 1) * 100
    outperformance         = total_return - benchmark_total_return
    port_volatility        = portfolio_returns.std() * np.sqrt(252) * 100
    bench_volatility       = benchmark_returns.std() * np.sqrt(252) * 100
    annualized_return      = portfolio_returns.mean() * 252
    sharpe_ratio           = annualized_return / (portfolio_returns.std() * np.sqrt(252))
    return total_return, benchmark_total_return, outperformance, port_volatility, bench_volatility, sharpe_ratio


def build_interpretation(benchmark, total_return, benchmark_total_return,
                          outperformance, port_volatility, bench_volatility, sharpe_ratio) -> list:
    """
    PROCESSING: Convert all numeric results into plain-English sentences.
    Returns a list of three interpretation strings.
    """
    lines = []
    if outperformance > 0:
        lines.append(f"The portfolio outperformed {benchmark} by {outperformance:.2f}%.")
    else:
        lines.append(f"The portfolio underperformed {benchmark} by {abs(outperformance):.2f}%.")
    if port_volatility > bench_volatility:
        lines.append(f"The portfolio carried more risk than the benchmark ({port_volatility:.2f}% vs {bench_volatility:.2f}% volatility).")
    else:
        lines.append(f"The portfolio carried less risk than the benchmark ({port_volatility:.2f}% vs {bench_volatility:.2f}% volatility).")
    if sharpe_ratio > 1:
        lines.append(f"A Sharpe ratio of {sharpe_ratio:.2f} suggests good risk-adjusted returns.")
    elif sharpe_ratio > 0:
        lines.append(f"A Sharpe ratio of {sharpe_ratio:.2f} suggests modest risk-adjusted returns.")
    else:
        lines.append(f"A Sharpe ratio of {sharpe_ratio:.2f} suggests returns did not compensate for the risk taken.")
    return lines


# ─── 4. CHART FUNCTIONS ────────────────────────────────────────────────────────
# Each function builds one matplotlib figure and returns it.
# No st.pyplot() calls here, keeping charts separate from display logic.

def chart_price_ma(df, ticker):
    """DISPLAY: Closing price line chart with 20-day and 50-day moving averages."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(df.index, df["Close"], label="Close Price", color=CH_PRICE, linewidth=1.5)
    ax.plot(df.index, df["MA20"],  label="20-Day MA",   color=CH_MA20,  linestyle="--", linewidth=1)
    ax.plot(df.index, df["MA50"],  label="50-Day MA",   color=CH_MA50,  linestyle="--", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left")
    style_chart(ax, fig, title=f"{ticker}  |  Price & Moving Averages")
    fig.tight_layout()
    return fig


def chart_rsi(df, ticker):
    """DISPLAY: RSI line chart with shaded overbought (>70) and oversold (<30) zones."""
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.plot(df.index, df["RSI"], label="RSI (14)", color=CH_RSI, linewidth=1.5)
    ax.axhline(70, color=C_DANGER, linestyle="--", linewidth=0.9, label="Overbought (70)")
    ax.axhline(30, color=C_ACCENT, linestyle="--", linewidth=0.9, label="Oversold (30)")
    ax.fill_between(df.index, 70, 100, alpha=0.07, color=C_DANGER)   # red zone above 70
    ax.fill_between(df.index, 0,  30,  alpha=0.07, color=C_ACCENT)   # green zone below 30
    ax.set_ylim(0, 100)
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI")
    ax.legend(loc="upper left")
    style_chart(ax, fig, title=f"{ticker}  |  RSI (14-Day)")
    fig.tight_layout()
    return fig


def chart_cumulative_returns(portfolio_cumulative, benchmark_cumulative, benchmark):
    """DISPLAY: Cumulative return of portfolio vs benchmark (growth of $1 invested)."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(portfolio_cumulative.index, portfolio_cumulative, label="Portfolio", color=CH_PORT,  linewidth=2)
    ax.plot(benchmark_cumulative.index, benchmark_cumulative, label=benchmark,   color=CH_BENCH, linewidth=1.5, linestyle="--")
    ax.axhline(1.0, color=C_BORDER, linewidth=0.8, linestyle=":")   # break-even reference line
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    style_chart(ax, fig, title="Portfolio vs Benchmark  |  Cumulative Return")
    fig.tight_layout()
    return fig


def chart_individual_returns(returns, tickers):
    """DISPLAY: Bar chart of each stock's total 1-year return. Green = positive, red = negative."""
    individual_returns = ((1 + returns[tickers]).cumprod().iloc[-1] - 1) * 100
    colors = [C_ACCENT if v >= 0 else C_DANGER for v in individual_returns.values]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(individual_returns.index, individual_returns.values, color=colors, width=0.5)
    ax.axhline(0, color=C_MUTED, linewidth=0.8)
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Return (%)")
    # Add percentage labels above / below each bar
    for i, (t, val) in enumerate(individual_returns.items()):
        offset = 1.0 if val >= 0 else -3.5
        ax.text(i, val + offset, f"{val:.1f}%",
                ha="center", fontsize=8, fontfamily="monospace",
                color=C_ACCENT if val >= 0 else C_DANGER)
    style_chart(ax, fig, title="Individual Stock Returns (1 Year)")
    fig.tight_layout()
    return fig


# ─── 5. DISPLAY / UI FUNCTIONS ─────────────────────────────────────────────────
# These functions control what Streamlit renders on screen.
# All st.* calls live here. No calculation logic in this section.

def ui_step_header(step_num: int, title: str):
    """DISPLAY: Render a labeled step header with a divider line below it."""
    st.markdown(f"<p class='step-label'>Step {step_num}</p>", unsafe_allow_html=True)
    st.subheader(title)
    st.markdown(
        f"<hr style='border:none; border-top:1px solid {C_BORDER}; margin: 0.2rem 0 1rem 0'>",
        unsafe_allow_html=True
    )


def ui_badge(recommendation: str, explanation: str):
    """DISPLAY: Show a colored BUY / SELL / HOLD badge followed by the explanation text."""
    badge_class = {
        "BUY":  "badge-buy",
        "SELL": "badge-sell",
        "HOLD": "badge-hold"
    }.get(recommendation, "badge-hold")
    st.markdown(f"<div class='{badge_class}'>{recommendation}</div>", unsafe_allow_html=True)
    st.write("")
    st.write(explanation)


def ui_part1(ticker: str):
    """
    DISPLAY: Render all 5 steps for Part 1 (Individual Stock Analysis).
    Calls calculation functions to get the data, then renders the results.
    """

    # ── Step 1: Data Collection ───────────────────────────────────────────────
    ui_step_header(1, "Data Collection")
    stock, df = fetch_stock_data(ticker)

    if df.empty:
        st.error(f"No data found for '{ticker}'. Check the ticker symbol.")
        st.stop()

    st.success(f"6 months of daily data loaded for {ticker}")
    with st.expander("View Raw Data (Last 10 Rows)", expanded=False):
        st.dataframe(df[["Open", "High", "Low", "Close", "Volume"]].tail(10),
                     use_container_width=True)

    # ── Step 2: Trend Analysis ────────────────────────────────────────────────
    ui_step_header(2, "Trend Analysis  |  Moving Averages")
    df    = calc_moving_averages(df)
    trend = calc_trend(df)
    price = df["Close"].iloc[-1]
    ma20  = df["MA20"].iloc[-1]
    ma50  = df["MA50"].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${price:.2f}")
    col2.metric("20-Day MA",     f"${ma20:.2f}")
    col3.metric("50-Day MA",     f"${ma50:.2f}")
    col4.metric("Trend Signal",  trend)

    st.pyplot(chart_price_ma(df, ticker), use_container_width=True)

    # ── Step 3: RSI Momentum ──────────────────────────────────────────────────
    ui_step_header(3, "Momentum  |  14-Day RSI")
    df        = calc_rsi(df)
    rsi_value = df["RSI"].iloc[-1]
    rsi_sig   = interpret_rsi(rsi_value)

    col1, col2 = st.columns(2)
    col1.metric("RSI (14-Day)", f"{rsi_value:.2f}")
    col2.metric("Signal",       rsi_sig)

    st.pyplot(chart_rsi(df, ticker), use_container_width=True)

    # ── Step 4: Volatility ────────────────────────────────────────────────────
    ui_step_header(4, "Volatility  |  20-Day Annualized")
    volatility = calc_volatility(df)
    vol_level  = classify_volatility(volatility)

    col1, col2 = st.columns(2)
    col1.metric("Annualized Volatility", f"{volatility:.2f}%")
    col2.metric("Volatility Level",      vol_level)

    # ── Step 5: Recommendation ────────────────────────────────────────────────
    ui_step_header(5, "Trading Recommendation")
    rec, exp = build_recommendation(ticker, trend, rsi_value, vol_level, volatility)
    ui_badge(rec, exp)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown(f"<hr style='border-top:1px solid {C_BORDER}; margin-top:2rem'>", unsafe_allow_html=True)
    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label="Download Stock Data as CSV",
        data=csv,
        file_name=f"{ticker}_stock_analysis.csv",
        mime="text/csv"
    )


def ui_part2(tickers_input: str, weights_input: str, benchmark: str):
    """
    DISPLAY: Render all 6 steps for Part 2 (Portfolio Performance Dashboard).
    Validates inputs first, then calls calculation functions and renders results.
    """

    # Parse comma-separated inputs from the sidebar
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    weights = [float(w.strip()) for w in weights_input.split(",")]

    # Input validation before any data downloads
    if len(tickers) != 5:
        st.error("Enter exactly 5 stock tickers.")
        st.stop()
    if len(weights) != 5:
        st.error("Enter exactly 5 weights.")
        st.stop()
    if abs(sum(weights) - 1.0) > 0.01:
        st.error(f"Weights must sum to 1.00. Current sum: {sum(weights):.2f}")
        st.stop()

    # ── Step 1: Portfolio Setup ───────────────────────────────────────────────
    ui_step_header(1, "Portfolio Setup")
    weight_df = pd.DataFrame({"Ticker": tickers, "Weight": [f"{w:.0%}" for w in weights]})
    col1, _ = st.columns([1, 3])
    col1.dataframe(weight_df, use_container_width=True, hide_index=True)

    # ── Step 2: Data Collection ───────────────────────────────────────────────
    ui_step_header(2, "Data Collection  |  1 Year")
    raw = fetch_portfolio_data(tickers, benchmark)

    if raw.empty:
        st.error("Could not download data. Check your ticker symbols.")
        st.stop()

    st.success(f"1 year of closing prices loaded for: {', '.join(tickers + [benchmark])}")
    with st.expander("View Closing Prices (Last 5 Rows)", expanded=False):
        st.dataframe(raw.tail(5), use_container_width=True)

    # ── Step 3: Return Calculations ───────────────────────────────────────────
    ui_step_header(3, "Return Calculations  |  Portfolio vs Benchmark")
    returns, portfolio_returns, benchmark_returns, portfolio_cumulative, benchmark_cumulative = \
        calc_portfolio_returns(raw, tickers, weights, benchmark)

    st.pyplot(chart_cumulative_returns(portfolio_cumulative, benchmark_cumulative, benchmark),
              use_container_width=True)

    # ── Step 4: Performance Metrics ───────────────────────────────────────────
    ui_step_header(4, "Performance Metrics")
    total_return, benchmark_total_return, outperformance, \
    port_volatility, bench_volatility, sharpe_ratio = \
        calc_performance_metrics(portfolio_returns, benchmark_returns,
                                  portfolio_cumulative, benchmark_cumulative)

    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio Return",  f"{total_return:.2f}%")
    col2.metric("Benchmark Return",  f"{benchmark_total_return:.2f}%",
                delta=f"{outperformance:+.2f}% vs benchmark")
    col3.metric("Outperformance",    f"{outperformance:+.2f}%")

    st.write("")  # vertical spacer

    col4, col5, col6 = st.columns(3)
    col4.metric("Portfolio Volatility",  f"{port_volatility:.2f}%")
    col5.metric("Benchmark Volatility",  f"{bench_volatility:.2f}%")
    col6.metric("Sharpe Ratio",          f"{sharpe_ratio:.2f}")

    # ── Step 5: Interpretation ────────────────────────────────────────────────
    ui_step_header(5, "Interpretation")
    lines = build_interpretation(benchmark, total_return, benchmark_total_return,
                                  outperformance, port_volatility, bench_volatility, sharpe_ratio)
    for line in lines:
        st.write(f"• {line}")

    # ── Step 6: Individual Stock Returns ──────────────────────────────────────
    st.write("")
    ui_step_header(6, "Individual Stock Returns")
    st.pyplot(chart_individual_returns(returns, tickers), use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown(f"<hr style='border-top:1px solid {C_BORDER}; margin-top:2rem'>", unsafe_allow_html=True)
    combined = pd.DataFrame({"Portfolio": portfolio_returns, benchmark: benchmark_returns})
    csv = combined.to_csv().encode("utf-8")
    st.download_button(
        label="Download Portfolio Returns as CSV",
        data=csv,
        file_name="portfolio_returns.csv",
        mime="text/csv"
    )


# ─── 6. MAIN APP ENTRY POINT ───────────────────────────────────────────────────
# Everything starts here. Page config, theme, sidebar navigation, and section routing.

def main():
    # Page config must be the very first Streamlit call
    st.set_page_config(
        page_title="FIN 330 Dashboard",
        page_icon="📈",
        layout="wide"
    )

    # Apply CSS theme to the whole app
    apply_theme()

    # ── App Header ────────────────────────────────────────────────────────────
    st.markdown(
        f"<h1 style='color:{C_GOLD}; font-family:Courier New; letter-spacing:0.06em;'>"
        "FIN 330  |  Stock Analytics Dashboard</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='color:{C_MUTED}; font-family:Courier New; margin-top:-0.5rem;'>"
        "Analyze individual stocks and evaluate a multi-asset portfolio using real Yahoo Finance data.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<hr style='border-top:1px solid {C_BORDER}; margin-bottom:1.5rem'>",
        unsafe_allow_html=True
    )

    # ── Sidebar: Navigation ───────────────────────────────────────────────────
    st.sidebar.markdown(
        f"<h2 style='color:{C_GOLD}; font-family:Courier New; font-size:1rem; "
        f"letter-spacing:0.1em;'>FIN 330</h2>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown(f"<hr style='border-top:1px solid {C_BORDER}'>", unsafe_allow_html=True)

    section = st.sidebar.radio(
        "Navigate",
        ["Part 1: Stock Analysis", "Part 2: Portfolio Dashboard"],
        label_visibility="collapsed"
    )
    st.sidebar.markdown(f"<hr style='border-top:1px solid {C_BORDER}'>", unsafe_allow_html=True)

    # ── Part 1 ────────────────────────────────────────────────────────────────
    if section == "Part 1: Stock Analysis":
        st.sidebar.markdown(
            f"<p style='color:{C_MUTED}; font-size:0.72rem; text-transform:uppercase; "
            f"letter-spacing:0.1em;'>Stock Settings</p>",
            unsafe_allow_html=True
        )
        ticker = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()
        st.sidebar.write("")
        run = st.sidebar.button("Run Stock Analysis")

        st.header("Part 1: Individual Stock Analysis")

        if run:
            ui_part1(ticker)
        else:
            st.markdown(
                f"<p style='color:{C_MUTED}; margin-top:1rem;'>"
                "Enter a ticker in the sidebar and click Run Stock Analysis to begin.</p>",
                unsafe_allow_html=True
            )

    # ── Part 2 ────────────────────────────────────────────────────────────────
    elif section == "Part 2: Portfolio Dashboard":
        st.sidebar.markdown(
            f"<p style='color:{C_MUTED}; font-size:0.72rem; text-transform:uppercase; "
            f"letter-spacing:0.1em;'>Portfolio Settings</p>",
            unsafe_allow_html=True
        )
        tickers_input = st.sidebar.text_input("5 Tickers (comma-separated)", "AAPL, MSFT, JPM, AMZN, NVDA")
        weights_input = st.sidebar.text_input("Weights (must sum to 1.00)",   "0.20, 0.20, 0.20, 0.20, 0.20")
        benchmark     = st.sidebar.text_input("Benchmark ETF", "SPY").upper()
        st.sidebar.write("")
        run = st.sidebar.button("Run Portfolio Analysis")

        st.header("Part 2: Portfolio Performance Dashboard")

        if run:
            ui_part2(tickers_input, weights_input, benchmark)
        else:
            st.markdown(
                f"<p style='color:{C_MUTED}; margin-top:1rem;'>"
                "Enter your tickers, weights, and benchmark in the sidebar, then click Run Portfolio Analysis.</p>",
                unsafe_allow_html=True
            )


# Run the app
if __name__ == "__main__":
    main()
