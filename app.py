import streamlit as st
import yfinance as yf
import pandas as pd
import ta

st.title("DeepTrend - Live Stock Analysis")

ticker = st.text_input("Enter stock ticker (e.g. RELIANCE.NS for India)", "RELIANCE.NS")
time_range = st.selectbox("Select trend period", ["1mo", "3mo", "6mo", "1y", "5y"])

if ticker:
    stock = yf.Ticker(ticker)
    hist = stock.history(period=time_range)

    if not hist.empty:
        # Calculate Indicators
        hist['SMA20'] = ta.trend.sma_indicator(hist['Close'], window=20)
        hist['SMA50'] = ta.trend.sma_indicator(hist['Close'], window=50)
        hist['RSI'] = ta.momentum.RSIIndicator(hist['Close'], window=14).rsi()
        macd = ta.trend.MACD(hist['Close'])
        hist['MACD'] = macd.macd()
        hist['MACD_Signal'] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(hist['Close'], window=20, window_dev=2)
        hist['BB_High'] = bb.bollinger_hband()
        hist['BB_Low'] = bb.bollinger_lband()

        # Price & performance
        latest_price = hist['Close'][-1]
        prev_close = hist['Close'][-2]
        price_change = latest_price - prev_close
        pct_change = (price_change / prev_close) * 100

        latest_rsi = hist['RSI'][-1]
        latest_macd = hist['MACD'][-1]
        latest_macd_signal = hist['MACD_Signal'][-1]

        st.subheader(f"{stock.info.get('longName', ticker)} ({ticker})")
        st.write(f"Price: ₹{latest_price:.2f}")
        st.write(f"Change: ₹{price_change:.2f} ({pct_change:.2f}%)")
        st.write(f"RSI (14): {latest_rsi:.2f}")

        st.line_chart(hist[['Close', 'SMA20', 'SMA50']])
        st.line_chart(hist[['MACD', 'MACD_Signal']])
        st.line_chart(hist[['Close', 'BB_High', 'BB_Low']])

        # === Indicator Interpretation === #
        # RSI
        if latest_rsi > 70:
            rsi_signal = "Overbought - possible reversal."
        elif latest_rsi < 30:
            rsi_signal = "Oversold - potential bounce."
        else:
            rsi_signal = "Neutral."

        # SMA
        sma_signal = "Bullish" if hist['SMA20'][-1] > hist['SMA50'][-1] else "Bearish"

        # MACD
        if latest_macd > latest_macd_signal:
            macd_signal = "MACD crossover suggests upward momentum (Buy signal)."
        else:
            macd_signal = "MACD crossover suggests downward momentum (Sell signal)."

        # Bollinger Bands
        if latest_price > hist['BB_High'][-1]:
            bb_signal = "Price is above upper Bollinger Band → Overbought."
        elif latest_price < hist['BB_Low'][-1]:
            bb_signal = "Price is below lower Bollinger Band → Oversold."
        else:
            bb_signal = "Price within bands → Stable."

        # === Recommendation Logic === #
        if latest_rsi < 30 and latest_macd > latest_macd_signal:
            recommendation = "BUY"
            confidence = "High"
            bull_case = [
                "RSI indicates oversold.",
                "MACD crossover supports upward trend.",
                "Price rebounding from lower Bollinger Band."
            ]
            bear_case = [
                "Might be short-lived recovery.",
                "Needs confirmation from volume and fundamentals."
            ]
        elif latest_rsi > 70 or latest_macd < latest_macd_signal:
            recommendation = "SELL"
            confidence = "Medium"
            bull_case = [
                "Momentum still strong despite overbought signals.",
                "Possible continuation if supported by fundamentals."
            ]
            bear_case = [
                "Overbought on RSI and Bollinger Bands.",
                "MACD shows weakening momentum."
            ]
        else:
            recommendation = "HOLD"
            confidence = "Medium"
            bull_case = [
                "No major warning signs.",
                "Sideways consolidation could lead to breakout."
            ]
            bear_case = [
                "Unclear trend direction.",
                "Momentum indicators not aligned."
            ]

        key_risks = [
            "Macro uncertainty may override technicals.",
            "False breakouts due to low volume or external events."
        ]

        # === Output to UI === #
        st.markdown(f"**Recommendation:** {recommendation}")
        st.markdown(f"**AI Confidence:** {confidence}")
        st.markdown(f"**RSI Insight:** {rsi_signal}")
        st.markdown(f"**SMA Insight:** {sma_signal}")
        st.markdown(f"**MACD Insight:** {macd_signal}")
        st.markdown(f"**Bollinger Bands Insight:** {bb_signal}")

        with st.expander("Bull Case"):
            for point in bull_case:
                st.write(f"- {point}")

        with st.expander("Bear Case"):
            for point in bear_case:
                st.write(f"- {point}")

        with st.expander("Key Risks"):
            for risk in key_risks:
                st.write(f"- {risk}")

    else:
        st.error("No data found for this ticker.")
