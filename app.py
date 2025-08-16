

import streamlit as st
import yfinance as yf

st.title("DeepTrend - Live Stock Analysis")

ticker = st.text_input("Enter stock ticker (e.g. RELIANCE.NS for India)", "RELIANCE.NS")

# Let user choose the time range
time_range = st.selectbox("Select trend period", ["1mo", "3mo", "6mo", "1y", "5y"])

if ticker:
    stock = yf.Ticker(ticker)
    hist = stock.history(period=time_range)
    
    if not hist.empty:
        latest_price = hist['Close'][-1]
        prev_close = hist['Close'][-2]
        price_change = latest_price - prev_close
        pct_change = (price_change / prev_close) * 100
        
        st.subheader(f"{stock.info.get('longName', ticker)} ({ticker})")
        st.write(f"Price: ₹{latest_price:.2f}")
        st.write(f"Change: ₹{price_change:.2f} ({pct_change:.2f}%)")
        
        st.line_chart(hist['Close'])
        
        recommendation = "BUY" if pct_change > 0 else "HOLD"
        confidence = "High" if pct_change > 0 else "Medium"
        
        st.markdown(f"**Recommendation:** {recommendation}")
        st.markdown(f"**AI Confidence:** {confidence}")
        
        with st.expander("Bull Case"):
            st.write("- Positive price momentum.")
            st.write("- Strong fundamentals.")
        
        with st.expander("Bear Case"):
            st.write("- Possible overvaluation after recent gains.")
            st.write("- Market volatility.")
        
        with st.expander("Key Risks"):
            st.write("- Economic changes.")
            st.write("- Regulatory environment.")
            
    else:
        st.error("No data found for this ticker.")
