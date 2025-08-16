import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from textblob import TextBlob

# --- Feature & Model Logic ---

def prepare_features(df):
    df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Return_1d'] = df['Close'].pct_change().shift(-1)
    df['Target'] = (df['Return_1d'] > 0).astype(int)

    # Replace infinite with NaN and drop
    df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Clip outliers
    df['Volume_Change'] = df['Volume_Change'].clip(-1, 1)
    df['RSI'] = df['RSI'].clip(0, 100)

    features = ['SMA20', 'RSI', 'MACD', 'Volume_Change']
    return df, features

def train_model(df):
    df, features = prepare_features(df)
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy, features

def predict_next_day(model, df, features):
    latest_data = df[features].iloc[-1:].values
    pred = model.predict(latest_data)[0]
    conf = max(model.predict_proba(latest_data)[0])
    return pred, conf

def volume_analysis(df):
    latest_volume = df['Volume'].iloc[-1]
    avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
    if latest_volume > 1.5 * avg_volume:
        return "📈 Volume spike detected — possible strong move."
    else:
        return "Volume normal."

def fundamental_summary(info):
    summary = []
    pe = info.get('trailingPE')
    market_cap = info.get('marketCap')

    if pe is not None:
        summary.append(f"P/E Ratio: {pe:.2f}")
        summary.append("High P/E may indicate overvaluation." if pe > 25 else "P/E is within reasonable range.")
    if market_cap is not None:
        summary.append(f"Market Cap: ₹{market_cap / 1e9:.2f} Billion")

    return summary

def simple_news_sentiment(ticker):
    # Dummy sentiment based on ticker string for demo
    if ticker[0].upper() <= 'M':
        return "Positive", 0.7, "Ticker starts with early alphabet — dummy logic"
    else:
        return "Neutral", 0.5, "No major news detected — dummy logic"

# --- Sidebar UI ---

st.sidebar.title("📊 DeepTrend Controls")

ticker = st.sidebar.text_input("Enter stock ticker", "RELIANCE.NS")
time_range = st.sidebar.selectbox("Select trend period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "10y"])

st.sidebar.markdown("---")
show_fundamentals = st.sidebar.checkbox("Show Fundamental Data", value=True)
show_sentiment = st.sidebar.checkbox("Show News Sentiment", value=True)
show_price_chart = st.sidebar.checkbox("Show Price Chart", value=True)

# --- Main App ---

st.title("💹 DeepTrend - AI-Powered Stock Analysis")

if ticker:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=time_range)

        if hist.empty:
            st.error("No data found for this ticker.")
        else:
            st.subheader(f"{stock.info.get('longName', ticker)} ({ticker})")
            st.write(f"Latest Price: ₹{hist['Close'].iloc[-1]:.2f}")

            # ML Model Prediction
            if len(hist) < 30:
                st.warning(f"Not enough data ({len(hist)} rows) for ML prediction. Select a longer period.")
            else:
                try:
                    model, accuracy, features = train_model(hist)
                    pred, conf = predict_next_day(model, hist, features)
                    st.write(f"✅ ML Model Accuracy (backtest): {accuracy*100:.2f}%")
                    st.markdown(f"📌 **Next-Day Prediction:** {'Up 📈' if pred == 1 else 'Down 📉'}")
                    st.markdown(f"🧠 **Model Confidence:** {conf*100:.2f}%")
                except Exception as e:
                    st.error(f"ML model error: {e}")

            # Volume analysis
            volume_msg = volume_analysis(hist)
            st.write(f"🔍 Volume Analysis: {volume_msg}")

            # Fundamentals
            if show_fundamentals:
                fundamentals = fundamental_summary(stock.info)
                with st.expander("📘 Fundamental Data"):
                    for line in fundamentals:
                        st.write(f"- {line}")

            # Sentiment (Dummy)
            if show_sentiment:
                sentiment, sentiment_conf, explanation = simple_news_sentiment(ticker)
                st.write(f"📰 News Sentiment: {sentiment} (Confidence: {sentiment_conf*100:.0f}%)")
                with st.expander("Sentiment Explanation"):
                    st.write(explanation)

            # Price Chart
            if show_price_chart:
                st.line_chart(hist['Close'])

    except Exception as e:
        st.error(f"Error loading stock data: {e}")
