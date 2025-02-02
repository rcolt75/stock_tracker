import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np
from plotly.subplots import make_subplots
import requests
from datetime import datetime

# Define popular stocks
POPULAR_STOCKS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'TSLA': 'Tesla Inc.',
    'NVDA': 'NVIDIA Corporation',
    'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.',
    'WMT': 'Walmart Inc.',
    'DIS': 'The Walt Disney Company',
    'NFLX': 'Netflix Inc.',
    'PYPL': 'PayPal Holdings Inc.',
    'INTC': 'Intel Corporation',
    'AMD': 'Advanced Micro Devices Inc.'
}

# Function to calculate technical indicators
def calculate_indicators(df):
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Calculate moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # Calculate Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    # Calculate Stochastic Oscillator
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    return df

def get_ai_analysis(symbol, hist_data, info):
    analysis = {
        'Recommendation': 'Hold',  # Default recommendation
        'Technical_Summary': {
            'Price_Trend': 'Neutral',
            'Momentum': 'Neutral',
            'RSI_Status': 'Neutral',
            'MACD_Signal': 'Neutral'
        },
        'Trading_Signals': [],
        'Risk_Factors': []
    }
    
    try:
        # Technical Analysis
        current_price = hist_data['Close'].iloc[-1]
        sma_20 = hist_data['SMA_20'].iloc[-1]
        sma_50 = hist_data['SMA_50'].iloc[-1]
        rsi = hist_data['RSI'].iloc[-1]
        macd = hist_data['MACD'].iloc[-1]
        signal = hist_data['Signal_Line'].iloc[-1]
        
        # Price Trend Analysis
        if sma_20 > sma_50:
            analysis['Technical_Summary']['Price_Trend'] = 'Upward'
        else:
            analysis['Technical_Summary']['Price_Trend'] = 'Downward'
        
        # Momentum Analysis
        if abs(macd) > abs(signal):
            analysis['Technical_Summary']['Momentum'] = 'Strong'
        else:
            analysis['Technical_Summary']['Momentum'] = 'Weak'
        
        # RSI Analysis
        if rsi > 70:
            analysis['Technical_Summary']['RSI_Status'] = 'Overbought'
        elif rsi < 30:
            analysis['Technical_Summary']['RSI_Status'] = 'Oversold'
        else:
            analysis['Technical_Summary']['RSI_Status'] = 'Neutral'
        
        # MACD Analysis
        if macd > signal:
            analysis['Technical_Summary']['MACD_Signal'] = 'Bullish'
        else:
            analysis['Technical_Summary']['MACD_Signal'] = 'Bearish'
        
        # Generate trading signals
        if current_price > sma_20 and sma_20 > sma_50:
            analysis['Trading_Signals'].append("Strong upward trend detected")
        
        if rsi < 30:
            analysis['Trading_Signals'].append("RSI indicates oversold conditions - potential buying opportunity")
        elif rsi > 70:
            analysis['Trading_Signals'].append("RSI indicates overbought conditions - consider taking profits")
        
        if macd > signal:
            analysis['Trading_Signals'].append("MACD shows bullish momentum")
        else:
            analysis['Trading_Signals'].append("MACD shows bearish momentum")
        
        # Risk Analysis
        if info:
            try:
                pe_ratio = info.get('trailingPE', 0)
                if pe_ratio > 30:
                    analysis['Risk_Factors'].append("High P/E ratio indicates potential overvaluation")
                
                beta = info.get('beta', 0)
                if beta > 1.5:
                    analysis['Risk_Factors'].append("High beta indicates increased volatility risk")
                
                debt_to_equity = info.get('debtToEquity', 0)
                if debt_to_equity > 2:
                    analysis['Risk_Factors'].append("High debt-to-equity ratio indicates financial risk")
            except:
                pass
        
        # Generate final recommendation
        score = 0
        if analysis['Technical_Summary']['Price_Trend'] == 'Upward':
            score += 1
        if analysis['Technical_Summary']['Momentum'] == 'Strong':
            score += 1
        if analysis['Technical_Summary']['RSI_Status'] == 'Oversold':
            score += 2
        elif analysis['Technical_Summary']['RSI_Status'] == 'Overbought':
            score -= 2
        if analysis['Technical_Summary']['MACD_Signal'] == 'Bullish':
            score += 1
        
        if score >= 3:
            analysis['Recommendation'] = 'Strong Buy'
        elif score == 2:
            analysis['Recommendation'] = 'Buy'
        elif score == -2:
            analysis['Recommendation'] = 'Sell'
        elif score <= -3:
            analysis['Recommendation'] = 'Strong Sell'
        else:
            analysis['Recommendation'] = 'Hold'
            
    except Exception as e:
        # If any error occurs, return the default neutral analysis
        pass
    
    return analysis

# Initialize session state for alerts
if 'alerts' not in st.session_state:
    st.session_state.alerts = {}

# Initialize session state for chat messages if not exists
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

def clear_input():
    st.session_state.user_input = ""

# LM Studio configuration
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_KEY = "lm-studio"

def get_stock_chat_response(prompt, symbol, hist_data, info):
    """Generate a response using LM Studio for stock-related questions."""
    try:
        # Debug: Show we're starting the request
        st.write("Preparing request to LM Studio...")
        
        # Get latest stock data
        current_price = hist_data['Close'].iloc[-1]
        price_change = ((current_price - hist_data['Close'].iloc[-2]) / hist_data['Close'].iloc[-2] * 100)
        
        # Get technical indicators
        rsi = hist_data['RSI'].iloc[-1]
        macd = hist_data['MACD'].iloc[-1]
        signal = hist_data['Signal_Line'].iloc[-1]
        sma_20 = hist_data['SMA_20'].iloc[-1]
        
        # Create context for the AI
        context = f"""
        Current stock data for {symbol}:
        - Price: ${current_price:.2f} ({price_change:+.2f}%)
        - RSI: {rsi:.2f}
        - MACD: {macd:.2f}
        - Signal Line: {signal:.2f}
        - 20-day SMA: ${sma_20:.2f}
        - Trading Volume: {hist_data['Volume'].iloc[-1]:,.0f}
        
        Additional Information:
        - Market Cap: ${info.get('marketCap', 0)/1e9:.2f}B
        - P/E Ratio: {info.get('trailingPE', 'N/A')}
        - Beta: {info.get('beta', 'N/A')}
        """
        
        # Create the message for LM Studio
        messages = [
            {
                "role": "system",
                "content": f"""You are a knowledgeable stock market assistant. 
                Analyze the provided stock data and answer questions about {symbol}.
                Always provide clear, concise responses based on the data.
                Include relevant numbers and percentages.
                If making predictions or recommendations, always include a disclaimer about financial advice."""
            },
            {
                "role": "user",
                "content": f"Here's the current data:\n{context}\n\nUser's question: {prompt}"
            }
        ]
        
        # Debug: Show the request payload
        st.write("Sending request to LM Studio...")
        
        # Prepare headers and payload
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LM_STUDIO_KEY}"
        }
        
        payload = {
            "model": "deepseek-r1-distill-qwen-7b",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 300,
            "top_p": 0.9,
            "stream": False
        }
        
        # Make request to LM Studio chat completions endpoint
        try:
            st.write(f"Request URL: {LM_STUDIO_URL}/chat/completions")
            
            response = requests.post(
                f"{LM_STUDIO_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Debug: Show response status
            st.write(f"Response status code: {response.status_code}")
            st.write("Raw response:", response.text)
            
            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    return response_data['choices'][0]['message']['content']
            
            return "I apologize, but I couldn't generate a response. Please try asking your question differently."
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to LM Studio: {str(e)}")
            return "I apologize, but I'm having trouble connecting to the AI service. Please make sure LM Studio is running and try again."
            
    except Exception as e:
        st.error(f"Error in chat response: {str(e)}")
        return f"I apologize, but I encountered an error while processing your question about {symbol}. Please try asking something else."

def get_stock_news(symbol):
    try:
        stock = yf.Ticker(symbol)
        news = stock.news
        return news[:5]  # Return top 5 news items
    except:
        return []

def get_real_time_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        current_price = stock.info.get('regularMarketPrice', 0)
        change = stock.info.get('regularMarketChangePercent', 0)
        volume = stock.info.get('regularMarketVolume', 0)
        return current_price, change, volume
    except:
        return 0, 0, 0

def get_top_stocks_data():
    # List of top stocks by market cap
    top_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK-B', 'TSM', 'V', 'JPM']
    data = []
    
    for symbol in top_symbols:
        price, change, volume = get_real_time_price(symbol)
        data.append({
            'Symbol': symbol,
            'Price': price,
            'Change %': change,
            'Volume': volume
        })
    
    return pd.DataFrame(data)

# Set page config
st.set_page_config(page_title="Stock Tracker", layout="wide")

# Title and description
st.title("Stock Tracker")
st.markdown("Track your favorite stocks in real-time!")

# Sidebar for stock input and alerts
with st.sidebar:
    st.header("Stock Settings")
    
    # Add dropdown for popular stocks
    selected_stock = st.selectbox(
        "Select a Popular Stock",
        options=list(POPULAR_STOCKS.keys()),
        format_func=lambda x: f"{x} - {POPULAR_STOCKS[x]}"
    )
    
    # Add custom stock input
    custom_symbol = st.text_input("Or Enter Custom Stock Symbol:", "")
    
    # Use custom symbol if provided, otherwise use selected stock
    symbol = custom_symbol.upper() if custom_symbol else selected_stock
    
    period = st.selectbox(
        "Select Time Period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=2
    )
    
    # Price Alert Settings
    st.header("Price Alerts")
    with st.form("alert_form"):
        alert_price = st.number_input("Set Price Alert ($)", min_value=0.0, value=0.0, step=0.01)
        alert_condition = st.selectbox("Alert Condition", ["Above", "Below"])
        submitted = st.form_submit_button("Set Alert")
        
        if submitted and alert_price > 0:
            st.session_state.alerts[symbol] = {
                "price": alert_price,
                "condition": alert_condition
            }
            st.success(f"Alert set for {symbol} when price goes {alert_condition.lower()} ${alert_price}")

# Create three columns for the main content
col1, col2 = st.columns([2, 1])

with col1:
    if not symbol:
        st.error("Please enter a stock symbol")
        st.stop()

    try:
        # Initialize progress
        progress_text = st.empty()
        
        # Get stock data with minimal period first
        progress_text.info(f"Fetching data for {symbol}...")
        
        # Initialize stock
        stock = yf.Ticker(symbol)
        
        # Get 1 month of historical data by default
        hist = stock.history(period='1mo')
        if hist.empty:
            st.error(f"No historical data available for {symbol}")
            st.stop()
            
        # Calculate basic indicators
        progress_text.info("Calculating indicators...")
        hist = calculate_indicators(hist)
        
        # Get basic info without full company details
        info = {}
        try:
            info = {
                'symbol': symbol,
                'marketCap': stock.info.get('marketCap', 0),
                'trailingPE': stock.info.get('trailingPE', 0),
                'beta': stock.info.get('beta', 0)
            }
        except:
            pass
        
        # Generate basic analysis
        analysis = get_ai_analysis(symbol, hist, info)
        
        # Clear progress
        progress_text.empty()
        
        # Display current stock info
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2]
        price_change = ((current_price - prev_price) / prev_price * 100)
        
        # Display basic metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}%")
        with col2:
            st.metric("Day High", f"${hist['High'].iloc[-1]:.2f}")
        with col3:
            st.metric("Day Low", f"${hist['Low'].iloc[-1]:.2f}")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Technical Analysis", "AI Analysis", "Historical Data"])
        
        with tab1:
            # Simple price chart
            fig = go.Figure(data=[go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name=symbol
            )])
            
            fig.update_layout(
                title=f"{symbol} Stock Price",
                yaxis_title="Price (USD)",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Technical indicators
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RSI (14)", f"{hist['RSI'].iloc[-1]:.2f}")
                st.metric("MACD", f"{hist['MACD'].iloc[-1]:.2f}")
            with col2:
                st.metric("Signal Line", f"{hist['Signal_Line'].iloc[-1]:.2f}")
                st.metric("20-day SMA", f"${hist['SMA_20'].iloc[-1]:.2f}")
        
        with tab3:
            # AI Analysis
            st.subheader("AI Recommendation")
            st.write(f"**{analysis['Recommendation']}**")
            
            st.subheader("Technical Summary")
            for metric, value in analysis['Technical_Summary'].items():
                st.write(f"**{metric}:** {value}")
            
            if analysis['Trading_Signals']:
                st.subheader("Trading Signals")
                for signal in analysis['Trading_Signals']:
                    st.write(f"• {signal}")
        
        with tab4:
            st.dataframe(
                hist.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']]
                .style.format("{:.2f}")
            )
        
        # Chat interface (in main column, after tabs)
        st.markdown("---")
        st.subheader(f"💬 Chat about {symbol}")
        
        # Create a container for chat messages
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.chat_messages:
                if message["role"] == "user":
                    st.info(f"You: {message['content']}")
                else:
                    st.success(f"Assistant: {message['content']}")
        
        # Chat input using text_input and button
        chat_col1, chat_col2 = st.columns([4, 1])
        with chat_col1:
            user_input = st.text_input(
                f"Ask about {symbol}...", 
                key="user_input",
                on_change=clear_input,
                value=st.session_state.user_input
            )
        with chat_col2:
            send_button = st.button("Send", key="send_button")
        
        if send_button and user_input:
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            # Get and add assistant response
            response = get_stock_chat_response(user_input, symbol, hist, info)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            
            # Force a rerun to update the chat
            st.rerun()
        
        # Load additional data in the background if needed
        if 'full_info' not in st.session_state:
            st.session_state.full_info = {}
        
        if symbol not in st.session_state.full_info:
            try:
                with st.spinner("Loading additional company information..."):
                    full_info = stock.info
                    st.session_state.full_info[symbol] = full_info
            except:
                pass

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")

# Real-time Top 10 Stocks and News Feed
with col2:
    st.subheader("Top Stocks")
    top_stocks_df = get_top_stocks_data()
    st.dataframe(top_stocks_df)
    
    st.markdown("---")
    st.subheader("Recent News")
    if symbol:
        news = get_stock_news(symbol)
        if news:
            for article in news[:5]:  # Display top 5 news articles
                if isinstance(article, dict):
                    title = article.get('title', 'No title available')
                    description = article.get('description', 'No description available')
                    st.markdown(f"**{title}**")
                    st.write(description)
                    st.markdown("---")
        else:
            st.write("No recent news available")
