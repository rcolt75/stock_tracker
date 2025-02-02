# Stock Tracker App

A real-time stock tracking application built with Python and Streamlit. Track your favorite stocks, view historical data, analyze price trends with interactive charts, and chat about stocks using a local LLM powered by LM Studio.

## Features

- Real-time stock price tracking using Yahoo Finance API
- Interactive candlestick charts with technical indicators
- Historical price data and volume analysis
- Company information and fundamentals
- Multiple time period selections
- AI-powered stock chat using local LLM (LM Studio)
- Top stocks tracking
- Stock news integration
- Responsive design

## Data Sources

- **Stock Data**: Retrieved using the `yfinance` library, which provides:
  - Real-time and historical price data
  - Company information and fundamentals
  - Volume data
  - Corporate actions (splits, dividends)
- **Technical Indicators**: Calculated using `pandas_ta` library
- **News Data**: Fetched from relevant financial news APIs

## Installation

1. Clone this repository or download the files
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## LM Studio Setup

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Launch LM Studio and load your preferred local language model
3. Start the local server in LM Studio (default URL: http://localhost:1234/v1)
4. The app will automatically connect to LM Studio for AI-powered stock discussions

## Usage

1. Start LM Studio's local server
2. Run the application:
```bash
streamlit run app.py
```

3. Enter a stock symbol (e.g., AAPL for Apple, MSFT for Microsoft)
4. Select your desired time period
5. View the real-time stock data, charts, and indicators
6. Use the chat interface to ask questions about stocks - responses will be generated using your local LLM

## Technologies Used

- Python
- Streamlit - Web interface
- yfinance - Stock data retrieval
- Plotly - Interactive charts
- Pandas & pandas_ta - Data manipulation and technical analysis
- LM Studio - Local LLM integration for AI chat
- NumPy - Numerical computations
