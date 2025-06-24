# Crypto Trading Bot powered by Claude AI

A sophisticated cryptocurrency trading bot that uses Claude AI for intelligent trading decisions and supports both paper trading and live trading on the Robinhood platform. The bot features real-time market data, advanced technical analysis, and a comprehensive dashboard built with Streamlit.

## 🚀 Features

### Core Functionality
- **Real Claude AI Integration**: Uses Anthropic's Claude AI API for sophisticated market analysis and trading decisions
- **Real-time Market Data**: Fetches live cryptocurrency data using Yahoo Finance with 5-minute intervals
- **Paper Trading Mode**: Safe testing environment with virtual $10,000 starting balance
- **Live Trading Support**: Real Robinhood API integration for actual trading (when configured)
- **Advanced Technical Analysis**: RSI and MACD indicators with customizable parameters
- **SQLite Database**: Persistent storage for trade history, portfolio data, and market data caching
- **Thread-safe Operations**: Concurrent data processing with proper locking mechanisms

### Dashboard Features
- **Real-time Portfolio Tracking**: Live portfolio value updates and performance charts
- **Position Management**: Current holdings with profit/loss calculations
- **Trade History**: Complete log of all executed trades with reasoning
- **Market Analysis**: Interactive charts for price data and technical indicators
- **API Status Monitoring**: Real-time connection status for all services
- **Customizable Rules**: Easy configuration through JSON settings

### Risk Management
- **Position Limits**: Maximum investment per trade and total open positions
- **Stop-loss Protection**: Configurable stop-loss percentages
- **Portfolio Allocation**: Limits on single crypto and total portfolio exposure
- **Daily Trading Limits**: Maximum number of trades per day

## 📦 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd crypto-trading-bot
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure API credentials in `cash/rules.json`:**
```json
{
  "api_credentials": {
    "robinhood_username": "YOUR_ROBINHOOD_USERNAME_HERE",
    "robinhood_password": "YOUR_ROBINHOOD_PASSWORD_HERE", 
    "anthropic_api_key": "YOUR_ANTHROPIC_API_KEY_HERE"
  }
}
```

## 🎯 Usage

### Starting the Bot
```bash
streamlit run bot.py
```

### Dashboard Access
Open your browser to `http://localhost:8501` to access the trading dashboard.

### Trading Modes

**Paper Trading (Default):**
- Safe testing environment
- $10,000 virtual starting balance
- Real market data and analysis
- No actual money at risk

**Live Trading:**
- Set `"paper_trading": false` in rules.json
- Requires valid Robinhood credentials
- Uses real money - trade carefully!

## ⚙️ Configuration

The bot is configured through `cash/rules.json`. Here's the complete configuration structure:

```json
{
  "general_settings": {
    "paper_trading": true,
    "paper_trading_balance": 10000.00
  },
  "api_credentials": {
    "robinhood_username": "YOUR_ROBINHOOD_USERNAME_HERE",
    "robinhood_password": "YOUR_ROBINHOOD_PASSWORD_HERE",
    "anthropic_api_key": "YOUR_ANTHROPIC_API_KEY_HERE"
  },
  "trading_parameters": {
    "max_investment_per_trade": 100.00,
    "stop_loss_percentage": 5.0,
    "take_profit_percentage": 10.0,
    "max_daily_trades": 10,
    "max_open_positions": 5
  },
  "risk_management": {
    "max_portfolio_allocation_percentage": 20.0,
    "max_single_crypto_allocation_percentage": 5.0,
    "daily_loss_limit_percentage": 3.0
  },
  "cryptocurrencies": [
    "BTC", "ETH", "SOL", "ADA", "DOT"
  ],
  "trading_strategy": {
    "timeframe": "5m",
    "use_custom_strategy": true,
    "custom_strategy": "Buy when RSI is below 30 and MACD histogram is turning positive. Sell when RSI is above 70 and MACD histogram is turning negative. Use a 5% stop loss on all trades.",
    "indicators": [
      {
        "name": "RSI",
        "parameters": {
          "period": 14,
          "overbought": 70,
          "oversold": 30
        }
      },
      {
        "name": "MACD",
        "parameters": {
          "fast_period": 12,
          "slow_period": 26,
          "signal_period": 9
        }
      }
    ],
    "claude_ai": {
      "confidence_threshold": 0.7,
      "analysis_frequency": "hourly",
      "data_sources": ["price_data", "market_sentiment", "news"]
    }
  }
}
```

## 🔧 How It Works

### Data Flow
1. **Market Data Collection**: Fetches real-time 5-minute cryptocurrency data from Yahoo Finance
2. **Technical Analysis**: Calculates RSI and MACD indicators using the `ta` library
3. **AI Analysis**: Sends market data and indicators to Claude AI for trading recommendations
4. **Decision Making**: Compares Claude's confidence against the configured threshold
5. **Trade Execution**: Executes trades through Robinhood API (live) or paper trading simulation
6. **Data Storage**: Stores all trades and portfolio values in SQLite database

### Trading Logic
- **Buy Signal**: RSI < 30 (oversold) AND MACD histogram turning positive
- **Sell Signal**: RSI > 70 (overbought) AND MACD histogram turning negative
- **Hold**: All other conditions or insufficient confidence
- **Risk Management**: Automatic stop-loss and position sizing

### Claude AI Integration
The bot sends structured market data to Claude AI with prompts like:
```
Analyze the following cryptocurrency market data for BTC and provide a trading recommendation:

Current Market Data:
- Price: $45,230.50
- RSI: 28.5
- MACD Histogram: 0.0023
- Recent Price Trend: down

Trading Strategy: Buy when RSI is below 30 and MACD histogram is turning positive...

Please provide:
1. Action: "buy", "sell", or "hold"
2. Confidence: A number between 0.0 and 1.0
3. Reason: Brief explanation for your recommendation

Format your response as JSON:
{"action": "buy/sell/hold", "confidence": 0.0-1.0, "reason": "explanation"}
```

## 📊 Dashboard Sections

### Controls Panel
- **Start/Stop Bot**: Control bot execution
- **Trading Status**: Real-time bot status indicator
- **API Status**: Connection status for Claude AI and Robinhood
- **Trading Rules**: View current configuration

### Portfolio Overview
- **Portfolio Value Chart**: Historical performance tracking
- **Current Positions**: Open positions with P&L calculations
- **Paper Trading Balance**: Available virtual funds (paper trading mode)

### Trading Activity
- **Trade History**: Complete log of all executed trades
- **Trade Reasoning**: AI-generated explanations for each trade
- **Performance Metrics**: Win/loss ratios and returns

### Market Analysis
- **Price Charts**: Real-time 5-minute candlestick data
- **Technical Indicators**: RSI and MACD visualizations
- **Current Metrics**: Live indicator values and signals

## 🔒 Security & Safety

### API Key Management
- Store API keys in `cash/rules.json` (never commit to version control)
- Use environment variables for production deployments
- Rotate keys regularly

### Paper Trading First
- **Always start with paper trading** to test strategies
- Verify bot behavior before risking real money
- Test with different market conditions

### Risk Controls
- Set conservative position sizes initially
- Use stop-loss orders on all trades
- Monitor bot performance regularly
- Have manual override capabilities

## 🐛 Troubleshooting

### Common Issues

**"No market data available"**
- Check internet connection
- Verify cryptocurrency symbols in rules.json
- Yahoo Finance may have temporary outages

**"Claude AI connection failed"**
- Verify Anthropic API key is correct
- Check API key permissions and credits
- Bot will use fallback analysis if Claude is unavailable

**"Robinhood login failed"**
- Verify username/password are correct
- Check for 2FA requirements
- Robinhood may require additional verification

**Database errors**
- Delete `data/trading_bot.db` to reset database
- Check file permissions in data directory
- Ensure SQLite is properly installed

## 📈 Performance Optimization

### Market Data Caching
- Historical data is cached in SQLite database
- Reduces API calls and improves performance
- Automatic cache cleanup for old data

### Threading
- Market data fetching runs in separate threads
- UI updates don't block trading operations
- Thread-safe data access with locks

### Memory Management
- Limited historical data retention
- Efficient pandas operations
- Garbage collection for large datasets

## 🚨 Important Disclaimers

⚠️ **Trading Risks**
- Cryptocurrency trading involves substantial risk
- Past performance doesn't guarantee future results
- Never invest more than you can afford to lose
- This bot is for educational purposes

⚠️ **Technical Limitations**
- Market data delays may affect performance
- API rate limits may restrict operations
- Bot performance depends on market conditions
- No guarantee of profitable trades

⚠️ **Regulatory Compliance**
- Ensure compliance with local trading regulations
- Some jurisdictions restrict automated trading
- Tax implications may apply to trading profits
- Consult financial advisors as needed

## 📝 Development Notes

### Architecture
- **Backend**: Python with threading for concurrent operations
- **Database**: SQLite for persistence and caching
- **Frontend**: Streamlit for real-time dashboard
- **APIs**: Yahoo Finance (market data), Anthropic (AI), Robinhood (trading)

### Code Structure
- `CryptoTradingBot`: Main trading engine class
- `create_ui()`: Streamlit dashboard interface
- Database methods: SQLite operations for persistence
- API integrations: Separate methods for each service

### Future Enhancements
- Support for additional exchanges (Coinbase, Binance)
- More technical indicators (Bollinger Bands, Stochastic)
- Advanced AI models and strategies
- Mobile app interface
- Backtesting capabilities
- Portfolio optimization algorithms

## 📞 Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Review the logs in `trading_bot.log`
3. Ensure all dependencies are properly installed
4. Verify API credentials and permissions

---

**Happy Trading! 🚀📈**

*Remember: Start with paper trading, use proper risk management, and never invest more than you can afford to lose.*
