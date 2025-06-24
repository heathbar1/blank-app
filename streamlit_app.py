import json
import os
import time
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import anthropic
import robin_stocks.robinhood as rh
from threading import Thread, Lock
import logging
import requests
from ta.momentum import RSIIndicator
from ta.trend import MACD
import yfinance as yf
import sqlite3
from typing import Dict, List, Optional, Tuple
import hashlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("crypto_trading_bot")

class CryptoTradingBot:
    def __init__(self, rules_path="cash/rules.json"):
        """Initialize the trading bot with rules from the specified JSON file."""
        self.rules_path = rules_path
        self.rules = self._load_rules()
        self.claude_client = None
        self.robinhood_logged_in = False
        self.is_running = False
        self.portfolio_history = []
        self.trade_history = []
        self.current_positions = {}
        self.market_data = {}
        self.data_lock = Lock()  # Thread safety
        
        # Initialize paper trading balance with proper error handling
        if self.rules:
            self.paper_trading_balance = self.rules.get("general_settings", {}).get("paper_trading_balance", 10000.0)
        else:
            self.paper_trading_balance = 10000.0
            logger.error("Rules not loaded - using default paper trading balance")
        
        self.paper_trading_positions = {}
        
        # Crypto symbol mapping for yfinance
        self.crypto_symbols = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD", 
            "SOL": "SOL-USD",
            "ADA": "ADA-USD",
            "DOT": "DOT-USD"
        }
        
        # Initialize data storage
        self.initialize_data_storage()
        
    def _load_rules(self):
        """Load trading rules from JSON file."""
        try:
            with open(self.rules_path, 'r') as f:
                rules = json.load(f)
            
            # Validate required fields
            required_fields = [
                "general_settings",
                "api_credentials", 
                "trading_parameters",
                "cryptocurrencies",
                "trading_strategy"
            ]
            
            for field in required_fields:
                if field not in rules:
                    raise ValueError(f"Missing required field: {field}")
            
            logger.info(f"Rules loaded successfully from {self.rules_path}")
            return rules
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            return None
            
    def initialize_data_storage(self):
        """Initialize SQLite database for portfolio and trade history."""
        try:
            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            
            # Initialize SQLite database
            self.db_path = "data/trading_bot.db"
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create portfolio history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    portfolio_value REAL,
                    paper_trading BOOLEAN
                )
            ''')
            
            # Create trade history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    crypto TEXT,
                    action TEXT,
                    price REAL,
                    quantity REAL,
                    value REAL,
                    reason TEXT,
                    paper_trading BOOLEAN
                )
            ''')
            
            # Create market data cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crypto TEXT,
                    timestamp DATETIME,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume REAL,
                    UNIQUE(crypto, timestamp)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def connect_to_robinhood(self):
        """Connect to Robinhood API using credentials from rules."""
        try:
            if not self.rules:
                logger.error("Rules not loaded - cannot connect to Robinhood")
                return False
                
            if self.rules["general_settings"]["paper_trading"]:
                logger.info("Paper trading mode - Robinhood connection simulated")
                self.robinhood_logged_in = True
                return True
            
            username = self.rules["api_credentials"]["robinhood_username"]
            password = self.rules["api_credentials"]["robinhood_password"]
            
            if username == "YOUR_ROBINHOOD_USERNAME_HERE" or password == "YOUR_ROBINHOOD_PASSWORD_HERE":
                logger.warning("Robinhood credentials not configured - using paper trading mode")
                self.rules["general_settings"]["paper_trading"] = True
                self.robinhood_logged_in = True
                return True
            
            # Attempt to login to Robinhood
            login_result = rh.login(username, password)
            
            if login_result:
                logger.info("Connected to Robinhood API successfully")
                self.robinhood_logged_in = True
                return True
            else:
                logger.error("Failed to login to Robinhood")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to Robinhood: {e}")
            return False
    
    def connect_to_claude_ai(self):
        """Initialize connection to Claude AI."""
        try:
            if not self.rules:
                logger.error("Rules not loaded - cannot connect to Claude AI")
                self.claude_client = None
                return True  # Continue with fallback analysis
                
            api_key = self.rules["api_credentials"]["anthropic_api_key"]
            
            if api_key == "YOUR_ANTHROPIC_API_KEY_HERE":
                logger.warning("Anthropic API key not configured - using fallback analysis")
                self.claude_client = None
                return True
            
            # Initialize Anthropic client
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            
            # Test the connection with a simple request
            test_message = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            logger.info("Connected to Claude AI successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Claude AI: {e}")
            self.claude_client = None
            return True  # Continue with fallback analysis
    
    def get_market_data(self, crypto, period="5m", days=1):
        """Get real market data for a specific cryptocurrency using yfinance."""
        try:
            with self.data_lock:
                symbol = self.crypto_symbols.get(crypto, f"{crypto}-USD")
                
                # Get data from yfinance
                ticker = yf.Ticker(symbol)
                
                # Get 5-minute interval data for the specified number of days
                data = ticker.history(period=f"{days}d", interval=period)
                
                if data.empty:
                    logger.warning(f"No market data available for {crypto}")
                    return None
                
                # Convert to our format
                timestamps = [dt.to_pydatetime() for dt in data.index]
                prices = data['Close'].tolist()
                volumes = data['Volume'].tolist()
                highs = data['High'].tolist()
                lows = data['Low'].tolist()
                opens = data['Open'].tolist()
                
                # Cache the data in database
                self._cache_market_data(crypto, data)
                
                # Store in memory for quick access
                self.market_data[crypto] = {
                    "price_history": prices,
                    "timestamps": timestamps,
                    "volumes": volumes,
                    "highs": highs,
                    "lows": lows,
                    "opens": opens,
                    "current_price": prices[-1] if prices else 0
                }
                
                return {
                    "price": prices[-1] if prices else 0,
                    "price_history": prices,
                    "timestamps": timestamps,
                    "volumes": volumes,
                    "highs": highs,
                    "lows": lows,
                    "opens": opens
                }
                
        except Exception as e:
            logger.error(f"Error getting market data for {crypto}: {e}")
            # Try to get cached data as fallback
            return self._get_cached_market_data(crypto)
    
    def _cache_market_data(self, crypto, data):
        """Cache market data in SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for timestamp, row in data.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO market_data_cache 
                    (crypto, timestamp, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    crypto,
                    timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    float(row['Volume'])
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error caching market data for {crypto}: {e}")
    
    def _get_cached_market_data(self, crypto):
        """Get cached market data from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, open_price, high_price, low_price, close_price, volume
                FROM market_data_cache 
                WHERE crypto = ?
                ORDER BY timestamp DESC
                LIMIT 288
            ''', (crypto,))  # 288 = 24 hours * 12 (5-minute intervals)
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return None
            
            timestamps = [datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S') for row in rows]
            opens = [row[1] for row in rows]
            highs = [row[2] for row in rows]
            lows = [row[3] for row in rows]
            prices = [row[4] for row in rows]
            volumes = [row[5] for row in rows]
            
            # Reverse to get chronological order
            timestamps.reverse()
            opens.reverse()
            highs.reverse()
            lows.reverse()
            prices.reverse()
            volumes.reverse()
            
            return {
                "price": prices[-1] if prices else 0,
                "price_history": prices,
                "timestamps": timestamps,
                "volumes": volumes,
                "highs": highs,
                "lows": lows,
                "opens": opens
            }
            
        except Exception as e:
            logger.error(f"Error getting cached market data for {crypto}: {e}")
            return None
    
    def calculate_indicators(self, crypto):
        """Calculate technical indicators for a cryptocurrency."""
        try:
            if not self.rules:
                logger.error("Rules not loaded - cannot calculate indicators")
                return None
                
            market_data = self.get_market_data(crypto)
            if not market_data:
                return None
            
            price_history = market_data["price_history"]
            
            if len(price_history) < 26:  # Need at least 26 periods for MACD
                logger.warning(f"Insufficient data for indicators calculation for {crypto}")
                return None
            
            # Convert to DataFrame for indicator calculation
            df = pd.DataFrame({
                "close": price_history,
                "timestamp": market_data["timestamps"]
            })
            
            # Calculate RSI
            rsi_period = self.rules["trading_strategy"]["indicators"][0]["parameters"]["period"]
            if len(price_history) >= rsi_period:
                rsi = RSIIndicator(close=df["close"], window=rsi_period)
                df["rsi"] = rsi.rsi()
            else:
                df["rsi"] = np.nan
            
            # Calculate MACD
            macd_params = self.rules["trading_strategy"]["indicators"][1]["parameters"]
            if len(price_history) >= macd_params["slow_period"]:
                macd = MACD(
                    close=df["close"],
                    window_slow=macd_params["slow_period"],
                    window_fast=macd_params["fast_period"],
                    window_sign=macd_params["signal_period"]
                )
                df["macd"] = macd.macd()
                df["macd_signal"] = macd.macd_signal()
                df["macd_diff"] = macd.macd_diff()
            else:
                df["macd"] = np.nan
                df["macd_signal"] = np.nan
                df["macd_diff"] = np.nan
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators for {crypto}: {e}")
            return None
    
    def get_claude_analysis(self, crypto, indicators_df):
        """Get trading analysis from Claude AI."""
        try:
            if indicators_df.empty:
                return {
                    "action": "hold",
                    "confidence": 0.5,
                    "reason": "Insufficient data for analysis",
                    "price": 0,
                    "timestamp": datetime.datetime.now()
                }
            
            # Extract the latest indicator values
            latest = indicators_df.iloc[-1]
            rsi = latest.get("rsi", np.nan)
            macd_diff = latest.get("macd_diff", np.nan)
            price = latest.get("close", 0)
            
            # Use Claude AI if available
            if self.claude_client:
                return self._get_claude_ai_analysis(crypto, indicators_df, latest)
            else:
                return self._get_fallback_analysis(crypto, latest, rsi, macd_diff, price)
                
        except Exception as e:
            logger.error(f"Error getting Claude analysis for {crypto}: {e}")
            return self._get_fallback_analysis(crypto, {}, np.nan, np.nan, 0)
    
    def _get_claude_ai_analysis(self, crypto, indicators_df, latest):
        """Get analysis from Claude AI API."""
        try:
            # Prepare market data summary for Claude
            recent_data = indicators_df.tail(10)  # Last 10 data points
            
            market_summary = {
                "crypto": crypto,
                "current_price": float(latest.get("close", 0)),
                "rsi": float(latest.get("rsi", 0)) if pd.notna(latest.get("rsi")) else None,
                "macd": float(latest.get("macd", 0)) if pd.notna(latest.get("macd")) else None,
                "macd_signal": float(latest.get("macd_signal", 0)) if pd.notna(latest.get("macd_signal")) else None,
                "macd_histogram": float(latest.get("macd_diff", 0)) if pd.notna(latest.get("macd_diff")) else None,
                "price_trend": "up" if len(recent_data) > 1 and recent_data.iloc[-1]["close"] > recent_data.iloc[-2]["close"] else "down",
                "custom_strategy": self.rules["trading_strategy"].get("custom_strategy", "")
            }
            
            # Create prompt for Claude
            prompt = f"""
            Analyze the following cryptocurrency market data for {crypto} and provide a trading recommendation:
            
            Current Market Data:
            - Price: ${market_summary['current_price']:.2f}
            - RSI: {market_summary['rsi']}
            - MACD: {market_summary['macd']}
            - MACD Signal: {market_summary['macd_signal']}
            - MACD Histogram: {market_summary['macd_histogram']}
            - Recent Price Trend: {market_summary['price_trend']}
            
            Trading Strategy: {market_summary['custom_strategy']}
            
            Please provide:
            1. Action: "buy", "sell", or "hold"
            2. Confidence: A number between 0.0 and 1.0
            3. Reason: Brief explanation for your recommendation
            
            Format your response as JSON:
            {{"action": "buy/sell/hold", "confidence": 0.0-1.0, "reason": "explanation"}}
            """
            
            # Send request to Claude
            message = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse Claude's response
            response_text = message.content[0].text.strip()
            
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                claude_response = json.loads(json_match.group())
                
                return {
                    "action": claude_response.get("action", "hold").lower(),
                    "confidence": float(claude_response.get("confidence", 0.5)),
                    "reason": f"Claude AI: {claude_response.get('reason', 'No specific reason provided')}",
                    "price": market_summary['current_price'],
                    "timestamp": datetime.datetime.now()
                }
            else:
                # Fallback if JSON parsing fails
                logger.warning(f"Could not parse Claude response for {crypto}: {response_text}")
                return self._get_fallback_analysis(crypto, latest, market_summary['rsi'], market_summary['macd_histogram'], market_summary['current_price'])
                
        except Exception as e:
            logger.error(f"Error getting Claude AI analysis for {crypto}: {e}")
            return self._get_fallback_analysis(crypto, latest, latest.get("rsi", np.nan), latest.get("macd_diff", np.nan), latest.get("close", 0))
    
    def _get_fallback_analysis(self, crypto, latest, rsi, macd_diff, price):
        """Fallback analysis when Claude AI is not available."""
        try:
            # Rule-based analysis using the custom strategy
            if not self.rules:
                return {
                    "action": "hold",
                    "confidence": 0.3,
                    "reason": "Fallback Analysis: Rules not loaded",
                    "price": price,
                    "timestamp": datetime.datetime.now()
                }
                
            if pd.notna(rsi) and pd.notna(macd_diff):
                rsi_params = self.rules["trading_strategy"]["indicators"][0]["parameters"]
                
                if rsi < rsi_params["oversold"] and macd_diff > 0:
                    confidence = 0.75
                    action = "buy"
                    reason = f"Fallback Analysis: RSI oversold ({rsi:.2f}) with positive MACD momentum ({macd_diff:.4f})"
                elif rsi > rsi_params["overbought"] and macd_diff < 0:
                    confidence = 0.75
                    action = "sell"
                    reason = f"Fallback Analysis: RSI overbought ({rsi:.2f}) with negative MACD momentum ({macd_diff:.4f})"
                else:
                    confidence = 0.4
                    action = "hold"
                    reason = f"Fallback Analysis: No clear signal - RSI={rsi:.2f}, MACD histogram={macd_diff:.4f}"
            else:
                confidence = 0.3
                action = "hold"
                reason = "Fallback Analysis: Insufficient indicator data for decision"
            
            return {
                "action": action,
                "confidence": confidence,
                "reason": reason,
                "price": price,
                "timestamp": datetime.datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error in fallback analysis for {crypto}: {e}")
            return {
                "action": "hold",
                "confidence": 0.3,
                "reason": f"Error in analysis: {str(e)}",
                "price": price,
                "timestamp": datetime.datetime.now()
            }
    
    def execute_trade(self, crypto, action, price, reason):
        """Execute a trade on Robinhood or in paper trading mode."""
        try:
            timestamp = datetime.datetime.now()
            is_paper_trading = self.rules["general_settings"]["paper_trading"]
            
            if action == "buy":
                # Calculate quantity based on max investment per trade
                max_investment = self.rules["trading_parameters"]["max_investment_per_trade"]
                quantity = max_investment / price
                
                if is_paper_trading:
                    # Paper trading logic
                    if self.paper_trading_balance >= max_investment:
                        self.paper_trading_balance -= max_investment
                        
                        if crypto in self.paper_trading_positions:
                            # Average down the position
                            old_quantity = self.paper_trading_positions[crypto]["quantity"]
                            old_value = self.paper_trading_positions[crypto]["value"]
                            new_quantity = old_quantity + quantity
                            new_value = old_value + max_investment
                            
                            self.paper_trading_positions[crypto] = {
                                "quantity": new_quantity,
                                "value": new_value,
                                "average_price": new_value / new_quantity
                            }
                        else:
                            self.paper_trading_positions[crypto] = {
                                "quantity": quantity,
                                "value": max_investment,
                                "average_price": price
                            }
                        
                        # Update current positions for UI
                        self.current_positions = self.paper_trading_positions.copy()
                        
                        logger.info(f"PAPER BUY: {quantity:.6f} {crypto} at ${price:.2f} for ${max_investment:.2f}")
                    else:
                        logger.warning(f"Insufficient paper trading balance for {crypto} buy order")
                        return False
                else:
                    # Real trading logic
                    try:
                        # Use Robinhood API to place buy order
                        order = rh.order_buy_crypto_by_price(crypto, max_investment)
                        if order and order.get('state') == 'confirmed':
                            # Update positions
                            if crypto in self.current_positions:
                                self.current_positions[crypto]["quantity"] += quantity
                                self.current_positions[crypto]["value"] += max_investment
                            else:
                                self.current_positions[crypto] = {
                                    "quantity": quantity,
                                    "value": max_investment,
                                    "average_price": price
                                }
                            logger.info(f"REAL BUY: {quantity:.6f} {crypto} at ${price:.2f} for ${max_investment:.2f}")
                        else:
                            logger.error(f"Failed to execute real buy order for {crypto}")
                            return False
                    except Exception as e:
                        logger.error(f"Error executing real buy order for {crypto}: {e}")
                        return False
                
                # Record trade
                trade = {
                    "timestamp": timestamp,
                    "crypto": crypto,
                    "action": action,
                    "price": price,
                    "quantity": quantity,
                    "value": max_investment,
                    "reason": reason
                }
                
            elif action == "sell":
                positions_to_check = self.paper_trading_positions if is_paper_trading else self.current_positions
                
                if crypto in positions_to_check:
                    # Sell all holdings of this crypto
                    quantity = positions_to_check[crypto]["quantity"]
                    value = quantity * price
                    
                    if is_paper_trading:
                        # Paper trading logic
                        self.paper_trading_balance += value
                        del self.paper_trading_positions[crypto]
                        self.current_positions = self.paper_trading_positions.copy()
                        logger.info(f"PAPER SELL: {quantity:.6f} {crypto} at ${price:.2f} for ${value:.2f}")
                    else:
                        # Real trading logic
                        try:
                            order = rh.order_sell_crypto_by_quantity(crypto, quantity)
                            if order and order.get('state') == 'confirmed':
                                del self.current_positions[crypto]
                                logger.info(f"REAL SELL: {quantity:.6f} {crypto} at ${price:.2f} for ${value:.2f}")
                            else:
                                logger.error(f"Failed to execute real sell order for {crypto}")
                                return False
                        except Exception as e:
                            logger.error(f"Error executing real sell order for {crypto}: {e}")
                            return False
                    
                    # Record trade
                    trade = {
                        "timestamp": timestamp,
                        "crypto": crypto,
                        "action": action,
                        "price": price,
                        "quantity": quantity,
                        "value": value,
                        "reason": reason
                    }
                else:
                    logger.warning(f"No position to sell for {crypto}")
                    return False
            else:
                return False
            
            # Add to trade history
            self.trade_history.append(trade)
            
            # Save trade to database
            self._save_trade_to_db(trade, is_paper_trading)
            
            return True
        except Exception as e:
            logger.error(f"Error executing trade for {crypto}: {e}")
            return False
    
    def _save_trade_to_db(self, trade, is_paper_trading):
        """Save trade to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trade_history 
                (timestamp, crypto, action, price, quantity, value, reason, paper_trading)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade["timestamp"].strftime('%Y-%m-%d %H:%M:%S'),
                trade["crypto"],
                trade["action"],
                trade["price"],
                trade["quantity"],
                trade["value"],
                trade["reason"],
                is_paper_trading
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")
    
    def update_portfolio_value(self):
        """Update and record the current portfolio value."""
        try:
            total_value = 0
            is_paper_trading = self.rules["general_settings"]["paper_trading"]
            
            if is_paper_trading:
                # Add cash balance
                total_value += self.paper_trading_balance
                
                # Calculate value of all positions
                for crypto, position in self.paper_trading_positions.items():
                    market_data = self.get_market_data(crypto)
                    if market_data:
                        current_price = market_data["price"]
                        position_value = position["quantity"] * current_price
                        total_value += position_value
            else:
                # Real trading - get actual portfolio value from Robinhood
                try:
                    portfolio = rh.load_portfolio_profile()
                    if portfolio:
                        total_value = float(portfolio.get('total_return_today', 0))
                except Exception as e:
                    logger.error(f"Error getting real portfolio value: {e}")
                    # Fallback to calculated value
                    for crypto, position in self.current_positions.items():
                        market_data = self.get_market_data(crypto)
                        if market_data:
                            current_price = market_data["price"]
                            position_value = position["quantity"] * current_price
                            total_value += position_value
            
            # Record portfolio value
            timestamp = datetime.datetime.now()
            portfolio_value = {
                "timestamp": timestamp,
                "portfolio_value": total_value
            }
            
            self.portfolio_history.append(portfolio_value)
            
            # Save to database
            self._save_portfolio_value_to_db(portfolio_value, is_paper_trading)
            
            return total_value
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
            return 0
    
    def _save_portfolio_value_to_db(self, portfolio_value, is_paper_trading):
        """Save portfolio value to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolio_history 
                (timestamp, portfolio_value, paper_trading)
                VALUES (?, ?, ?)
            ''', (
                portfolio_value["timestamp"].strftime('%Y-%m-%d %H:%M:%S'),
                portfolio_value["portfolio_value"],
                is_paper_trading
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving portfolio value to database: {e}")
    
    def trading_loop(self):
        """Main trading loop that runs continuously."""
        while self.is_running:
            try:
                # Update portfolio value
                self.update_portfolio_value()
                
                # Check each cryptocurrency in the rules
                for crypto in self.rules["cryptocurrencies"]:
                    # Calculate indicators
                    indicators_df = self.calculate_indicators(crypto)
                    
                    if indicators_df is not None and not indicators_df.empty:
                        # Get Claude AI analysis
                        analysis = self.get_claude_analysis(crypto, indicators_df)
                        
                        if analysis:
                            # Check if confidence meets threshold
                            confidence_threshold = self.rules["trading_strategy"]["claude_ai"]["confidence_threshold"]
                            
                            if analysis["confidence"] >= confidence_threshold:
                                # Execute trade based on Claude's recommendation
                                if analysis["action"] in ["buy", "sell"]:
                                    self.execute_trade(
                                        crypto=crypto,
                                        action=analysis["action"],
                                        price=analysis["price"],
                                        reason=analysis["reason"]
                                    )
                
                # Sleep for 5 minutes (real-time 5-minute bars)
                time.sleep(300)  # 5 minutes = 300 seconds
                    
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Sleep and retry
    
    def start(self):
        """Start the trading bot."""
        if self.is_running:
            return False
        
        if not self.rules:
            logger.error("Cannot start bot - rules not loaded")
            return False
        
        # Connect to APIs
        robinhood_connected = self.connect_to_robinhood()
        claude_connected = self.connect_to_claude_ai()
        
        if robinhood_connected and claude_connected:
            self.is_running = True
            
            # Start trading loop in a separate thread
            self.trading_thread = Thread(target=self.trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            logger.info("Trading bot started successfully")
            return True
        else:
            logger.error("Failed to start trading bot due to connection issues")
            return False
    
    def stop(self):
        """Stop the trading bot."""
        if not self.is_running:
            return False
        
        self.is_running = False
        logger.info("Trading bot stopped")
        return True
    
    def get_portfolio_data_from_db(self):
        """Get portfolio history from database."""
        try:
            if not self.rules:
                logger.error("Rules not loaded - cannot get portfolio data")
                return pd.DataFrame()
                
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('''
                SELECT timestamp, portfolio_value 
                FROM portfolio_history 
                WHERE paper_trading = ?
                ORDER BY timestamp
            ''', conn, params=(self.rules["general_settings"]["paper_trading"],))
            conn.close()
            
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            return df
        except Exception as e:
            logger.error(f"Error getting portfolio data from database: {e}")
            return pd.DataFrame()
    
    def get_trade_data_from_db(self):
        """Get trade history from database."""
        try:
            if not self.rules:
                logger.error("Rules not loaded - cannot get trade data")
                return pd.DataFrame()
                
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('''
                SELECT timestamp, crypto, action, price, quantity, value, reason 
                FROM trade_history 
                WHERE paper_trading = ?
                ORDER BY timestamp DESC
            ''', conn, params=(self.rules["general_settings"]["paper_trading"],))
            conn.close()
            
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            return df
        except Exception as e:
            logger.error(f"Error getting trade data from database: {e}")
            return pd.DataFrame()


# Streamlit UI
def create_ui():
    st.set_page_config(
        page_title="Crypto Trading Bot",
        page_icon="💹",
        layout="wide"
    )
    
    # Initialize bot
    if 'bot' not in st.session_state:
        st.session_state.bot = CryptoTradingBot()
    
    bot = st.session_state.bot
    
    # Header
    st.title("Crypto Trading Bot powered by Claude AI")
    
    # Show trading mode
    trading_mode = "Paper Trading" if bot.rules and bot.rules["general_settings"]["paper_trading"] else "Live Trading"
    st.subheader(f"Mode: {trading_mode}")
    
    if bot.rules and bot.rules["general_settings"]["paper_trading"]:
        st.info(f"💰 Paper Trading Balance: ${bot.paper_trading_balance:.2f}")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        if not bot.is_running:
            if st.button("Start Bot"):
                if bot.start():
                    st.success("Bot started successfully!")
                    st.rerun()
                else:
                    st.error("Failed to start bot")
        else:
            if st.button("Stop Bot"):
                if bot.stop():
                    st.success("Bot stopped successfully!")
                    st.rerun()
                else:
                    st.error("Failed to stop bot")
        
        # Bot status
        status = "🟢 Running" if bot.is_running else "🔴 Stopped"
        st.write(f"Status: {status}")
        
        st.header("Settings")
        
        # Display current rules
        with st.expander("Trading Rules", expanded=False):
            if bot.rules:
                st.json(bot.rules)
            else:
                st.error("Rules not loaded")
        
        # API Connection Status
        st.header("API Status")
        if bot.rules:
            # Claude AI status
            claude_status = "🟢 Connected" if bot.claude_client else "🟡 Fallback Mode"
            st.write(f"Claude AI: {claude_status}")
            
            # Robinhood status
            if bot.rules["general_settings"]["paper_trading"]:
                rh_status = "🟡 Paper Trading"
            elif bot.robinhood_logged_in:
                rh_status = "🟢 Connected"
            else:
                rh_status = "🔴 Disconnected"
            st.write(f"Robinhood: {rh_status}")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Value")
        
        # Load portfolio history from database
        try:
            portfolio_df = bot.get_portfolio_data_from_db()
            if not portfolio_df.empty:
                # Create portfolio value chart
                fig = px.line(
                    portfolio_df, 
                    x="timestamp", 
                    y="portfolio_value",
                    title="Portfolio Value Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display current portfolio value
                latest_value = portfolio_df["portfolio_value"].iloc[-1] if len(portfolio_df) > 0 else 0
                st.metric("Current Portfolio Value", f"${latest_value:.2f}")
            else:
                st.info("No portfolio data available yet")
        except Exception as e:
            st.error(f"Error loading portfolio data: {e}")
    
    with col2:
        st.subheader("Current Positions")
        
        # Display current positions
        if bot.current_positions:
            positions_data = []
            
            for crypto, position in bot.current_positions.items():
                market_data = bot.get_market_data(crypto)
                current_price = market_data["price"] if market_data else 0
                current_value = position["quantity"] * current_price
                profit_loss = current_value - position["value"]
                profit_loss_pct = (profit_loss / position["value"]) * 100 if position["value"] > 0 else 0
                
                positions_data.append({
                    "Crypto": crypto,
                    "Quantity": f"{position['quantity']:.6f}",
                    "Avg Price": f"${position['average_price']:.2f}",
                    "Current Price": f"${current_price:.2f}",
                    "Value": f"${current_value:.2f}",
                    "P/L": f"${profit_loss:.2f} ({profit_loss_pct:.2f}%)"
                })
            
            st.table(positions_data)
        else:
            st.info("No open positions")
    
    # Trade History
    st.subheader("Trade History")
    
    try:
        trade_df = bot.get_trade_data_from_db()
        if not trade_df.empty:
            # Format the dataframe for display
            display_df = trade_df.copy()
            display_df["price"] = display_df["price"].apply(lambda x: f"${x:.2f}")
            display_df["value"] = display_df["value"].apply(lambda x: f"${x:.2f}")
            display_df["action"] = display_df["action"].str.upper()
            
            st.dataframe(
                display_df[["timestamp", "crypto", "action", "price", "quantity", "value", "reason"]],
                use_container_width=True
            )
        else:
            st.info("No trade history available yet")
    except Exception as e:
        st.error(f"Error loading trade history: {e}")
    
    # Market Data
    st.subheader("Market Data")
    
    # Create tabs for each cryptocurrency
    if bot.rules:
        crypto_tabs = st.tabs(bot.rules["cryptocurrencies"])
        
        for i, crypto in enumerate(bot.rules["cryptocurrencies"]):
            with crypto_tabs[i]:
                market_data = bot.get_market_data(crypto)
                
                if market_data:
                    # Create price chart
                    price_df = pd.DataFrame({
                        "timestamp": market_data["timestamps"],
                        "price": market_data["price_history"]
                    })
                    
                    fig = px.line(
                        price_df, 
                        x="timestamp", 
                        y="price",
                        title=f"{crypto} Price (5-minute bars)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display current price
                    current_price = market_data["price"]
                    st.metric(f"{crypto} Current Price", f"${current_price:.2f}")
                    
                    # Calculate and display indicators
                    indicators_df = bot.calculate_indicators(crypto)
                    
                    if indicators_df is not None and not indicators_df.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # RSI Chart
                            fig_rsi = px.line(
                                indicators_df, 
                                x="timestamp", 
                                y="rsi",
                                title=f"{crypto} RSI"
                            )
                            
                            # Add overbought/oversold lines
                            rsi_params = bot.rules["trading_strategy"]["indicators"][0]["parameters"]
                            fig_rsi.add_hline(y=rsi_params["overbought"], line_dash="dash", line_color="red", annotation_text="Overbought")
                            fig_rsi.add_hline(y=rsi_params["oversold"], line_dash="dash", line_color="green", annotation_text="Oversold")
                            
                            st.plotly_chart(fig_rsi, use_container_width=True)
                            
                            # Display current RSI
                            current_rsi = indicators_df["rsi"].iloc[-1] if not pd.isna(indicators_df["rsi"].iloc[-1]) else "N/A"
                            st.metric(f"{crypto} RSI", f"{current_rsi:.2f}" if current_rsi != "N/A" else "N/A")
                        
                        with col2:
                            # MACD Chart
                            fig_macd = go.Figure()
                            
                            # Add MACD line
                            fig_macd.add_trace(
                                go.Scatter(
                                    x=indicators_df["timestamp"],
                                    y=indicators_df["macd"],
                                    mode="lines",
                                    name="MACD",
                                    line=dict(color="blue")
                                )
                            )
                            
                            # Add Signal line
                            fig_macd.add_trace(
                                go.Scatter(
                                    x=indicators_df["timestamp"],
                                    y=indicators_df["macd_signal"],
                                    mode="lines",
                                    name="Signal",
                                    line=dict(color="red")
                                )
                            )
                            
                            # Add MACD histogram
                            fig_macd.add_trace(
                                go.Bar(
                                    x=indicators_df["timestamp"],
                                    y=indicators_df["macd_diff"],
                                    name="Histogram",
                                    marker_color="green"
                                )
                            )
                            
                            fig_macd.update_layout(title=f"{crypto} MACD")
                            st.plotly_chart(fig_macd, use_container_width=True)
                            
                            # Display current MACD
                            current_macd = indicators_df["macd_diff"].iloc[-1] if not pd.isna(indicators_df["macd_diff"].iloc[-1]) else "N/A"
                            st.metric(f"{crypto} MACD Histogram", f"{current_macd:.4f}" if current_macd != "N/A" else "N/A")
                    else:
                        st.info(f"No indicator data available for {crypto} yet")
                else:
                    st.info(f"No market data available for {crypto} yet")
    
    # Auto-refresh every 30 seconds
    time.sleep(1)  # Small delay to prevent too frequent refreshes
    if bot.is_running:
        st.rerun()


if __name__ == "__main__":
    create_ui()
