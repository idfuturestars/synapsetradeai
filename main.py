# main.py - PalisadesBlueOceanTradeAgentAI Complete Replit Package
"""
PalisadesBlueOceanTradeAgentAI (TradeAI)
The Síkat Agency Trading Platform
Complete deployment package for Replit
"""

import os
import json
import sqlite3
import asyncio
import random
import hashlib
import jwt
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
import time

from flask import Flask, render_template_string, jsonify, request, make_response
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TradeAI")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'sikat-tradeai-secret-2024')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# === Database Setup ===
def init_database():
    """Initialize SQLite database with all required tables"""
    conn = sqlite3.connect('tradeai.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        api_key TEXT UNIQUE,
        balance REAL DEFAULT 100000.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Positions table
    c.execute('''CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        size REAL NOT NULL,
        entry_price REAL NOT NULL,
        current_price REAL,
        stop_loss REAL,
        take_profit REAL,
        pnl REAL DEFAULT 0,
        status TEXT DEFAULT 'open',
        opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        closed_at TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    
    # Orders table
    c.execute('''CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        order_type TEXT NOT NULL,
        size REAL NOT NULL,
        price REAL,
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        executed_at TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    
    # Market data table
    c.execute('''CREATE TABLE IF NOT EXISTS market_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        price REAL NOT NULL,
        bid REAL NOT NULL,
        ask REAL NOT NULL,
        volume REAL NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Trading signals table
    c.execute('''CREATE TABLE IF NOT EXISTS trading_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        direction TEXT NOT NULL,
        confidence REAL NOT NULL,
        predicted_price REAL NOT NULL,
        stop_loss REAL NOT NULL,
        take_profit REAL NOT NULL,
        model_name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    conn.commit()
    conn.close()
    
    # Create default user
    create_default_user()

def create_default_user():
    """Create a default demo user"""
    conn = sqlite3.connect('tradeai.db')
    c = conn.cursor()
    
    # Check if demo user exists
    c.execute("SELECT id FROM users WHERE username = ?", ('demo',))
    if not c.fetchone():
        password_hash = generate_password_hash('demo123')
        api_key = hashlib.sha256(f"demo-{datetime.now()}".encode()).hexdigest()
        
        c.execute("""INSERT INTO users (username, email, password_hash, api_key) 
                     VALUES (?, ?, ?, ?)""",
                  ('demo', 'demo@sikat.agency', password_hash, api_key))
        conn.commit()
        logger.info("Created demo user - Username: demo, Password: demo123")
    
    conn.close()

# === AI Models (Simulated) ===
class TradeAIPredictor:
    """Simulated AI prediction engine"""
    
    def __init__(self):
        self.models = {
            'lstm': {'weight': 0.25, 'accuracy': 0.82},
            'transformer': {'weight': 0.35, 'accuracy': 0.87},
            'lightgbm': {'weight': 0.20, 'accuracy': 0.79},
            'random_forest': {'weight': 0.20, 'accuracy': 0.76}
        }
        
    def predict(self, symbol: str, market_data: Dict) -> Dict:
        """Generate trading prediction"""
        # Simulate different model predictions
        predictions = {}
        
        for model_name, model_info in self.models.items():
            # Add some randomness but bias towards trends
            base_prediction = market_data['price'] * (1 + random.gauss(0, 0.002))
            predictions[model_name] = base_prediction
        
        # Weighted ensemble
        final_prediction = sum(
            predictions[model] * self.models[model]['weight'] 
            for model in predictions
        )
        
        # Determine direction
        price_change = (final_prediction - market_data['price']) / market_data['price']
        
        if abs(price_change) < 0.0005:  # 0.05% threshold
            direction = 'neutral'
        elif price_change > 0:
            direction = 'long'
        else:
            direction = 'short'
        
        # Calculate confidence (higher for stronger signals)
        confidence = min(0.95, 0.6 + abs(price_change) * 50)
        
        # Risk management levels
        stop_loss = market_data['price'] * (0.98 if direction == 'long' else 1.02)
        take_profit = market_data['price'] * (1.03 if direction == 'long' else 0.97)
        
        return {
            'symbol': symbol,
            'direction': direction,
            'confidence': confidence,
            'predicted_price': final_prediction,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'model_name': 'ensemble',
            'model_weights': self.models
        }

# === Market Data Simulator ===
class MarketDataSimulator:
    """Simulate realistic market data"""
    
    def __init__(self):
        self.symbols = {
            'USD/PHP': {'price': 56.50, 'volatility': 0.002},
            'EUR/USD': {'price': 1.0856, 'volatility': 0.001},
            'GBP/USD': {'price': 1.2745, 'volatility': 0.0015},
            'USD/JPY': {'price': 148.92, 'volatility': 0.0012},
            'BTC/USD': {'price': 45000, 'volatility': 0.02}
        }
        self.running = False
        self.thread = None
        
    def start(self):
        """Start market data simulation"""
        self.running = True
        self.thread = threading.Thread(target=self._simulate_market)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop market data simulation"""
        self.running = False
        
    def _simulate_market(self):
        """Simulate market movements"""
        while self.running:
            conn = sqlite3.connect('tradeai.db')
            c = conn.cursor()
            
            for symbol, params in self.symbols.items():
                # Random walk with momentum
                change = random.gauss(0, params['volatility'])
                params['price'] *= (1 + change)
                
                # Create realistic bid/ask spread
                spread = params['price'] * 0.0001  # 0.01% spread
                bid = params['price'] - spread/2
                ask = params['price'] + spread/2
                
                # Volume simulation
                volume = random.uniform(1000000, 5000000)
                
                # Store in database
                c.execute("""INSERT INTO market_data (symbol, price, bid, ask, volume)
                            VALUES (?, ?, ?, ?, ?)""",
                         (symbol, params['price'], bid, ask, volume))
                
                # Emit via WebSocket
                socketio.emit('market_update', {
                    'symbol': symbol,
                    'price': params['price'],
                    'bid': bid,
                    'ask': ask,
                    'volume': volume,
                    'timestamp': datetime.now().isoformat()
                }, room='market_data')
            
            conn.commit()
            conn.close()
            
            # Also generate occasional trading signals
            if random.random() < 0.1:  # 10% chance per tick
                self._generate_signal()
            
            time.sleep(2)  # Update every 2 seconds
            
    def _generate_signal(self):
        """Generate trading signal"""
        symbol = random.choice(list(self.symbols.keys()))
        market_data = {
            'price': self.symbols[symbol]['price'],
            'volatility': self.symbols[symbol]['volatility']
        }
        
        predictor = TradeAIPredictor()
        signal = predictor.predict(symbol, market_data)
        
        # Store signal
        conn = sqlite3.connect('tradeai.db')
        c = conn.cursor()
        c.execute("""INSERT INTO trading_signals 
                     (symbol, direction, confidence, predicted_price, stop_loss, take_profit, model_name)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (signal['symbol'], signal['direction'], signal['confidence'],
                   signal['predicted_price'], signal['stop_loss'], 
                   signal['take_profit'], signal['model_name']))
        conn.commit()
        conn.close()
        
        # Emit signal
        socketio.emit('trading_signal', signal, room='signals')

# === Trading Engine ===
class TradingEngine:
    """Core trading execution engine"""
    
    @staticmethod
    def execute_order(user_id: int, order: Dict) -> Dict:
        """Execute a trading order"""
        conn = sqlite3.connect('tradeai.db')
        c = conn.cursor()
        
        try:
            # Get current market price
            c.execute("""SELECT price, bid, ask FROM market_data 
                        WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1""",
                     (order['symbol'],))
            market = c.fetchone()
            
            if not market:
                return {'success': False, 'error': 'No market data available'}
            
            price, bid, ask = market
            
            # Determine execution price
            if order['side'] == 'buy':
                exec_price = ask
            else:
                exec_price = bid
            
            # Check user balance
            c.execute("SELECT balance FROM users WHERE id = ?", (user_id,))
            balance = c.fetchone()[0]
            
            required_margin = order['size'] * exec_price * 0.1  # 10% margin
            
            if required_margin > balance:
                return {'success': False, 'error': 'Insufficient balance'}
            
            # Create position
            c.execute("""INSERT INTO positions 
                        (user_id, symbol, side, size, entry_price, current_price, stop_loss, take_profit, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     (user_id, order['symbol'], order['side'], order['size'],
                      exec_price, exec_price, order.get('stop_loss'), 
                      order.get('take_profit'), 'open'))
            
            position_id = c.lastrowid
            
            # Update user balance
            c.execute("UPDATE users SET balance = balance - ? WHERE id = ?",
                     (required_margin, user_id))
            
            # Update order status
            c.execute("""UPDATE orders SET status = 'filled', executed_at = CURRENT_TIMESTAMP
                        WHERE id = ?""", (order.get('order_id'),))
            
            conn.commit()
            
            return {
                'success': True,
                'position_id': position_id,
                'execution_price': exec_price,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            conn.close()
    
    @staticmethod
    def update_positions():
        """Update all open positions with current prices"""
        conn = sqlite3.connect('tradeai.db')
        c = conn.cursor()
        
        # Get all open positions
        c.execute("""SELECT p.id, p.symbol, p.side, p.size, p.entry_price, p.stop_loss, p.take_profit
                     FROM positions p WHERE p.status = 'open'""")
        positions = c.fetchall()
        
        for pos in positions:
            pos_id, symbol, side, size, entry_price, stop_loss, take_profit = pos
            
            # Get current price
            c.execute("""SELECT price FROM market_data 
                        WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1""", (symbol,))
            current = c.fetchone()
            
            if current:
                current_price = current[0]
                
                # Calculate P&L
                if side == 'long':
                    pnl = (current_price - entry_price) * size
                else:
                    pnl = (entry_price - current_price) * size
                
                # Update position
                c.execute("""UPDATE positions 
                            SET current_price = ?, pnl = ? 
                            WHERE id = ?""",
                         (current_price, pnl, pos_id))
                
                # Check stop loss / take profit
                if stop_loss and ((side == 'long' and current_price <= stop_loss) or 
                                 (side == 'short' and current_price >= stop_loss)):
                    TradingEngine._close_position(pos_id, current_price, 'stop_loss')
                elif take_profit and ((side == 'long' and current_price >= take_profit) or 
                                    (side == 'short' and current_price <= take_profit)):
                    TradingEngine._close_position(pos_id, current_price, 'take_profit')
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def _close_position(position_id: int, price: float, reason: str):
        """Close a position"""
        conn = sqlite3.connect('tradeai.db')
        c = conn.cursor()
        
        c.execute("""UPDATE positions 
                    SET status = 'closed', current_price = ?, closed_at = CURRENT_TIMESTAMP
                    WHERE id = ?""",
                 (price, position_id))
        
        # Return margin to user
        c.execute("""SELECT user_id, size, entry_price FROM positions WHERE id = ?""",
                 (position_id,))
        user_id, size, entry_price = c.fetchone()
        
        margin = size * entry_price * 0.1
        c.execute("UPDATE users SET balance = balance + ? WHERE id = ?",
                 (margin, user_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Position {position_id} closed due to {reason}")

# === API Routes ===

@app.route('/')
def index():
    """Serve the main trading interface"""
    return render_template_string(TRADING_INTERFACE_HTML)

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    conn = sqlite3.connect('tradeai.db')
    c = conn.cursor()
    
    c.execute("SELECT id, password_hash, api_key FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    
    if user and check_password_hash(user[1], password):
        token = jwt.encode({
            'user_id': user[0],
            'username': username,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'success': True,
            'token': token,
            'api_key': user[2]
        })
    
    return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration"""
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not all([username, email, password]):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
    
    conn = sqlite3.connect('tradeai.db')
    c = conn.cursor()
    
    try:
        password_hash = generate_password_hash(password)
        api_key = hashlib.sha256(f"{username}-{datetime.now()}".encode()).hexdigest()
        
        c.execute("""INSERT INTO users (username, email, password_hash, api_key)
                     VALUES (?, ?, ?, ?)""",
                  (username, email, password_hash, api_key))
        conn.commit()
        
        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'api_key': api_key
        })
        
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'error': 'Username or email already exists'}), 400
    finally:
        conn.close()

@app.route('/api/market/data/<symbol>')
def get_market_data(symbol):
    """Get latest market data for a symbol"""
    conn = sqlite3.connect('tradeai.db')
    c = conn.cursor()
    
    c.execute("""SELECT price, bid, ask, volume, timestamp 
                 FROM market_data 
                 WHERE symbol = ? 
                 ORDER BY timestamp DESC 
                 LIMIT 50""", (symbol,))
    
    data = []
    for row in c.fetchall():
        data.append({
            'price': row[0],
            'bid': row[1],
            'ask': row[2],
            'volume': row[3],
            'timestamp': row[4]
        })
    
    conn.close()
    
    return jsonify({
        'symbol': symbol,
        'data': data
    })

@app.route('/api/trading/signals')
def get_trading_signals():
    """Get latest trading signals"""
    conn = sqlite3.connect('tradeai.db')
    c = conn.cursor()
    
    c.execute("""SELECT symbol, direction, confidence, predicted_price, 
                        stop_loss, take_profit, model_name, created_at
                 FROM trading_signals 
                 ORDER BY created_at DESC 
                 LIMIT 10""")
    
    signals = []
    for row in c.fetchall():
        signals.append({
            'symbol': row[0],
            'direction': row[1],
            'confidence': row[2],
            'predicted_price': row[3],
            'stop_loss': row[4],
            'take_profit': row[5],
            'model_name': row[6],
            'timestamp': row[7]
        })
    
    conn.close()
    
    return jsonify({'signals': signals})

@app.route('/api/trading/execute', methods=['POST'])
def execute_trade():
    """Execute a trade"""
    # Get user from token
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        user_id = payload['user_id']
    except:
        return jsonify({'success': False, 'error': 'Invalid token'}), 401
    
    data = request.json
    
    # Create order
    conn = sqlite3.connect('tradeai.db')
    c = conn.cursor()
    
    c.execute("""INSERT INTO orders (user_id, symbol, side, order_type, size, price)
                 VALUES (?, ?, ?, ?, ?, ?)""",
              (user_id, data['symbol'], data['side'], data.get('order_type', 'market'),
               data['size'], data.get('price')))
    
    order_id = c.lastrowid
    conn.commit()
    conn.close()
    
    # Execute order
    order = {
        'order_id': order_id,
        'symbol': data['symbol'],
        'side': data['side'],
        'size': data['size'],
        'stop_loss': data.get('stop_loss'),
        'take_profit': data.get('take_profit')
    }
    
    result = TradingEngine.execute_order(user_id, order)
    
    return jsonify(result)

@app.route('/api/portfolio/positions')
def get_positions():
    """Get user's positions"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        user_id = payload['user_id']
    except:
        return jsonify({'success': False, 'error': 'Invalid token'}), 401
    
    conn = sqlite3.connect('tradeai.db')
    c = conn.cursor()
    
    c.execute("""SELECT id, symbol, side, size, entry_price, current_price, 
                        pnl, status, opened_at
                 FROM positions 
                 WHERE user_id = ? AND status = 'open'
                 ORDER BY opened_at DESC""", (user_id,))
    
    positions = []
    for row in c.fetchall():
        positions.append({
            'id': row[0],
            'symbol': row[1],
            'side': row[2],
            'size': row[3],
            'entry_price': row[4],
            'current_price': row[5],
            'pnl': row[6],
            'status': row[7],
            'opened_at': row[8]
        })
    
    # Get account balance
    c.execute("SELECT balance FROM users WHERE id = ?", (user_id,))
    balance = c.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'positions': positions,
        'balance': balance,
        'total_pnl': sum(p['pnl'] for p in positions if p['pnl'])
    })

@app.route('/api/portfolio/performance')
def get_performance():
    """Get portfolio performance metrics"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        user_id = payload['user_id']
    except:
        return jsonify({'success': False, 'error': 'Invalid token'}), 401
    
    conn = sqlite3.connect('tradeai.db')
    c = conn.cursor()
    
    # Calculate metrics
    c.execute("""SELECT COUNT(*), SUM(pnl) 
                 FROM positions 
                 WHERE user_id = ? AND status = 'closed'""", (user_id,))
    
    total_trades, total_pnl = c.fetchone()
    
    c.execute("""SELECT COUNT(*) 
                 FROM positions 
                 WHERE user_id = ? AND status = 'closed' AND pnl > 0""", (user_id,))
    
    winning_trades = c.fetchone()[0]
    
    conn.close()
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    return jsonify({
        'total_trades': total_trades or 0,
        'total_pnl': total_pnl or 0,
        'win_rate': win_rate,
        'sharpe_ratio': 1.85,  # Simulated
        'max_drawdown': 0.12   # Simulated
    })

# === WebSocket Events ===

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    join_room('market_data')
    join_room('signals')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")
    leave_room('market_data')
    leave_room('signals')

@socketio.on('subscribe')
def handle_subscribe(data):
    """Handle subscription requests"""
    room = data.get('room')
    if room in ['market_data', 'signals']:
        join_room(room)
        emit('subscribed', {'room': room})

# === HTML Interface ===
TRADING_INTERFACE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PalisadesBlueOceanTradeAgentAI - The Síkat Agency</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .logo-circle {
            width: 50px;
            height: 50px;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 20px;
        }
        
        .account-info {
            text-align: right;
        }
        
        .balance {
            font-size: 24px;
            font-weight: bold;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: #1a1a1a;
            border-radius: 15px;
            padding: 25px;
            border: 1px solid #333;
        }
        
        .card h3 {
            margin-bottom: 20px;
            color: #667eea;
        }
        
        .market-ticker {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: #252525;
            border-radius: 10px;
            margin-bottom: 10px;
            transition: all 0.3s ease;
        }
        
        .market-ticker:hover {
            background: #303030;
            transform: translateX(5px);
        }
        
        .price-up {
            color: #4ecdc4;
        }
        
        .price-down {
            color: #ff6b6b;
        }
        
        .signal {
            padding: 15px;
            background: #252525;
            border-radius: 10px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }
        
        .signal-bullish {
            border-left-color: #4ecdc4;
        }
        
        .signal-bearish {
            border-left-color: #ff6b6b;
        }
        
        .confidence-bar {
            height: 6px;
            background: #333;
            border-radius: 3px;
            margin-top: 10px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
        
        .trade-form {
            display: grid;
            gap: 15px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        
        .form-group label {
            font-size: 14px;
            color: #999;
        }
        
        .form-group input, .form-group select {
            padding: 12px;
            background: #252525;
            border: 1px solid #444;
            border-radius: 8px;
            color: white;
            font-size: 16px;
        }
        
        .button-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn-buy {
            background: #4ecdc4;
            color: white;
        }
        
        .btn-sell {
            background: #ff6b6b;
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .position {
            padding: 15px;
            background: #252525;
            border-radius: 10px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .position-info {
            flex: 1;
        }
        
        .position-pnl {
            font-size: 18px;
            font-weight: bold;
        }
        
        .login-container {
            max-width: 400px;
            margin: 100px auto;
            background: #1a1a1a;
            padding: 40px;
            border-radius: 15px;
            border: 1px solid #333;
        }
        
        .tab-nav {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 1px solid #333;
        }
        
        .tab {
            padding: 10px 20px;
            background: none;
            border: none;
            color: #999;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        .tab.active {
            color: #667eea;
            border-bottom: 2px solid #667eea;
        }
        
        .chart-container {
            position: relative;
            height: 400px;
            margin: 20px 0;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .live-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #4ecdc4;
            border-radius: 50%;
            margin-left: 10px;
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div id="app">
        <!-- Login Screen -->
        <div id="loginScreen" class="login-container">
            <h2 style="text-align: center; margin-bottom: 30px;">
                <div class="logo-circle" style="margin: 0 auto 20px; width: 80px; height: 80px; font-size: 30px;">S</div>
                PalisadesBlueOceanTradeAgentAI
            </h2>
            <p style="text-align: center; color: #999; margin-bottom: 30px;">The Síkat Agency Trading Platform</p>
            
            <div class="tab-nav">
                <button class="tab active" onclick="showTab('login')">Login</button>
                <button class="tab" onclick="showTab('register')">Register</button>
            </div>
            
            <form id="loginForm">
                <div class="form-group">
                    <label>Username</label>
                    <input type="text" id="loginUsername" required value="demo">
                </div>
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" id="loginPassword" required value="demo123">
                </div>
                <button type="submit" class="btn btn-buy" style="width: 100%; margin-top: 20px;">Login</button>
            </form>
            
            <form id="registerForm" style="display: none;">
                <div class="form-group">
                    <label>Username</label>
                    <input type="text" id="regUsername" required>
                </div>
                <div class="form-group">
                    <label>Email</label>
                    <input type="email" id="regEmail" required>
                </div>
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" id="regPassword" required>
                </div>
                <button type="submit" class="btn btn-buy" style="width: 100%; margin-top: 20px;">Register</button>
            </form>
            
            <p style="text-align: center; margin-top: 20px; color: #666;">
                Demo credentials: username: <strong>demo</strong>, password: <strong>demo123</strong>
            </p>
        </div>
        
        <!-- Main Trading Interface -->
        <div id="mainInterface" style="display: none;">
            <div class="container">
                <div class="header">
                    <div class="logo">
                        <div class="logo-circle">S</div>
                        <div>
                            <h1>PalisadesBlueOceanTradeAgentAI</h1>
                            <p style="opacity: 0.8;">The Síkat Agency Trading Platform</p>
                        </div>
                    </div>
                    <div class="account-info">
                        <p>Account Balance</p>
                        <div class="balance" id="accountBalance">$100,000.00</div>
                        <p style="font-size: 14px; opacity: 0.8;">Total P&L: <span id="totalPnl">$0.00</span></p>
                    </div>
                </div>
                
                <div class="grid">
                    <!-- Market Data -->
                    <div class="card">
                        <h3>Market Data <span class="live-indicator"></span></h3>
                        <div id="marketTickers"></div>
                    </div>
                    
                    <!-- Trading Signals -->
                    <div class="card">
                        <h3>AI Trading Signals</h3>
                        <div id="tradingSignals"></div>
                    </div>
                </div>
                
                <!-- Chart -->
                <div class="card">
                    <h3>Price Chart - USD/PHP</h3>
                    <div class="chart-container">
                        <canvas id="priceChart"></canvas>
                    </div>
                </div>
                
                <div class="grid">
                    <!-- Trade Execution -->
                    <div class="card">
                        <h3>Execute Trade</h3>
                        <form id="tradeForm" class="trade-form">
                            <div class="form-group">
                                <label>Symbol</label>
                                <select id="tradeSymbol">
                                    <option value="USD/PHP">USD/PHP</option>
                                    <option value="EUR/USD">EUR/USD</option>
                                    <option value="GBP/USD">GBP/USD</option>
                                    <option value="USD/JPY">USD/JPY</option>
                                    <option value="BTC/USD">BTC/USD</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Size</label>
                                <input type="number" id="tradeSize" value="10000" min="1000" step="1000">
                            </div>
                            <div class="form-group">
                                <label>Stop Loss (Optional)</label>
                                <input type="number" id="stopLoss" step="0.0001">
                            </div>
                            <div class="form-group">
                                <label>Take Profit (Optional)</label>
                                <input type="number" id="takeProfit" step="0.0001">
                            </div>
                            <div class="button-group">
                                <button type="button" class="btn btn-buy" onclick="executeTrade('buy')">BUY</button>
                                <button type="button" class="btn btn-sell" onclick="executeTrade('sell')">SELL</button>
                            </div>
                        </form>
                    </div>
                    
                    <!-- Open Positions -->
                    <div class="card">
                        <h3>Open Positions</h3>
                        <div id="openPositions"></div>
                    </div>
                </div>
                
                <!-- Performance Metrics -->
                <div class="card">
                    <h3>Performance Metrics</h3>
                    <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
                        <div style="text-align: center;">
                            <p style="color: #999;">Total Trades</p>
                            <p style="font-size: 24px; font-weight: bold;" id="totalTrades">0</p>
                        </div>
                        <div style="text-align: center;">
                            <p style="color: #999;">Win Rate</p>
                            <p style="font-size: 24px; font-weight: bold;" id="winRate">0%</p>
                        </div>
                        <div style="text-align: center;">
                            <p style="color: #999;">Sharpe Ratio</p>
                            <p style="font-size: 24px; font-weight: bold;" id="sharpeRatio">0.00</p>
                        </div>
                        <div style="text-align: center;">
                            <p style="color: #999;">Max Drawdown</p>
                            <p style="font-size: 24px; font-weight: bold;" id="maxDrawdown">0%</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Global variables
        let socket = null;
        let authToken = null;
        let priceChart = null;
        let marketData = {};
        let chartData = {
            labels: [],
            datasets: [{
                label: 'USD/PHP',
                data: [],
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4
            }]
        };
        
        // Tab switching
        function showTab(tab) {
            if (tab === 'login') {
                document.getElementById('loginForm').style.display = 'block';
                document.getElementById('registerForm').style.display = 'none';
            } else {
                document.getElementById('loginForm').style.display = 'none';
                document.getElementById('registerForm').style.display = 'block';
            }
            
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
        }
        
        // Login
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    username: document.getElementById('loginUsername').value,
                    password: document.getElementById('loginPassword').value
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                authToken = data.token;
                localStorage.setItem('authToken', authToken);
                showMainInterface();
            } else {
                alert('Login failed: ' + data.error);
            }
        });
        
        // Register
        document.getElementById('registerForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const response = await fetch('/api/auth/register', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    username: document.getElementById('regUsername').value,
                    email: document.getElementById('regEmail').value,
                    password: document.getElementById('regPassword').value
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                alert('Registration successful! Your API key: ' + data.api_key);
                showTab('login');
            } else {
                alert('Registration failed: ' + data.error);
            }
        });
        
        // Show main interface
        function showMainInterface() {
            document.getElementById('loginScreen').style.display = 'none';
            document.getElementById('mainInterface').style.display = 'block';
            
            initializeChart();
            initializeWebSocket();
            loadInitialData();
            
            // Update positions every 5 seconds
            setInterval(updatePositions, 5000);
            setInterval(updatePerformance, 10000);
        }
        
        // Initialize price chart
        function initializeChart() {
            const ctx = document.getElementById('priceChart').getContext('2d');
            priceChart = new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                color: '#333'
                            },
                            ticks: {
                                color: '#999'
                            }
                        },
                        y: {
                            grid: {
                                color: '#333'
                            },
                            ticks: {
                                color: '#999'
                            }
                        }
                    }
                }
            });
        }
        
        // Initialize WebSocket
        function initializeWebSocket() {
            socket = io();
            
            socket.on('connect', () => {
                console.log('Connected to TradeAI');
                socket.emit('subscribe', {room: 'market_data'});
                socket.emit('subscribe', {room: 'signals'});
            });
            
            socket.on('market_update', (data) => {
                updateMarketTicker(data);
                
                // Update chart for USD/PHP
                if (data.symbol === 'USD/PHP') {
                    updateChart(data);
                }
            });
            
            socket.on('trading_signal', (signal) => {
                displaySignal(signal);
            });
        }
        
        // Load initial data
        async function loadInitialData() {
            // Load market data
            const symbols = ['USD/PHP', 'EUR/USD', 'GBP/USD', 'USD/JPY', 'BTC/USD'];
            for (const symbol of symbols) {
                const response = await fetch(`/api/market/data/${symbol}`);
                const data = await response.json();
                if (data.data.length > 0) {
                    marketData[symbol] = data.data[0];
                    updateMarketTicker({
                        symbol: symbol,
                        ...data.data[0]
                    });
                }
            }
            
            // Load positions
            await updatePositions();
            
            // Load performance
            await updatePerformance();
            
            // Load signals
            const signalsResponse = await fetch('/api/trading/signals');
            const signalsData = await signalsResponse.json();
            signalsData.signals.forEach(signal => displaySignal(signal));
        }
        
        // Update market ticker
        function updateMarketTicker(data) {
            const tickersDiv = document.getElementById('marketTickers');
            let ticker = document.getElementById(`ticker-${data.symbol.replace('/', '-')}`);
            
            if (!ticker) {
                ticker = document.createElement('div');
                ticker.id = `ticker-${data.symbol.replace('/', '-')}`;
                ticker.className = 'market-ticker';
                tickersDiv.appendChild(ticker);
            }
            
            const previousPrice = marketData[data.symbol]?.price || data.price;
            const priceChange = data.price - previousPrice;
            const priceClass = priceChange >= 0 ? 'price-up' : 'price-down';
            
            ticker.innerHTML = `
                <div>
                    <strong>${data.symbol}</strong>
                    <div style="font-size: 12px; color: #999;">
                        Bid: ${data.bid.toFixed(4)} | Ask: ${data.ask.toFixed(4)}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div class="${priceClass}" style="font-size: 20px; font-weight: bold;">
                        ${data.price.toFixed(4)}
                    </div>
                    <div style="font-size: 12px;">
                        Vol: ${(data.volume / 1000000).toFixed(2)}M
                    </div>
                </div>
            `;
            
            marketData[data.symbol] = data;
        }
        
        // Update chart
        function updateChart(data) {
            if (chartData.labels.length > 50) {
                chartData.labels.shift();
                chartData.datasets[0].data.shift();
            }
            
            chartData.labels.push(new Date(data.timestamp).toLocaleTimeString());
            chartData.datasets[0].data.push(data.price);
            
            priceChart.update('none');
        }
        
        // Display trading signal
        function displaySignal(signal) {
            const signalsDiv = document.getElementById('tradingSignals');
            const signalDiv = document.createElement('div');
            signalDiv.className = `signal signal-${signal.direction === 'long' ? 'bullish' : 'bearish'}`;
            
            signalDiv.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>${signal.symbol}</strong> - ${signal.direction.toUpperCase()}
                        <div style="font-size: 12px; color: #999; margin-top: 5px;">
                            Target: ${signal.predicted_price.toFixed(4)} | 
                            SL: ${signal.stop_loss.toFixed(4)} | 
                            TP: ${signal.take_profit.toFixed(4)}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 18px; font-weight: bold;">
                            ${(signal.confidence * 100).toFixed(1)}%
                        </div>
                        <div style="font-size: 12px; color: #999;">
                            ${signal.model_name}
                        </div>
                    </div>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${signal.confidence * 100}%"></div>
                </div>
            `;
            
            signalsDiv.insertBefore(signalDiv, signalsDiv.firstChild);
            
            // Keep only last 5 signals
            while (signalsDiv.children.length > 5) {
                signalsDiv.removeChild(signalsDiv.lastChild);
            }
        }
        
        // Execute trade
        async function executeTrade(side) {
            const symbol = document.getElementById('tradeSymbol').value;
            const size = parseFloat(document.getElementById('tradeSize').value);
            const stopLoss = parseFloat(document.getElementById('stopLoss').value) || null;
            const takeProfit = parseFloat(document.getElementById('takeProfit').value) || null;
            
            const response = await fetch('/api/trading/execute', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${authToken}`
                },
                body: JSON.stringify({
                    symbol,
                    side,
                    size,
                    stop_loss: stopLoss,
                    take_profit: takeProfit
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                alert(`Trade executed! Position ID: ${result.position_id}`);
                await updatePositions();
            } else {
                alert(`Trade failed: ${result.error}`);
            }
        }
        
        // Update positions
        async function updatePositions() {
            const response = await fetch('/api/portfolio/positions', {
                headers: {
                    'Authorization': `Bearer ${authToken}`
                }
            });
            
            const data = await response.json();
            
            // Update balance
            document.getElementById('accountBalance').textContent = `$${data.balance.toFixed(2)}`;
            document.getElementById('totalPnl').textContent = `$${data.total_pnl.toFixed(2)}`;
            
            // Update positions list
            const positionsDiv = document.getElementById('openPositions');
            positionsDiv.innerHTML = '';
            
            data.positions.forEach(position => {
                const posDiv = document.createElement('div');
                posDiv.className = 'position';
                
                const pnlClass = position.pnl >= 0 ? 'price-up' : 'price-down';
                const pnlPercent = (position.pnl / (position.size * position.entry_price) * 100).toFixed(2);
                
                posDiv.innerHTML = `
                    <div class="position-info">
                        <strong>${position.symbol} - ${position.side.toUpperCase()}</strong>
                        <div style="font-size: 12px; color: #999;">
                            Size: ${position.size.toLocaleString()} | 
                            Entry: ${position.entry_price.toFixed(4)} | 
                            Current: ${position.current_price?.toFixed(4) || 'N/A'}
                        </div>
                    </div>
                    <div class="position-pnl ${pnlClass}">
                        ${position.pnl >= 0 ? '+' : ''}$${position.pnl.toFixed(2)}
                        <div style="font-size: 12px;">
                            ${position.pnl >= 0 ? '+' : ''}${pnlPercent}%
                        </div>
                    </div>
                `;
                
                positionsDiv.appendChild(posDiv);
            });
            
            if (data.positions.length === 0) {
                positionsDiv.innerHTML = '<p style="color: #999; text-align: center;">No open positions</p>';
            }
        }
        
        // Update performance metrics
        async function updatePerformance() {
            const response = await fetch('/api/portfolio/performance', {
                headers: {
                    'Authorization': `Bearer ${authToken}`
                }
            });
            
            const data = await response.json();
            
            document.getElementById('totalTrades').textContent = data.total_trades;
            document.getElementById('winRate').textContent = `${data.win_rate.toFixed(1)}%`;
            document.getElementById('sharpeRatio').textContent = data.sharpe_ratio.toFixed(2);
            document.getElementById('maxDrawdown').textContent = `${(data.max_drawdown * 100).toFixed(1)}%`;
        }
        
        // Check for existing token
        const savedToken = localStorage.getItem('authToken');
        if (savedToken) {
            authToken = savedToken;
            showMainInterface();
        }
    </script>
</body>
</html>
'''

# === Background Tasks ===
def start_background_tasks():
    """Start all background tasks"""
    # Start market data simulator
    market_sim = MarketDataSimulator()
    market_sim.start()
    
    # Position updater
    def update_positions_loop():
        while True:
            TradingEngine.update_positions()
            time.sleep(5)
    
    position_thread = threading.Thread(target=update_positions_loop)
    position_thread.daemon = True
    position_thread.start()

# === Main Entry Point ===
if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Start background tasks
    start_background_tasks()
    
    # Log startup info
    logger.info("=" * 60)
    logger.info("PalisadesBlueOceanTradeAgentAI (TradeAI)")
    logger.info("The Síkat Agency Trading Platform")
    logger.info("=" * 60)
    logger.info("Demo credentials - Username: demo, Password: demo123")
    logger.info("Server starting on http://0.0.0.0:5000")
    logger.info("=" * 60)
    
    # Run the app
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)