#!/usr/bin/env python3
"""
Creates all files needed for SynapseTrade AI™ deployment
Run this script to generate all files at once
"""

import os

print("🚀 Creating all SynapseTrade AI™ files...")

# Note: app.py content is too large for this script
# You need to copy it manually from the artifact above

# Create requirements.txt
requirements_content = """# Core Flask dependencies
Flask==2.3.3
Flask-CORS==4.0.0
gunicorn==21.2.0
Werkzeug==2.3.7

# Data processing and analysis
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0

# Natural Language Processing
textblob==0.17.1

# Market data
yfinance==0.2.28

# Utilities
python-dotenv==1.0.0
requests==2.31.0

# Note: TensorFlow excluded to reduce deployment time
# Add later if needed: tensorflow-cpu==2.13.0"""

with open('requirements.txt', 'w') as f:
    f.write(requirements_content)
print("✅ Created requirements.txt")

# Create Procfile
procfile_content = """web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --log-level info"""

with open('Procfile', 'w') as f:
    f.write(procfile_content)
print("✅ Created Procfile")

# Create runtime.txt
runtime_content = """python-3.10.12"""

with open('runtime.txt', 'w') as f:
    f.write(runtime_content)
print("✅ Created runtime.txt")

# Create .env.example
env_content = """# SynapseTrade AI™ Environment Variables
SECRET_KEY=change-this-to-random-secret-key-123456789
FLASK_ENV=production
DATABASE_URL=sqlite:///synapsetrade.db
ENABLE_ADVANCED_FEATURES=true"""

with open('.env.example', 'w') as f:
    f.write(env_content)
print("✅ Created .env.example")

# Create README.md
readme_content = """# 🧠 SynapseTrade AI™ - Advanced Trading Intelligence Platform

## Overview

SynapseTrade AI™ is a comprehensive AI-powered trading platform featuring:
- Real-time market data analysis and technical indicators
- AI-powered sentiment analysis using TextBlob/NLP
- Machine learning price predictions
- Complete backtesting engine with Sharpe ratio calculations
- Risk management tools including position sizing
- RESTful API with comprehensive endpoints
- Beautiful responsive web UI

## Tech Stack

- **Backend**: Flask (Python 3.10)
- **ML/AI**: scikit-learn, TextBlob, pandas, numpy
- **Market Data**: yfinance
- **Database**: SQLite (auto-initialized)
- **Web Server**: Gunicorn

## Quick Start

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app.py`
4. Visit: `http://localhost:8080`

## API Endpoints

- `GET /` - Web interface
- `GET /api/health` - System health check
- `POST /api/sentiment/analyze` - Analyze news sentiment
- `GET /api/technical/{symbol}` - Technical indicators
- `POST /api/backtest` - Run strategy backtest
- `GET /api/ml/predict/{symbol}` - ML predictions
- `POST /api/risk/position-size` - Position sizing
- `GET /api/comparison/fundamental` - Stock comparison
- `GET /api/docs` - Full API documentation

## Deployment

This app is configured for deployment on Emergent.app. See deployment instructions in the main documentation.

## License

© 2024 SynapseTrade AI™ - All rights reserved"""

with open('README.md', 'w') as f:
    f.write(readme_content)
print("✅ Created README.md")

# Create EMERGENT_PROMPT.txt
prompt_content = """Deploy SynapseTrade AI™ - Advanced AI-Powered Trading Intelligence Platform

This is a production-ready trading platform featuring:
- Real-time market data analysis and technical indicators
- AI-powered sentiment analysis using TextBlob/NLP
- Machine learning price predictions
- Complete backtesting engine with Sharpe ratio calculations
- Risk management tools including position sizing
- RESTful API with comprehensive endpoints
- Beautiful responsive web UI

Tech Stack: Flask, Python 3.10, scikit-learn, pandas, numpy, yfinance
Database: SQLite (auto-initialized)
Web Server: Gunicorn

The app auto-initializes its database on first run and includes demo data.
All dependencies are in requirements.txt. Ready for immediate deployment.

Repository: https://github.com/idfuturestars/synapsetradeai"""

with open('EMERGENT_PROMPT.txt', 'w') as f:
    f.write(prompt_content)
print("✅ Created EMERGENT_PROMPT.txt")

print("\n" + "="*50)
print("✅ All files created successfully!")
print("="*50)
print("\n⚠️  IMPORTANT: You still need to:")
print("1. Copy the app.py content from the artifact above")
print("2. Save it as 'app.py' in this directory")
print("3. Upload all files to GitHub")
print("4. Deploy on Emergent.app")
print("\nYour GitHub repo: https://github.com/idfuturestars/synapsetradeai")