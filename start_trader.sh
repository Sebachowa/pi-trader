#!/bin/bash

echo "════════════════════════════════════════════════════════════════"
echo "    📈 TRADER - Autonomous Trading System"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📁 Project: trader"
echo "🎯 Target: 10% Annual Return"
echo "💰 Capital: 0.3 BTC"
echo ""

# Initialize git repository
cd /Users/seba/code/trader

if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit - Trader autonomous trading system"
    echo "✅ Git repository created"
fi

echo ""
echo "To run Trader:"
echo "1. cd /Users/seba/code/trader"
echo "2. pip install -r requirements.txt"
echo "3. python autonomous_trading/run_autonomous.py"
echo ""
echo "📊 Let's trade autonomously!"