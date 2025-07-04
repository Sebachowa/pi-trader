#!/bin/bash
# Quick start script for the trading bot

echo "🤖 Raspberry Pi Trading Bot Launcher"
echo "===================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if ! python -c "import ccxt" 2>/dev/null; then
    echo "📚 Installing dependencies..."
    pip install -r requirements-pi.txt
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found!"
    echo "Creating .env from template..."
    cp .env.example .env
    echo "✅ Created .env file with placeholders"
fi

# Menu
echo ""
echo "Select mode:"
echo "1) Demo Mode (no API keys needed)"
echo "2) Paper Trading (requires API keys)"
echo "3) Live Trading (requires API keys)"
echo "4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo "🎮 Starting DEMO mode..."
        python run.py --demo
        ;;
    2)
        echo "📝 Starting PAPER trading..."
        python run.py --paper
        ;;
    3)
        echo "💰 Starting LIVE trading..."
        echo "⚠️  WARNING: This will use real money!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            python run.py
        else
            echo "Cancelled."
        fi
        ;;
    4)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac