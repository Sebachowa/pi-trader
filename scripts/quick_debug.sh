#!/bin/bash
# Quick debug script for Raspberry Pi

echo "🔍 Trading Bot Debug"
echo "===================="

# Check if service exists
echo "1️⃣ Service status:"
sudo systemctl status trader --no-pager || echo "❌ Service not installed"

# Check directory
echo -e "\n2️⃣ Project directory:"
cd ~/code/pi-trader && pwd && ls -la

# Check venv
echo -e "\n3️⃣ Virtual environment:"
if [ -d "venv" ]; then
    echo "✅ venv exists"
    source venv/bin/activate
    echo "Python: $(which python)"
    echo "Version: $(python --version)"
else
    echo "❌ No venv found"
fi

# Check dependencies
echo -e "\n4️⃣ Key dependencies:"
python -c "import ccxt; print('✅ ccxt:', ccxt.__version__)" 2>&1
python -c "import pandas; print('✅ pandas:', pandas.__version__)" 2>&1
python -c "from core.engine import TradingEngine; print('✅ Engine OK')" 2>&1

# Check logs
echo -e "\n5️⃣ Recent logs:"
if [ -d "logs" ]; then
    ls -lt logs/ | head -5
else
    echo "No logs directory"
fi

# Test run
echo -e "\n6️⃣ Test import:"
python -c "
try:
    from core.engine import TradingEngine
    from core.market_scanner import MarketScanner
    print('✅ All imports successful!')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo -e "\n===================="
echo "Debug complete!"