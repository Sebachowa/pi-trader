#!/bin/bash
# Setup script for Raspberry Pi Trading Bot

echo "🚀 Setting up Raspberry Pi Trading Bot..."
echo "========================================"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  WARNING: Not in a virtual environment!"
    echo "   It's recommended to use a virtual environment."
    echo "   Create one with: python3 -m venv venv"
    echo "   Activate with: source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update pip
echo "📦 Updating pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements-pi.txt

# Check if installation was successful
echo ""
echo "✅ Checking installation..."
python -c "import ccxt; print('✓ ccxt installed')"
python -c "import colorama; print('✓ colorama installed')"
python -c "import psutil; print('✓ psutil installed')"
python -c "import telegram; print('✓ python-telegram-bot installed')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the bot:"
echo "   python run.py"
echo ""
echo "📝 For testnet/demo mode:"
echo "   python run.py --demo"
echo ""
echo "📖 See docs/logging_guide.md for emoji reference"