#!/bin/bash

# Setup script for NautilusOracle - Autonomous Trading System

PROJECT_NAME="NautilusOracle"
GITHUB_USERNAME="your-username"  # Change this!

echo "════════════════════════════════════════════════════════════════"
echo "    🌊 Setting up $PROJECT_NAME Repository"
echo "════════════════════════════════════════════════════════════════"

# Initialize git repository
echo "📁 Initializing git repository..."
git init

# Create initial commit
echo "📝 Creating initial commit..."
git add .
git commit -m "🌊 Initial commit - $PROJECT_NAME autonomous trading system

- Autonomous trading engine with self-healing capabilities
- ML-powered strategy selection and optimization  
- Adaptive risk management across market regimes
- Target: 10% annual returns with minimal intervention
- Built on Nautilus Trader infrastructure"

# Set up main branch
git branch -M main

# Create directory structure
echo "📂 Creating project structure..."
mkdir -p tests
mkdir -p scripts  
mkdir -p docs/images
mkdir -p logs
mkdir -p data/historical
mkdir -p backtest_results

# Create example test file
cat > tests/test_engine.py << 'EOF'
"""Tests for NautilusOracle autonomous engine."""

import pytest
from autonomous_trading.core.engine import AutonomousEngine


def test_engine_initialization():
    """Test engine initializes correctly."""
    # Add tests here
    pass
EOF

# Create run script
cat > run_nautilusoracle.py << 'EOF'
#!/usr/bin/env python3
"""
NautilusOracle - Main entry point
Autonomous trading system for passive income generation
"""

import asyncio
import click
from autonomous_trading.run_autonomous import main


@click.command()
@click.option('--mode', default='paper', help='Trading mode: paper or live')
@click.option('--config', default='config.json', help='Configuration file')
@click.option('--capital', default=0.3, help='Starting capital in BTC')
def run(mode, config, capital):
    """Run NautilusOracle autonomous trading system."""
    print(f"🌊 Starting NautilusOracle in {mode} mode")
    print(f"💰 Capital: {capital} BTC")
    print(f"📋 Config: {config}")
    
    # Run the system
    asyncio.run(main(mode, config, capital))


if __name__ == "__main__":
    run()
EOF

chmod +x run_nautilusoracle.py

# Create LICENSE file
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 $PROJECT_NAME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

echo ""
echo "✅ Repository structure created!"
echo ""
echo "Next steps:"
echo "1. Update GITHUB_USERNAME in this script"
echo "2. Create repository on GitHub: $PROJECT_NAME"
echo "3. Run: git remote add origin git@github.com:$GITHUB_USERNAME/$PROJECT_NAME.git"
echo "4. Run: git push -u origin main"
echo ""
echo "🌊 $PROJECT_NAME is ready for autonomous trading!"