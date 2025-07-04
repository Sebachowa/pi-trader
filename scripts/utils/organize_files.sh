#!/bin/bash
# Organize root directory files into proper structure

echo "🧹 Organizing project files..."
echo "================================"

# Move test files to tests/
echo "📂 Moving test files to tests/"
mv test_*.py tests/ 2>/dev/null
mv run_tests.sh tests/ 2>/dev/null

# Move deployment scripts
echo "📂 Moving deployment scripts to scripts/deployment/"
mv deploy_to_pi.sh scripts/deployment/ 2>/dev/null
mv setup_pi.sh scripts/deployment/ 2>/dev/null

# Move utility scripts
echo "📂 Moving utility scripts to scripts/utils/"
mv cleanup_docs.sh scripts/utils/ 2>/dev/null
mv organize_files.sh scripts/utils/ 2>/dev/null  # This script itself
mv debug_*.py scripts/utils/ 2>/dev/null

# Move example files
echo "📂 Moving example files to examples/"
mv example_usage.py examples/ 2>/dev/null
mv demo.py examples/ 2>/dev/null
mv run_demo.py examples/ 2>/dev/null

# Move Docker files
echo "📂 Moving Docker files to docker/"
mv Dockerfile* docker/ 2>/dev/null
mv docker-compose*.yml docker/ 2>/dev/null

# Move extra requirements files to docs/
echo "📂 Moving extra requirements to docs/"
mkdir -p docs/requirements
mv requirements-minimal.txt docs/requirements/ 2>/dev/null
mv requirements-pi.txt docs/requirements/ 2>/dev/null

# Keep only main files in root
echo ""
echo "✅ Files kept in root:"
echo "  - run.py (main entry point)"
echo "  - start.sh (quick start script)"
echo "  - requirements.txt (main dependencies)"
echo "  - README.md"
echo "  - CLAUDE.md"
echo "  - AUTONOMOUS_BOT_ARCHITECTURE.md"
echo "  - .env.example"
echo "  - .gitignore"
echo "  - LICENSE"

# Create README for each directory
echo ""
echo "📝 Creating directory README files..."

# tests/README.md
cat > tests/README.md << 'EOF'
# 🧪 Tests

Test files for the trading bot.

## Files
- `test_binance_testnet.py` - Test Binance testnet connection
- `test_paper_trading.py` - Test paper trading functionality
- `test_scanner_opportunities.py` - Test scanner opportunity detection
- `test_testnet_scanner.py` - Test testnet-specific scanner
- `test_beautiful_logging.py` - Test logging system
- `run_tests.sh` - Run all tests

## Usage
```bash
# Run specific test
python tests/test_binance_testnet.py

# Run all tests
cd tests && ./run_tests.sh
```
EOF

# scripts/README.md
cat > scripts/README.md << 'EOF'
# 📜 Scripts

Utility and deployment scripts.

## Structure
- `deployment/` - Deployment and setup scripts
  - `deploy_to_pi.sh` - Deploy to Raspberry Pi
  - `setup_pi.sh` - Initial Pi setup
- `utils/` - Utility scripts
  - `debug_scanner.py` - Debug scanner issues
  - `debug_opportunity.py` - Debug opportunity detection
  - `cleanup_docs.sh` - Clean up documentation

## Usage
```bash
# Deploy to Pi
./scripts/deployment/deploy_to_pi.sh

# Debug scanner
python scripts/utils/debug_scanner.py
```
EOF

# examples/README.md
cat > examples/README.md << 'EOF'
# 💡 Examples

Example usage and demo files.

## Files
- `demo.py` - Simple demo mode
- `run_demo.py` - Advanced demo with fake data
- `example_usage.py` - Usage examples

## Usage
```bash
# Run demo mode
python examples/demo.py

# Run advanced demo
python examples/run_demo.py
```
EOF

# docker/README.md
cat > docker/README.md << 'EOF'
# 🐳 Docker Files

Docker configurations for containerized deployment.

## Files
- `Dockerfile` - Main Docker image
- `Dockerfile.multiarch` - Multi-architecture build
- `Dockerfile.node` - Node.js variant
- `docker-compose.yml` - Compose configuration

## Usage
```bash
# Build image
docker build -f docker/Dockerfile -t trading-bot .

# Run with compose
docker-compose -f docker/docker-compose.yml up
```
EOF

echo ""
echo "✅ Organization complete!"
echo ""
echo "📁 New structure:"
echo "."
echo "├── run.py                    # Main entry point"
echo "├── start.sh                  # Quick start"
echo "├── requirements.txt          # Dependencies"
echo "├── core/                     # Core bot logic"
echo "├── strategies/               # Trading strategies"
echo "├── config/                   # Configuration files"
echo "├── logs/                     # Log files"
echo "├── data/                     # Data storage"
echo "├── docs/                     # Documentation"
echo "│   └── requirements/         # Alternative requirements"
echo "├── tests/                    # All test files"
echo "├── scripts/                  # Utility scripts"
echo "│   ├── deployment/           # Deployment scripts"
echo "│   └── utils/                # Debug and utility scripts"
echo "├── examples/                 # Example usage"
echo "└── docker/                   # Docker files"

echo ""
echo "🎯 Root directory is now clean and organized!"