# 📁 Project Structure

Clean and organized directory structure for the Raspberry Pi Trading Bot.

```
pi-trader/
├── 📄 Core Files (Root)
│   ├── run.py                    # Main entry point
│   ├── start.sh                  # Quick start script
│   ├── requirements.txt          # Core dependencies
│   ├── requirements-dev.txt      # Development dependencies
│   ├── README.md                 # Project overview
│   ├── STRUCTURE.md              # This file
│   ├── CLAUDE.md                 # Claude Code configuration
│   ├── AUTONOMOUS_BOT_ARCHITECTURE.md  # Design philosophy
│   ├── LICENSE                   # MIT license
│   ├── .env.example              # Environment template
│   └── .gitignore                # Git ignore rules
│
├── 🧠 core/                      # Core bot logic
│   ├── engine.py                 # Main trading engine
│   ├── market_scanner.py         # Market opportunity scanner
│   ├── testnet_scanner.py        # Testnet-optimized scanner
│   ├── risk_manager.py           # Risk management
│   ├── monitor.py                # System monitoring
│   ├── logger.py                 # Beautiful logging system
│   └── ...                       # Other core modules
│
├── 📈 strategies/                # Trading strategies
│   ├── trend_following.py        # Trend following strategy
│   ├── mean_reversion.py         # Mean reversion strategy
│   └── base_strategy.py          # Base strategy class
│
├── ⚙️ config/                    # Configuration
│   ├── config.json               # Default configuration
│   └── config.example.json       # Example configuration
│
├── 📚 docs/                      # Documentation
│   ├── README.md                 # Documentation index
│   ├── getting-started/          # Setup guides
│   ├── technical/                # Technical documentation
│   ├── features/                 # Feature documentation
│   ├── analysis/                 # Comparisons and analysis
│   └── requirements/             # Alternative requirements
│       ├── requirements-minimal.txt
│       └── requirements-pi.txt
│
├── 🧪 tests/                     # Test files
│   ├── README.md                 # Test documentation
│   ├── test_binance_testnet.py   # Testnet connection test
│   ├── test_paper_trading.py     # Paper trading test
│   ├── test_scanner_opportunities.py
│   ├── test_testnet_scanner.py
│   ├── test_beautiful_logging.py
│   └── run_tests.sh              # Run all tests
│
├── 📜 scripts/                   # Utility scripts
│   ├── README.md                 # Scripts documentation
│   ├── deployment/               # Deployment scripts
│   │   ├── deploy_to_pi.sh       # Deploy to Raspberry Pi
│   │   └── setup_pi.sh           # Initial Pi setup
│   └── utils/                    # Utility scripts
│       ├── debug_scanner.py      # Debug scanner
│       ├── debug_opportunity.py  # Debug opportunities
│       ├── cleanup_docs.sh       # Clean documentation
│       └── organize_files.sh     # Organize project files
│
├── 💡 examples/                  # Example usage
│   ├── README.md                 # Examples documentation
│   ├── demo.py                   # Simple demo
│   ├── run_demo.py               # Advanced demo
│   └── example_usage.py          # Usage examples
│
├── 🐳 docker/                    # Docker files
│   ├── README.md                 # Docker documentation
│   ├── Dockerfile                # Main Docker image
│   ├── Dockerfile.multiarch      # Multi-architecture build
│   ├── Dockerfile.node           # Node.js variant
│   └── docker-compose.yml        # Compose configuration
│
├── 📊 data/                      # Data storage (gitignored)
│   ├── README.md                 # Data directory info
│   └── ...                       # Trading data, backtest results
│
├── 📝 logs/                      # Log files (gitignored)
│   └── trader_*.log              # Trading logs
│
└── 🔒 venv/                      # Virtual environment (gitignored)
```

## 🎯 Key Principles

### 1. **Clean Root Directory**
- Only essential files in root
- Everything else organized in subdirectories
- Clear entry point (`run.py`)

### 2. **Logical Grouping**
- Core logic in `core/`
- Trading strategies in `strategies/`
- Tests in `tests/`
- Scripts in `scripts/`
- Documentation in `docs/`

### 3. **Clear Separation**
- Source code vs configuration
- Core dependencies vs development tools
- Production files vs examples/tests

### 4. **Easy Navigation**
- README in each major directory
- Descriptive file names
- Consistent structure

## 🚀 Common Tasks

### Running the Bot
```bash
python run.py              # Main bot
python run.py --demo       # Demo mode
./start.sh                 # Quick start script
```

### Running Tests
```bash
python tests/test_binance_testnet.py    # Single test
cd tests && ./run_tests.sh              # All tests
```

### Deployment
```bash
./scripts/deployment/deploy_to_pi.sh     # Deploy to Pi
./scripts/deployment/setup_pi.sh         # Setup Pi
```

### Debugging
```bash
python scripts/utils/debug_scanner.py    # Debug scanner
python scripts/utils/debug_opportunity.py # Debug opportunities
```

### Development
```bash
pip install -r requirements-dev.txt      # Install dev tools
black core/                              # Format code
ruff check core/                         # Lint code
```

## 📋 File Types

### Configuration Files
- `.json` - Configuration files
- `.env` - Environment variables (never commit!)
- `.example` - Example/template files

### Documentation
- `.md` - Markdown documentation
- `README.md` - Directory-specific guides

### Code Files
- `.py` - Python source code
- `.sh` - Shell scripts

### Docker Files
- `Dockerfile*` - Docker images
- `docker-compose.yml` - Compose configs

## 🔒 Security Notes

### Never Commit
- `.env` files
- API keys or secrets
- `telegram_config.json`
- Any credentials

### Always Gitignore
- `venv/` - Virtual environment
- `logs/` - Log files
- `data/` - Trading data
- `__pycache__/` - Python cache

---

This structure keeps the project organized, maintainable, and easy to navigate for both new users and experienced developers.