# 📝 Changelog

All notable changes to the Raspberry Pi Trading Bot project.

## [Unreleased] - 2025-07-04

### 🧹 Project Reorganization
- **Cleaned root directory** from 20+ files to just essentials
- **Organized files** into logical directories:
  - `tests/` - All test files
  - `scripts/` - Utility and deployment scripts
  - `examples/` - Demo and example files
  - `docker/` - Docker configurations
  - `docs/requirements/` - Alternative requirement files
- **Added** `STRUCTURE.md` documenting project organization
- **Added** `requirements-dev.txt` for development dependencies

### 📚 Documentation Overhaul
- **Consolidated** 22+ scattered markdown files into 12 organized files
- **Created** comprehensive documentation structure:
  - `docs/getting-started/` - Setup and deployment guides
  - `docs/technical/` - Architecture and technical details
  - `docs/features/` - Feature documentation
  - `docs/analysis/` - Comparisons and advantages
- **Rewrote** main README.md with professional presentation
- **Added** documentation index at `docs/README.md`

### 🌈 Beautiful Logging System
- **Implemented** colored logging with emojis
- **Added** contextual emojis for different events:
  - 💡 Opportunities, 🎯 Signals, 💰 Trades
  - 🔍 Scanner, ⚙️ System, ⚠️ Warnings
- **Fixed** duplicate log messages issue
- **Created** logging guide with emoji reference

### 🧪 Testnet Optimization
- **Created** `TestnetScanner` with adjusted thresholds
- **Lowered** requirements for testnet's low volatility:
  - Momentum: 0.2% moves (was 2%)
  - Volume: 1.2-1.5x (was 2-3x)
  - Score threshold: 30-40 (was 70)
- **Fixed** division by zero warnings in scanner

### 🔧 Trading Improvements
- **Fixed** position sizing bug (was always 0)
- **Added** missing price field for risk calculations
- **Improved** opportunity detection and execution flow
- **Enhanced** error handling and logging

### 🚀 Deployment & Setup
- **Created** automated GitHub Actions deployment
- **Added** `setup_pi.sh` for easy Raspberry Pi setup
- **Documented** systemd service configuration
- **Improved** deployment guides

### 📦 Dependencies
- **Cleaned** requirements from 20+ heavy packages to 10 essentials
- **Removed** unused: nautilus-trader, tensorflow, scikit-learn
- **Added** missing: colorama, asyncio-throttle
- **Reduced** install size from ~2GB to ~200MB

## [1.0.0] - 2025-07-03

### Initial Release
- Core trading engine with 4 strategies
- Market scanner for 100+ pairs
- Risk management system
- Tax tracking integration
- Telegram notifications
- Paper trading mode
- Binance testnet support

---

**Note**: This project follows [Semantic Versioning](https://semver.org/).

**Legend**:
- 🚀 New features
- 🔧 Improvements
- 🐛 Bug fixes
- 📚 Documentation
- 🧹 Maintenance
- ⚠️ Breaking changes