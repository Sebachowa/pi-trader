#!/bin/bash
#
# Autonomous Trading System Startup Script
# This script starts the 24/7 autonomous trading system with proper configuration
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
CONFIG_FILE="${SCRIPT_DIR}/autonomous_trading/config/trading_config.json"
VENV_PATH="${SCRIPT_DIR}/venv"

# Set Python command
if command -v pyenv &> /dev/null; then
    PYTHON_CMD="$(pyenv which python)"
else
    PYTHON_CMD="python3"
fi

# Functions
print_banner() {
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║           AUTONOMOUS TRADING SYSTEM - AUTOBOT                  ║"
    echo "║                    24/7 Trading Platform                       ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_requirements() {
    echo -e "${YELLOW}Checking system requirements...${NC}"
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 is not installed${NC}"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    REQUIRED_VERSION="3.10"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        echo -e "${RED}Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)${NC}"
        exit 1
    fi
    
    # Check config file
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Requirements satisfied${NC}"
}

setup_environment() {
    echo -e "${YELLOW}Setting up environment...${NC}"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating virtual environment..."
        $PYTHON_CMD -m venv "$VENV_PATH"
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip --quiet
    
    # Install requirements
    if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
        echo "Installing dependencies..."
        pip install -r "${SCRIPT_DIR}/requirements.txt" --quiet
    fi
    
    # Create necessary directories
    mkdir -p "$LOG_DIR"
    mkdir -p "${SCRIPT_DIR}/data"
    mkdir -p "${SCRIPT_DIR}/backups"
    
    echo -e "${GREEN}✓ Environment ready${NC}"
}

check_credentials() {
    echo -e "${YELLOW}Checking credentials...${NC}"
    
    # Read config to determine mode
    MODE=$("${VENV_PATH}/bin/python" -c "import json; print(json.load(open('$CONFIG_FILE'))['mode'])")
    
    if [ "$MODE" == "live" ]; then
        # Check for required environment variables
        if [ -z "$BINANCE_API_KEY" ] || [ -z "$BINANCE_API_SECRET" ]; then
            echo -e "${RED}Error: BINANCE_API_KEY and BINANCE_API_SECRET must be set for live trading${NC}"
            echo "Please export these variables or add them to your .env file"
            exit 1
        fi
    else
        echo -e "${GREEN}Running in paper trading mode${NC}"
    fi
    
    # Check Telegram credentials if enabled
    TELEGRAM_ENABLED=$("${VENV_PATH}/bin/python" -c "import json; print(json.load(open('$CONFIG_FILE'))['notifications']['telegram']['enabled'])")
    
    if [ "$TELEGRAM_ENABLED" == "True" ]; then
        if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ]; then
            echo -e "${YELLOW}Warning: Telegram credentials not set. Notifications will be disabled.${NC}"
        fi
    fi
    
    echo -e "${GREEN}✓ Credentials checked${NC}"
}

start_trading_system() {
    echo -e "${YELLOW}Starting Autonomous Trading System...${NC}"
    
    # Export Python path
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
    
    # Create log file with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${LOG_DIR}/autobot_${TIMESTAMP}.log"
    
    # Display configuration summary
    echo -e "${GREEN}Configuration Summary:${NC}"
    "${VENV_PATH}/bin/python" -c "
import json
config = json.load(open('$CONFIG_FILE'))
print(f'  Mode: {config[\"mode\"]}')
print(f'  Initial Capital: {config[\"initial_capital\"]} BTC')
print(f'  Target Return: {config[\"target_annual_return\"]*100}%')
print(f'  Max Drawdown: {config[\"max_drawdown\"]*100}%')
print(f'  Instruments: {len(config[\"instruments\"])}')
print(f'  Max Strategies: {config[\"strategy_management\"][\"max_concurrent_strategies\"]}')
"
    
    echo ""
    echo -e "${YELLOW}Starting trading system...${NC}"
    echo "Logs will be written to: $LOG_FILE"
    echo ""
    
    # Start the trading system
    if [ "$1" == "--daemon" ]; then
        # Run as daemon
        nohup "${VENV_PATH}/bin/python" -u "${SCRIPT_DIR}/autonomous_trading/main_trading_system.py" "$CONFIG_FILE" \
            > "$LOG_FILE" 2>&1 &
        
        PID=$!
        echo $PID > "${SCRIPT_DIR}/.autobot.pid"
        
        echo -e "${GREEN}✓ Trading system started as daemon (PID: $PID)${NC}"
        echo "To monitor: tail -f $LOG_FILE"
        echo "To stop: ./stop_autonomous_trading.sh"
    else
        # Run in foreground
        "${VENV_PATH}/bin/python" -u "${SCRIPT_DIR}/autonomous_trading/main_trading_system.py" "$CONFIG_FILE" \
            2>&1 | tee "$LOG_FILE"
    fi
}

# Main execution
main() {
    print_banner
    check_requirements
    setup_environment
    check_credentials
    start_trading_system "$@"
}

# Run main function with all arguments
main "$@"