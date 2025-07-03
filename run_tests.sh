#!/bin/bash
#
# Test Runner for Autonomous Trading System
# Runs all tests with various options
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
TEST_TYPE="all"
COVERAGE=true
VERBOSE=false
MARKERS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            MARKERS="-m 'not integration'"
            ;;
        --integration)
            TEST_TYPE="integration"
            MARKERS="-m integration"
            ;;
        --no-coverage)
            COVERAGE=false
            ;;
        --verbose)
            VERBOSE=true
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --unit          Run only unit tests"
            echo "  --integration   Run only integration tests"
            echo "  --no-coverage   Skip coverage report"
            echo "  --verbose       Verbose output"
            echo "  --help          Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
    shift
done

# Print header
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              AUTONOMOUS TRADING SYSTEM - TESTS                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Setup virtual environment if needed
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install test dependencies
echo -e "${YELLOW}Installing test dependencies...${NC}"
pip install -q pytest pytest-asyncio pytest-cov pytest-mock

# Create test directories
mkdir -p tests/reports

# Run tests
echo -e "${YELLOW}Running $TEST_TYPE tests...${NC}"

# Build pytest command
PYTEST_CMD="python -m pytest tests/"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v -s"
else
    PYTEST_CMD="$PYTEST_CMD -q"
fi

if [ ! -z "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD $MARKERS"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=autonomous_trading --cov=nautilus_challenge"
    PYTEST_CMD="$PYTEST_CMD --cov-report=html:tests/reports/htmlcov"
    PYTEST_CMD="$PYTEST_CMD --cov-report=term"
    PYTEST_CMD="$PYTEST_CMD --cov-report=xml:tests/reports/coverage.xml"
fi

# Add JUnit XML report
PYTEST_CMD="$PYTEST_CMD --junit-xml=tests/reports/junit.xml"

# Run the tests
echo -e "${GREEN}Command: $PYTEST_CMD${NC}"
echo ""

if $PYTEST_CMD; then
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                    TESTS PASSED ✓                              ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    if [ "$COVERAGE" = true ]; then
        echo -e "${BLUE}Coverage report: tests/reports/htmlcov/index.html${NC}"
    fi
    
    exit 0
else
    echo -e "${RED}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                    TESTS FAILED ✗                              ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    exit 1
fi