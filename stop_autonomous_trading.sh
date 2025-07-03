#!/bin/bash
#
# Autonomous Trading System Stop Script
# Gracefully stops the trading system
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/.autobot.pid"

echo -e "${YELLOW}Stopping Autonomous Trading System...${NC}"

if [ ! -f "$PID_FILE" ]; then
    echo -e "${RED}Error: PID file not found. Is the system running?${NC}"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
    echo -e "${RED}Process $PID is not running${NC}"
    rm -f "$PID_FILE"
    exit 1
fi

# Send SIGTERM for graceful shutdown
echo "Sending shutdown signal to process $PID..."
kill -TERM "$PID"

# Wait for process to stop (max 30 seconds)
COUNTER=0
while kill -0 "$PID" 2>/dev/null && [ $COUNTER -lt 30 ]; do
    echo -n "."
    sleep 1
    COUNTER=$((COUNTER + 1))
done

echo ""

if kill -0 "$PID" 2>/dev/null; then
    echo -e "${YELLOW}Process did not stop gracefully. Forcing shutdown...${NC}"
    kill -KILL "$PID"
    sleep 2
fi

rm -f "$PID_FILE"
echo -e "${GREEN}✓ Trading system stopped${NC}"