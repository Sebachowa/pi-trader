#!/bin/bash
# Check complete trading history from logs

echo "📊 TRADING BOT HISTORY ANALYSIS"
echo "==============================="

# Function to analyze a log file
analyze_log() {
    local logfile=$1
    local filename=$(basename "$logfile")
    
    echo -e "\n📄 Analyzing: $filename"
    echo "----------------------------------------"
    
    # Extract date from filename
    date_part=$(echo "$filename" | grep -o '[0-9]\{8\}_[0-9]\{6\}')
    
    # Count key events
    total_lines=$(wc -l < "$logfile")
    scans=$(grep -c "Scan completed" "$logfile")
    opportunities=$(grep "found [0-9]* opportunities" "$logfile" | grep -o '[0-9]*' | awk '{sum+=$1} END {print sum}')
    trades_opened=$(grep -c "TRADE OPENED\|Opened position" "$logfile")
    trades_closed=$(grep -c "TRADE CLOSED\|Closed position" "$logfile")
    errors=$(grep -ci "error\|failed" "$logfile")
    
    echo "   📈 Scans: $scans"
    echo "   💡 Opportunities: ${opportunities:-0}"
    echo "   💰 Trades opened: $trades_opened"
    echo "   📊 Trades closed: $trades_closed"
    echo "   ❌ Errors: $errors"
    
    # Show opportunities if any
    if [ "${opportunities:-0}" -gt 0 ]; then
        echo "   🎯 Opportunity details:"
        grep "OPPORTUNITY FOUND" "$logfile" | tail -3 | while read line; do
            echo "      $line" | cut -d' ' -f3-
        done
    fi
}

# Check all logs
LOG_COUNT=$(ls logs/trader_*.log 2>/dev/null | wc -l)

if [ $LOG_COUNT -eq 0 ]; then
    echo "❌ No log files found in logs/ directory"
    exit 1
fi

echo "📁 Found $LOG_COUNT log files"

# Overall statistics
TOTAL_SCANS=0
TOTAL_OPPORTUNITIES=0
TOTAL_TRADES=0

# Analyze each log
for logfile in logs/trader_*.log; do
    analyze_log "$logfile"
    
    # Add to totals
    scans=$(grep -c "Scan completed" "$logfile")
    opportunities=$(grep "found [0-9]* opportunities" "$logfile" | grep -o '[0-9]*' | awk '{sum+=$1} END {print sum}')
    trades=$(grep -c "TRADE OPENED\|Opened position" "$logfile")
    
    TOTAL_SCANS=$((TOTAL_SCANS + scans))
    TOTAL_OPPORTUNITIES=$((TOTAL_OPPORTUNITIES + ${opportunities:-0}))
    TOTAL_TRADES=$((TOTAL_TRADES + trades))
done

# Summary
echo -e "\n🎯 OVERALL SUMMARY"
echo "=================="
echo "   Total scans: $TOTAL_SCANS"
echo "   Total opportunities: $TOTAL_OPPORTUNITIES"
echo "   Total trades: $TOTAL_TRADES"

if [ $TOTAL_SCANS -gt 0 ]; then
    echo "   Opportunities per scan: $(echo "scale=2; $TOTAL_OPPORTUNITIES / $TOTAL_SCANS" | bc)"
fi

# Check current status
echo -e "\n🔍 CURRENT STATUS"
echo "================"
LATEST_LOG=$(ls -t logs/trader_*.log | head -1)
echo "Latest log: $(basename "$LATEST_LOG")"
echo "Last 10 activities:"
tail -10 "$LATEST_LOG" | grep -E "Scan completed|OPPORTUNITY|TRADE|ERROR" | while read line; do
    echo "   $line"
done

# Recommendations
echo -e "\n💡 ANALYSIS"
echo "==========="
if [ $TOTAL_OPPORTUNITIES -eq 0 ]; then
    echo "⚠️  NO OPPORTUNITIES FOUND - Check:"
    echo "   1. min_opportunity_score in config.json (try 30)"
    echo "   2. Testnet mode is enabled"
    echo "   3. Market volatility (testnet is quiet)"
    echo "   4. Run: python scripts/utils/debug_scanner.py"
elif [ $TOTAL_TRADES -eq 0 ] && [ $TOTAL_OPPORTUNITIES -gt 0 ]; then
    echo "⚠️  OPPORTUNITIES FOUND BUT NO TRADES - Check:"
    echo "   1. Risk management settings"
    echo "   2. Position sizing"
    echo "   3. API permissions"
    echo "   4. Balance availability"
else
    echo "✅ Bot is finding opportunities and trading"
fi

# Detailed analysis option
echo -e "\n📊 For detailed analysis run:"
echo "   python scripts/utils/analyze_logs.py --all"
echo "   python scripts/utils/monitor_live.py"