#!/bin/bash
# Quick check why bot isn't finding opportunities

echo "🔍 Quick Bot Status Check"
echo "========================="

# Check if bot is running
echo -e "\n1️⃣ Bot Process:"
if pgrep -f "python.*run.py" > /dev/null; then
    echo "   ✅ Bot is running"
    ps aux | grep "python.*run.py" | grep -v grep
else
    echo "   ❌ Bot is NOT running"
fi

# Check latest log
echo -e "\n2️⃣ Latest Log Activity:"
LATEST_LOG=$(ls -t logs/trader_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "   📄 Log: $LATEST_LOG"
    echo "   📊 Last 5 scans:"
    grep "Scan completed" "$LATEST_LOG" | tail -5
    
    # Count opportunities in last hour
    echo -e "\n3️⃣ Opportunities in Last Hour:"
    HOUR_AGO=$(date -d '1 hour ago' '+%Y-%m-%d %H:%M' 2>/dev/null || date -v-1H '+%Y-%m-%d %H:%M')
    grep "OPPORTUNITY FOUND" "$LATEST_LOG" | grep "$HOUR_AGO" | wc -l | xargs echo "   Found:"
    
    # Show last opportunities
    echo -e "\n4️⃣ Last 3 Opportunities:"
    grep "OPPORTUNITY FOUND" "$LATEST_LOG" | tail -3 | while read line; do
        echo "   $line"
    done
    
    # Check for errors
    echo -e "\n5️⃣ Recent Errors:"
    ERROR_COUNT=$(grep -i "error\|failed" "$LATEST_LOG" | tail -20 | wc -l)
    echo "   Errors in last 20 lines: $ERROR_COUNT"
    if [ $ERROR_COUNT -gt 0 ]; then
        echo "   Last error:"
        grep -i "error\|failed" "$LATEST_LOG" | tail -1
    fi
else
    echo "   ❌ No log files found"
fi

# Check configuration
echo -e "\n6️⃣ Configuration Check:"
if [ -f "config/config.json" ]; then
    echo "   Min score: $(grep min_opportunity_score config/config.json | grep -o '[0-9]*')"
    echo "   Min volume: $(grep min_volume_24h config/config.json | head -1 | grep -o '[0-9]*')"
    echo "   Testnet: $(grep testnet config/config.json | head -1 | grep -o 'true\|false')"
else
    echo "   ❌ config.json not found"
fi

# Suggestions
echo -e "\n💡 Quick Fixes:"
echo "   1. Check min_opportunity_score (should be 30-40 for testnet)"
echo "   2. Verify testnet=true in config"  
echo "   3. Run: python scripts/utils/debug_scanner.py"
echo "   4. Monitor live: python scripts/utils/monitor_live.py"
echo "   5. Full analysis: python scripts/utils/analyze_logs.py --latest"