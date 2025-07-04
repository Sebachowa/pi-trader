#!/usr/bin/env python3
"""
Monitor live trading bot activity and diagnose issues
"""
import subprocess
import time
import re
from datetime import datetime

def tail_log(log_file, lines=100):
    """Tail the log file and analyze in real-time"""
    print(f"🔍 Monitoring: {log_file}")
    print("=" * 60)
    print("Press Ctrl+C to stop\n")
    
    # Statistics
    start_time = datetime.now()
    scan_count = 0
    total_opportunities = 0
    last_opportunity_time = None
    opportunities_details = []
    
    try:
        # Use tail -f to follow the log
        process = subprocess.Popen(
            ['tail', '-f', '-n', str(lines), log_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Patterns
        scan_pattern = r"Scan completed in ([\d.]+)s, found (\d+) opportunities"
        opportunity_pattern = r"OPPORTUNITY FOUND! (\S+) - (\w+) \(score: ([\d.]+)\)"
        processed_pattern = r"Processed opportunity: (\S+) - (\w+) \(score: ([\d.]+)\)"
        position_pattern = r"TRADE OPENED: (\S+)"
        error_pattern = r"ERROR|Failed"
        
        for line in process.stdout:
            line = line.strip()
            
            # Print with color coding
            if "ERROR" in line or "Failed" in line:
                print(f"❌ {line}")
            elif "WARNING" in line:
                print(f"⚠️  {line}")
            elif "OPPORTUNITY FOUND" in line:
                print(f"💡 {line}")
            elif "TRADE OPENED" in line:
                print(f"💰 {line}")
            elif "Scan completed" in line:
                print(f"🔍 {line}")
            else:
                print(f"   {line}")
            
            # Analyze patterns
            scan_match = re.search(scan_pattern, line)
            if scan_match:
                scan_count += 1
                scan_time = float(scan_match.group(1))
                found = int(scan_match.group(2))
                total_opportunities += found
                
                # Calculate time since last opportunity
                if last_opportunity_time and found > 0:
                    last_opportunity_time = datetime.now()
                
                # Show running statistics
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                print(f"\n📊 STATS after {elapsed:.1f} minutes:")
                print(f"   Scans: {scan_count}")
                print(f"   Total opportunities: {total_opportunities}")
                print(f"   Avg opportunities/scan: {total_opportunities/scan_count:.2f}")
                
                if total_opportunities == 0 and scan_count > 5:
                    print("\n⚠️  NO OPPORTUNITIES FOUND - Possible issues:")
                    print("   1. Score threshold too high (check config.json)")
                    print("   2. Low market volatility")
                    print("   3. Volume requirements too strict")
                    print("   4. All strategies failing conditions")
                print()
            
            # Track opportunity details
            opp_match = re.search(opportunity_pattern, line)
            if opp_match:
                symbol = opp_match.group(1)
                strategy = opp_match.group(2)
                score = float(opp_match.group(3))
                opportunities_details.append({
                    'time': datetime.now(),
                    'symbol': symbol,
                    'strategy': strategy,
                    'score': score
                })
                
            # Check if opportunities are being processed
            proc_match = re.search(processed_pattern, line)
            if proc_match:
                symbol = proc_match.group(1)
                print(f"\n✅ Processing opportunity for {symbol}")
            
            # Position opened
            if re.search(position_pattern, line):
                print("\n🎉 POSITION OPENED! Bot is working!")
                
    except KeyboardInterrupt:
        print("\n\n📊 Final Statistics:")
        print(f"   Total runtime: {(datetime.now() - start_time).total_seconds() / 60:.1f} minutes")
        print(f"   Total scans: {scan_count}")
        print(f"   Total opportunities: {total_opportunities}")
        
        if opportunities_details:
            print(f"\n💡 Opportunities found:")
            for opp in opportunities_details[-10:]:  # Last 10
                print(f"   {opp['time'].strftime('%H:%M:%S')} - {opp['symbol']} "
                      f"({opp['strategy']}) score: {opp['score']:.1f}")
    
    finally:
        process.terminate()

def diagnose_no_opportunities():
    """Diagnose why no opportunities are being found"""
    print("\n🔧 DIAGNOSTICS: Why no opportunities?")
    print("=" * 60)
    
    print("\n1️⃣ CHECK CONFIGURATION (config/config.json):")
    print("   - min_opportunity_score: Should be 30-40 for testnet")
    print("   - min_volume_24h: Should be 10000-100000 for testnet")
    print("   - Scanner interval: 30-60 seconds is good")
    
    print("\n2️⃣ CHECK MARKET CONDITIONS:")
    print("   - Testnet has low volatility")
    print("   - Fewer traders = less price movement")
    print("   - Strategies need 0.2%+ moves")
    
    print("\n3️⃣ VERIFY TESTNET MODE:")
    print("   - config.json: \"testnet\": true")
    print("   - .env: BINANCE_TESTNET=true")
    print("   - Should see \"Testnet scanner initialized\"")
    
    print("\n4️⃣ COMMON FIXES:")
    print("   a) Lower min_opportunity_score to 30")
    print("   b) Reduce min_volume_24h to 10000")
    print("   c) Wait for market volatility")
    print("   d) Check API connectivity")
    
    print("\n5️⃣ TEST SCANNER DIRECTLY:")
    print("   python scripts/utils/debug_scanner.py")

if __name__ == "__main__":
    import argparse
    import glob
    import os
    
    parser = argparse.ArgumentParser(description='Monitor live bot activity')
    parser.add_argument('--log', help='Specific log file to monitor')
    parser.add_argument('--diagnose', action='store_true', 
                       help='Show diagnostics for no opportunities')
    
    args = parser.parse_args()
    
    if args.diagnose:
        diagnose_no_opportunities()
    else:
        # Find latest log file
        if args.log:
            log_file = args.log
        else:
            log_files = glob.glob('logs/trader_*.log')
            if not log_files:
                print("❌ No log files found in logs/")
                print("Make sure you're running from the project root")
                exit(1)
            log_file = max(log_files, key=os.path.getmtime)
        
        tail_log(log_file)