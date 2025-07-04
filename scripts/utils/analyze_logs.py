#!/usr/bin/env python3
"""
Analyze trading bot logs to understand why no opportunities are found
"""
import re
import sys
from datetime import datetime
from collections import defaultdict, Counter

def analyze_log_file(log_file):
    """Analyze a single log file for patterns"""
    print(f"\n📊 Analyzing: {log_file}")
    print("=" * 60)
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ File not found: {log_file}")
        return
    
    # Statistics
    total_lines = len(lines)
    scan_count = 0
    opportunities_found = 0
    errors = []
    warnings = []
    positions_opened = 0
    positions_closed = 0
    
    # Patterns to look for
    scan_pattern = r"Scan completed in ([\d.]+)s, found (\d+) opportunities"
    opportunity_pattern = r"OPPORTUNITY FOUND! (\S+) - (\w+) \(score: ([\d.]+)\)"
    error_pattern = r"ERROR|CRITICAL|Failed"
    warning_pattern = r"WARNING|warning"
    position_opened_pattern = r"TRADE OPENED|Opened position"
    position_closed_pattern = r"TRADE CLOSED|Closed position"
    processed_opportunity_pattern = r"Processed opportunity: (\S+) - (\w+) \(score: ([\d.]+)\)"
    
    # Opportunity details
    opportunities_by_symbol = defaultdict(list)
    opportunities_by_strategy = defaultdict(list)
    scores = []
    scan_times = []
    
    # Process lines
    for i, line in enumerate(lines):
        # Count scans and opportunities
        scan_match = re.search(scan_pattern, line)
        if scan_match:
            scan_count += 1
            scan_time = float(scan_match.group(1))
            found = int(scan_match.group(2))
            opportunities_found += found
            scan_times.append(scan_time)
        
        # Track opportunities
        opp_match = re.search(opportunity_pattern, line)
        if opp_match:
            symbol = opp_match.group(1)
            strategy = opp_match.group(2)
            score = float(opp_match.group(3))
            opportunities_by_symbol[symbol].append(score)
            opportunities_by_strategy[strategy].append(score)
            scores.append(score)
        
        # Track processed opportunities
        proc_match = re.search(processed_opportunity_pattern, line)
        if proc_match:
            symbol = proc_match.group(1)
            strategy = proc_match.group(2)
            score = float(proc_match.group(3))
            print(f"   📋 Processed: {symbol} - {strategy} (score: {score})")
        
        # Count positions
        if re.search(position_opened_pattern, line):
            positions_opened += 1
            print(f"   💰 Position opened: {line.strip()}")
        
        if re.search(position_closed_pattern, line):
            positions_closed += 1
            print(f"   📊 Position closed: {line.strip()}")
        
        # Collect errors and warnings
        if re.search(error_pattern, line, re.IGNORECASE):
            errors.append((i+1, line.strip()))
        
        if re.search(warning_pattern, line, re.IGNORECASE):
            warnings.append((i+1, line.strip()))
    
    # Print summary
    print(f"\n📈 SUMMARY:")
    print(f"   Total lines: {total_lines:,}")
    print(f"   Total scans: {scan_count}")
    print(f"   Opportunities found: {opportunities_found}")
    print(f"   Positions opened: {positions_opened}")
    print(f"   Positions closed: {positions_closed}")
    
    if scan_count > 0:
        print(f"\n⏱️  SCAN PERFORMANCE:")
        print(f"   Average scan time: {sum(scan_times)/len(scan_times):.2f}s")
        print(f"   Min scan time: {min(scan_times):.2f}s")
        print(f"   Max scan time: {max(scan_times):.2f}s")
        print(f"   Opportunities per scan: {opportunities_found/scan_count:.2f}")
    
    if scores:
        print(f"\n🎯 OPPORTUNITY SCORES:")
        print(f"   Average score: {sum(scores)/len(scores):.1f}")
        print(f"   Min score: {min(scores):.1f}")
        print(f"   Max score: {max(scores):.1f}")
        print(f"   Scores distribution:")
        for threshold in [30, 40, 50, 60, 70, 80, 90]:
            count = len([s for s in scores if s >= threshold])
            print(f"      Score >= {threshold}: {count} opportunities")
    
    if opportunities_by_symbol:
        print(f"\n💹 TOP SYMBOLS:")
        sorted_symbols = sorted(opportunities_by_symbol.items(), 
                              key=lambda x: len(x[1]), reverse=True)[:10]
        for symbol, symbol_scores in sorted_symbols:
            avg_score = sum(symbol_scores) / len(symbol_scores)
            print(f"   {symbol}: {len(symbol_scores)} opportunities (avg score: {avg_score:.1f})")
    
    if opportunities_by_strategy:
        print(f"\n🎲 STRATEGIES:")
        for strategy, strategy_scores in opportunities_by_strategy.items():
            avg_score = sum(strategy_scores) / len(strategy_scores)
            print(f"   {strategy}: {len(strategy_scores)} opportunities (avg score: {avg_score:.1f})")
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for line_num, error in errors[:5]:  # Show first 5
            print(f"   Line {line_num}: {error[:100]}...")
        if len(errors) > 5:
            print(f"   ... and {len(errors)-5} more errors")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for line_num, warning in warnings[:5]:  # Show first 5
            print(f"   Line {line_num}: {warning[:100]}...")
        if len(warnings) > 5:
            print(f"   ... and {len(warnings)-5} more warnings")

def find_latest_log():
    """Find the latest log file"""
    import glob
    import os
    
    log_files = glob.glob('logs/trader_*.log')
    if not log_files:
        return None
    
    # Sort by modification time
    log_files.sort(key=os.path.getmtime, reverse=True)
    return log_files[0]

def analyze_all_logs():
    """Analyze all log files"""
    import glob
    
    log_files = sorted(glob.glob('logs/trader_*.log'))
    
    if not log_files:
        print("❌ No log files found in logs/ directory")
        print("\n💡 Make sure you're running this from the project root directory")
        print("   and that the bot has been running and generating logs.")
        return
    
    print(f"📁 Found {len(log_files)} log files")
    
    # Aggregate statistics
    total_scans = 0
    total_opportunities = 0
    total_positions = 0
    all_scores = []
    
    for log_file in log_files:
        analyze_log_file(log_file)
    
    # Overall summary
    print("\n" + "=" * 60)
    print("📊 OVERALL ANALYSIS")
    print("=" * 60)

def check_current_status():
    """Check current bot status from logs"""
    latest_log = find_latest_log()
    if not latest_log:
        print("❌ No log files found")
        return
    
    print(f"\n🔍 Checking latest activity in: {latest_log}")
    
    # Get last 50 lines
    with open(latest_log, 'r') as f:
        lines = f.readlines()[-50:]
    
    print("\n📜 Last 50 log entries:")
    print("-" * 60)
    for line in lines:
        print(line.strip())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze trading bot logs')
    parser.add_argument('--file', help='Specific log file to analyze')
    parser.add_argument('--all', action='store_true', help='Analyze all log files')
    parser.add_argument('--latest', action='store_true', help='Analyze latest log file')
    parser.add_argument('--status', action='store_true', help='Show current status')
    
    args = parser.parse_args()
    
    if args.file:
        analyze_log_file(args.file)
    elif args.all:
        analyze_all_logs()
    elif args.status:
        check_current_status()
    elif args.latest:
        latest = find_latest_log()
        if latest:
            analyze_log_file(latest)
        else:
            print("❌ No log files found")
    else:
        # Default: analyze latest
        latest = find_latest_log()
        if latest:
            analyze_log_file(latest)
        else:
            analyze_all_logs()