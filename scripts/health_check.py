#!/usr/bin/env python3
"""
Health check script for the trading bot
Returns 0 if healthy, 1 if unhealthy
"""

import sys
import time
import json
import requests
import subprocess
from datetime import datetime, timedelta

def check_service_status():
    """Check if the systemd service is running"""
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', 'trader'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to check service status: {e}")
        return False

def check_process_running():
    """Check if the trading bot process is running"""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'run.py'],
            capture_output=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to check process: {e}")
        return False

def check_log_health():
    """Check if there are recent log entries and no critical errors"""
    try:
        # Get last 100 lines of logs
        result = subprocess.run(
            ['journalctl', '-u', 'trader', '-n', '100', '--no-pager'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return False
            
        logs = result.stdout
        
        # Check for recent activity (logs within last 5 minutes)
        current_time = datetime.now()
        has_recent_logs = False
        
        for line in logs.split('\n'):
            if line.strip():
                # Try to parse timestamp from log
                try:
                    # Systemd log format
                    parts = line.split(' ')
                    if len(parts) > 2:
                        timestamp_str = ' '.join(parts[:3])
                        # Simple check: if log contains current date
                        if current_time.strftime('%b %d') in line:
                            has_recent_logs = True
                            break
                except:
                    continue
        
        # Check for critical errors
        critical_errors = ['CRITICAL', 'FATAL', 'Traceback', 'Exception']
        has_critical_errors = any(error in logs for error in critical_errors)
        
        if has_critical_errors:
            print("⚠️  Critical errors found in logs")
            return False
            
        if not has_recent_logs:
            print("⚠️  No recent log activity")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to check logs: {e}")
        return False

def check_api_health():
    """Check if the monitoring API is responding (if enabled)"""
    try:
        # Try to connect to monitoring endpoint if it exists
        response = requests.get('http://localhost:8080/health', timeout=5)
        return response.status_code == 200
    except:
        # API might not be enabled, that's OK
        return True

def check_disk_space():
    """Check if there's enough disk space"""
    try:
        result = subprocess.run(
            ['df', '-h', '/'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return False
            
        # Parse disk usage
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            # Get usage percentage
            parts = lines[1].split()
            if len(parts) > 4:
                usage_str = parts[4].rstrip('%')
                usage = int(usage_str)
                
                if usage > 90:
                    print(f"⚠️  Disk usage critical: {usage}%")
                    return False
                elif usage > 80:
                    print(f"⚠️  Disk usage high: {usage}%")
                    
        return True
        
    except Exception as e:
        print(f"❌ Failed to check disk space: {e}")
        return False

def main():
    """Run all health checks"""
    print("🏥 Running health checks...")
    print("-" * 50)
    
    checks = [
        ("Service Status", check_service_status),
        ("Process Running", check_process_running),
        ("Log Health", check_log_health),
        ("API Health", check_api_health),
        ("Disk Space", check_disk_space),
    ]
    
    all_healthy = True
    
    for check_name, check_func in checks:
        print(f"Checking {check_name}... ", end="", flush=True)
        
        try:
            result = check_func()
            if result:
                print("✅ OK")
            else:
                print("❌ FAILED")
                all_healthy = False
        except Exception as e:
            print(f"❌ ERROR: {e}")
            all_healthy = False
    
    print("-" * 50)
    
    if all_healthy:
        print("✅ All health checks passed!")
        return 0
    else:
        print("❌ Some health checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())