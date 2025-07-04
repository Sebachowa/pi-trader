#!/usr/bin/env python3
"""
Check configuration before deployment
Ensures no API keys are exposed in commits
"""
import sys
import json
import os
from pathlib import Path

def check_config_files():
    """Check that no sensitive data is in config files"""
    print("🔍 Checking configuration files...")
    
    errors = []
    
    # Check config.json
    config_path = Path("config/config.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        # Check for real API keys
        api_key = config.get('exchange', {}).get('api_key', '')
        api_secret = config.get('exchange', {}).get('api_secret', '')
        
        if api_key and not api_key.startswith('YOUR_'):
            errors.append("❌ Real API key found in config.json!")
        
        if api_secret and not api_secret.startswith('YOUR_'):
            errors.append("❌ Real API secret found in config.json!")
    
    # Check .env is not being tracked
    if os.system("git ls-files .env | grep -q .env") == 0:
        errors.append("❌ .env file is being tracked by git!")
    
    # Check telegram config
    telegram_path = Path("telegram_config.json")
    if telegram_path.exists():
        if os.system(f"git ls-files {telegram_path} | grep -q {telegram_path}") == 0:
            errors.append("❌ telegram_config.json is being tracked!")
    
    # Check .env exists for local development
    if not Path(".env").exists() and not Path(".env.example").exists():
        print("⚠️  No .env file found. Create one from .env.example")
    
    if errors:
        print("\n".join(errors))
        print("\n🛑 Configuration check failed!")
        print("Fix these issues before committing.")
        return False
    else:
        print("✅ Configuration check passed!")
        print("✅ No API keys exposed in git")
        return True

if __name__ == "__main__":
    if not check_config_files():
        sys.exit(1)