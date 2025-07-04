"""
Configuration loader that combines JSON config with environment variables
Environment variables take precedence for sensitive data
"""
import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class ConfigLoader:
    """Load configuration from JSON and environment variables"""
    
    @staticmethod
    def load(config_path: str = "config/config.json") -> Dict[str, Any]:
        """
        Load configuration merging JSON file with environment variables
        Environment variables override JSON values for sensitive data
        """
        # Load base config from JSON
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Override with environment variables
        # Exchange configuration
        if os.getenv('BINANCE_API_KEY'):
            config['exchange']['api_key'] = os.getenv('BINANCE_API_KEY')
        if os.getenv('BINANCE_API_SECRET'):
            config['exchange']['api_secret'] = os.getenv('BINANCE_API_SECRET')
        if os.getenv('BINANCE_TESTNET'):
            config['exchange']['testnet'] = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        # Tax configuration
        if os.getenv('TAX_JURISDICTION'):
            config['tax']['jurisdiction'] = os.getenv('TAX_JURISDICTION')
        if os.getenv('TAX_YEAR'):
            config['tax']['tax_year'] = int(os.getenv('TAX_YEAR'))
        
        # Telegram configuration
        if os.getenv('TELEGRAM_BOT_TOKEN'):
            if 'telegram' not in config:
                config['telegram'] = {}
            config['telegram']['bot_token'] = os.getenv('TELEGRAM_BOT_TOKEN')
            config['telegram']['chat_id'] = os.getenv('TELEGRAM_CHAT_ID')
        
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate that all required configuration is present"""
        # Check if we're in demo mode
        is_demo_mode = os.getenv('DEMO_MODE', 'false').lower() == 'true'
        
        required = {
            'exchange': ['name', 'api_key', 'api_secret'],
            'trading': ['max_positions', 'position_size_pct'],
            'risk': ['max_drawdown_pct', 'max_position_size_usd'],
        }
        
        for section, keys in required.items():
            if section not in config:
                print(f"❌ Missing config section: {section}")
                return False
            
            for key in keys:
                if key not in config[section]:
                    print(f"❌ Missing config: {section}.{key}")
                    return False
                
                # Check for placeholder values (skip in demo mode)
                if not is_demo_mode:
                    value = config[section][key]
                    if isinstance(value, str) and ('YOUR_' in value or 'PLACEHOLDER_' in value):
                        print(f"⚠️  {section}.{key} has placeholder value")
                        print("   Run with DEMO_MODE=true or update .env file")
                        return False
        
        if is_demo_mode:
            print("🎮 Running in DEMO MODE - API keys not required")
        
        return True