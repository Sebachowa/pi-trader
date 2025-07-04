# 🚀 Deployment Guide

Complete guide for deploying your trading bot to production environments.

## 🎯 Deployment Options

| Method | Best For | Complexity | Automation |
|--------|----------|------------|------------|
| [Manual](#-manual-deployment) | Testing, learning | Low | None |
| [Systemd Service](#-systemd-service) | Raspberry Pi, Linux | Medium | Basic |
| [GitHub Actions](#-github-actions-automated) | Production, teams | High | Full |

## 📋 Prerequisites

### For Raspberry Pi
- Raspberry Pi 4+ (2GB+ RAM recommended)
- Raspberry Pi OS 64-bit
- Stable internet connection
- SSH access enabled

### For Development Machine
- Git configured with your repository
- GitHub account (for automated deployment)
- SSH client

## 🔧 Manual Deployment

Perfect for testing and understanding the deployment process.

### 1. Prepare Target System
```bash
# Update system (Raspberry Pi/Linux)
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install python3 python3-pip python3-venv git htop -y

# Create bot directory
mkdir -p ~/trading-bot
cd ~/trading-bot
```

### 2. Copy Files
```bash
# Option A: Git clone (recommended)
git clone https://github.com/YourUsername/pi-trader.git .

# Option B: Copy from local machine
rsync -avz --exclude 'venv' --exclude '__pycache__' \
  ./ pi@192.168.1.100:~/trading-bot/
```

### 3. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
./scripts/deployment/setup_pi.sh
# Or manually: pip install -r requirements.txt
```

### 4. Configure
```bash
# Create .env file
cp .env.example .env
nano .env

# Add your API keys
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=true
```

### 5. Test Run
```bash
# Test in foreground
python run.py --demo  # Safe testing

# Check logs
tail -f logs/trader_*.log
```

## 🏃 Systemd Service

Run the bot as a system service for 24/7 operation.

### 1. Create Service File
```bash
sudo nano /etc/systemd/system/trading-bot.service
```

Add this content:
```ini
[Unit]
Description=Cryptocurrency Trading Bot
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/trading-bot
Environment=PATH=/home/pi/trading-bot/venv/bin
ExecStart=/home/pi/trading-bot/venv/bin/python run.py
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=trading-bot

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/pi/trading-bot/logs

[Install]
WantedBy=multi-user.target
```

### 2. Enable and Start Service
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable trading-bot

# Start service
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot
```

### 3. Service Management
```bash
# View logs
sudo journalctl -u trading-bot -f

# Stop service
sudo systemctl stop trading-bot

# Restart service
sudo systemctl restart trading-bot

# Disable auto-start
sudo systemctl disable trading-bot
```

## 🤖 GitHub Actions (Automated)

Fully automated deployment with testing and rollback capabilities.

### 1. Setup SSH Access

On your development machine:
```bash
# Generate deployment key
ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/github_actions_deploy

# Copy public key to Pi
ssh-copy-id -i ~/.ssh/github_actions_deploy.pub pi@your-pi-ip
```

### 2. Configure GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these secrets:

| Secret | Value | Description |
|--------|-------|-------------|
| `PI_HOST` | `192.168.1.100` | Your Pi's IP address |
| `PI_USER` | `pi` | SSH username |
| `PI_SSH_KEY` | `(private key content)` | Content of ~/.ssh/github_actions_deploy |
| `BINANCE_API_KEY` | `your_api_key` | Your Binance API key |
| `BINANCE_API_SECRET` | `your_secret` | Your Binance secret |
| `BINANCE_TESTNET` | `true` | Use testnet |
| `TELEGRAM_BOT_TOKEN` | `123:ABC...` | (Optional) Telegram notifications |

### 3. Deployment Workflow

The GitHub Actions workflow (`.github/workflows/deploy.yml`) will:

1. ✅ **Run tests** - Ensure code quality
2. 🔒 **Create backup** - Backup current version
3. 📦 **Deploy code** - Copy new code to Pi
4. 🔧 **Install deps** - Update dependencies
5. ⚙️ **Configure** - Update .env from secrets
6. 🔄 **Restart service** - Restart the bot
7. 🏥 **Health check** - Verify deployment
8. 📱 **Notify** - Send status notification

### 4. Trigger Deployment

Deployment triggers automatically on:
- Push to `main` branch
- Push to `deploy` branch
- Manual workflow dispatch

Manual trigger:
1. Go to repository → Actions
2. Select "Deploy to Raspberry Pi"
3. Click "Run workflow"
4. Choose environment

## 📊 Monitoring & Maintenance

### System Monitoring
```bash
# Check system resources
htop

# Check disk space
df -h

# Check memory usage
free -h

# Check service status
sudo systemctl status trading-bot
```

### Log Management
```bash
# View live logs
tail -f logs/trader_*.log

# Check service logs
sudo journalctl -u trading-bot --since "1 hour ago"

# Log rotation (prevent disk full)
sudo nano /etc/logrotate.d/trading-bot
```

Add log rotation config:
```
/home/pi/trading-bot/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
```

### Health Checks
```bash
# Check if bot is running
ps aux | grep python

# Check network connectivity
ping api.binance.com

# Check API connectivity
python tests/test_binance_testnet.py

# Check disk space
df -h /home
```

### Updates & Maintenance
```bash
# Update code (manual)
cd ~/trading-bot
git pull origin main
sudo systemctl restart trading-bot

# Update dependencies
source venv/bin/activate
pip install -r requirements-pi.txt --upgrade

# Check for issues
sudo systemctl status trading-bot
sudo journalctl -u trading-bot -n 50
```

## 🚨 Troubleshooting

### Service Won't Start
```bash
# Check service status
sudo systemctl status trading-bot

# Check logs
sudo journalctl -u trading-bot -n 50

# Common issues:
# 1. Path problems in service file
# 2. Permission issues
# 3. Missing dependencies
# 4. Invalid configuration
```

### Performance Issues
```bash
# Check CPU/Memory
htop

# Reduce scanner frequency
nano config/config.json
# Set "interval_seconds": 60
# Set "max_concurrent_scans": 20

# Restart with new config
sudo systemctl restart trading-bot
```

### Connection Issues
```bash
# Test internet
ping google.com

# Test Binance
curl -s "https://api.binance.com/api/v3/ping"

# Check firewall
sudo ufw status

# Check API keys
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Keys loaded' if os.getenv('BINANCE_API_KEY') else 'No keys')"
```

### GitHub Actions Issues
- Check secrets are set correctly
- Verify SSH key has correct permissions
- Ensure Pi is accessible from internet (if needed)
- Check GitHub Actions logs for specific errors

## 🔒 Security Considerations

### API Key Security
- Use dedicated API keys for the bot
- Enable only required permissions (Read + Spot Trading)
- Never enable withdrawal permissions
- Rotate keys regularly

### System Security
```bash
# Update system regularly
sudo apt update && sudo apt upgrade

# Configure firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow from 192.168.1.0/24  # Local network only

# Disable SSH password auth (use keys only)
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart ssh
```

### Network Security
- Use VPN for remote access
- Consider port forwarding alternatives
- Monitor access logs regularly
- Use fail2ban for SSH protection

---

**Next Steps:**
- ✅ Monitor your deployment for 24 hours
- ✅ Set up log monitoring and alerts
- ✅ Configure backup strategies
- ✅ Test failure scenarios and recovery