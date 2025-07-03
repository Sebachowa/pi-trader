# 🚀 Automated Deployment Guide

This guide will help you set up automated deployment from GitHub to your Raspberry Pi.

## 📋 Prerequisites

- Raspberry Pi with SSH access
- GitHub repository for your trading bot
- GitHub account with Actions enabled

## 🔧 Initial Setup

### 1. Prepare your Raspberry Pi

SSH into your Raspberry Pi and run:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install python3 python3-pip python3-venv git -y

# Create deployment directory
sudo mkdir -p /home/pi/trader-bot
sudo chown pi:pi /home/pi/trader-bot
```

### 2. Generate SSH Keys for GitHub Actions

On your local machine:

```bash
# Run the setup script
chmod +x scripts/setup_github_secrets.sh
./scripts/setup_github_secrets.sh
```

This will:
- Generate an SSH key pair for GitHub Actions
- Show you the public key to add to your Pi
- Show you the private key to add to GitHub Secrets

### 3. Add SSH Key to Raspberry Pi

Copy the public key to your Pi:

```bash
# Replace PI_USER and PI_HOST with your values
ssh pi@raspberrypi.local 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys' < ~/.ssh/github_actions_deploy.pub
```

### 4. Configure GitHub Secrets

Go to your GitHub repository:
1. Navigate to Settings → Secrets and variables → Actions
2. Add these secrets:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `PI_HOST` | Your Pi's hostname or IP | `192.168.1.100` |
| `PI_USER` | SSH username | `pi` |
| `PI_PORT` | SSH port | `22` |
| `PI_SSH_KEY` | Private SSH key content | *(paste entire key)* |
| `CONFIG_JSON` | Trading config (optional) | *(your config.json content)* |
| `TELEGRAM_BOT_TOKEN` | For notifications (optional) | `123456:ABC-DEF...` |
| `TELEGRAM_CHAT_ID` | Your Telegram ID (optional) | `123456789` |

## 🎯 Deployment Workflow

### Automatic Deployment

The bot will automatically deploy when you:
- Push to `main` branch
- Push to `deploy` branch
- Manually trigger the workflow

### Manual Deployment

To manually trigger a deployment:
1. Go to Actions tab in your repository
2. Select "Deploy to Raspberry Pi" workflow
3. Click "Run workflow"
4. Select environment (production/staging/test)

## 🔄 Deployment Process

The automated deployment will:

1. **Run Tests** - Ensure code quality
2. **Create Backup** - Backup current version
3. **Deploy Code** - Copy new code to Pi
4. **Install Dependencies** - Update Python packages
5. **Update Configuration** - Apply config from secrets
6. **Restart Service** - Start the trading bot
7. **Health Check** - Verify deployment success
8. **Send Notification** - Notify via Telegram

## 📊 Monitoring

### View Logs

```bash
# Real-time logs
ssh pi@raspberrypi.local "journalctl -u trader -f"

# Last 100 lines
ssh pi@raspberrypi.local "journalctl -u trader -n 100"
```

### Check Status

```bash
# Service status
ssh pi@raspberrypi.local "systemctl status trader"

# Run health check
ssh pi@raspberrypi.local "cd /home/pi/trader-bot && python3 scripts/health_check.py"
```

## 🔄 Rollback

If deployment fails, the system automatically:
1. Stops the service
2. Restores the previous backup
3. Restarts the service
4. Sends failure notification

### Manual Rollback

```bash
ssh pi@raspberrypi.local
cd /home/pi
sudo systemctl stop trader

# List backups
ls -la trader-bot_backup_*

# Restore specific backup
rm -rf trader-bot
cp -r trader-bot_backup_20240102_120000 trader-bot
sudo systemctl start trader
```

## 🛠️ Troubleshooting

### Deployment Fails

1. Check GitHub Actions logs
2. Verify SSH connection:
   ```bash
   ssh -i ~/.ssh/github_actions_deploy pi@raspberrypi.local
   ```
3. Check Pi logs:
   ```bash
   ssh pi@raspberrypi.local "journalctl -u trader -n 50"
   ```

### Service Won't Start

1. Check Python version:
   ```bash
   python3 --version  # Should be 3.9+
   ```
2. Check dependencies:
   ```bash
   cd /home/pi/trader-bot
   source venv/bin/activate
   pip list
   ```
3. Test manually:
   ```bash
   python3 run.py --dry-run
   ```

### Permission Issues

```bash
# Fix ownership
sudo chown -R pi:pi /home/pi/trader-bot

# Fix service permissions
sudo systemctl daemon-reload
```

## 🔐 Security Best Practices

1. **Use strong SSH keys** - ED25519 recommended
2. **Limit SSH access** - Use firewall rules
3. **Rotate keys regularly** - Update every 3-6 months
4. **Monitor access logs** - Check for unauthorized access
5. **Use secrets for sensitive data** - Never commit credentials

## 📈 Advanced Features

### Multiple Environments

Create different branches for different environments:
- `main` → Production
- `staging` → Staging environment
- `develop` → Development

### Slack/Discord Notifications

Replace Telegram with your preferred platform in the workflow:

```yaml
- name: Send Discord notification
  uses: sarisia/actions-status-discord@v1
  if: always()
  with:
    webhook: ${{ secrets.DISCORD_WEBHOOK }}
```

### Database Backups

Add to deployment workflow:

```yaml
- name: Backup database
  run: |
    ssh pi@raspberrypi.local "cd /home/pi/trader-bot && ./scripts/backup_db.sh"
```

## 🎉 Success!

Once configured, every push to your repository will automatically:
- Test your code
- Deploy to your Raspberry Pi
- Start the trading bot
- Send you a notification

Happy automated trading! 🤖📈