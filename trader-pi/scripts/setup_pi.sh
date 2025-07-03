#!/bin/bash
# Initial setup script for Raspberry Pi

set -e

echo "🍓 Raspberry Pi Trading Bot Setup"
echo "================================"

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required packages
echo "📦 Installing required packages..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    curl \
    wget \
    htop \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-setuptools

# Install Docker (optional)
read -p "Install Docker? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    
    # Install Docker Compose
    echo "🐳 Installing Docker Compose..."
    sudo apt install -y docker-compose
    
    # Enable Docker on boot
    sudo systemctl enable docker
    echo "✅ Docker installed (logout and login again to use without sudo)"
fi

# Setup deployment directory
echo "📁 Creating deployment directory..."
sudo mkdir -p /home/pi/trader-bot
sudo chown pi:pi /home/pi/trader-bot

# Setup log directory
echo "📁 Creating log directory..."
mkdir -p /home/pi/trader-bot/logs
mkdir -p /home/pi/trader-bot/data

# Configure systemd service directory
echo "⚙️ Configuring systemd..."
sudo mkdir -p /etc/systemd/system

# Set up logrotate for trading bot logs
echo "📝 Setting up log rotation..."
sudo tee /etc/logrotate.d/trader-bot > /dev/null <<EOF
/home/pi/trader-bot/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 pi pi
}
EOF

# Optimize Raspberry Pi for trading bot
echo "🔧 Optimizing Raspberry Pi settings..."

# Increase swap size (useful for Pi with low RAM)
read -p "Increase swap size to 2GB? (recommended for Pi with <4GB RAM) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo dphys-swapfile swapoff
    sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
    sudo dphys-swapfile setup
    sudo dphys-swapfile swapon
    echo "✅ Swap size increased to 2GB"
fi

# Set up firewall
read -p "Configure UFW firewall? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo apt install -y ufw
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    sudo ufw allow ssh
    sudo ufw allow 8080/tcp  # For monitoring API
    sudo ufw allow 3000/tcp  # For Grafana (if using)
    sudo ufw --force enable
    echo "✅ Firewall configured"
fi

# Install monitoring tools
echo "📊 Installing monitoring tools..."
sudo apt install -y \
    iotop \
    nmon \
    ncdu

# Create helper scripts
echo "📝 Creating helper scripts..."

# Create status check script
cat > ~/check-trader.sh << 'EOF'
#!/bin/bash
echo "Trading Bot Status"
echo "=================="
echo ""
echo "Service Status:"
sudo systemctl status trader --no-pager
echo ""
echo "Last 20 log lines:"
sudo journalctl -u trader -n 20 --no-pager
echo ""
echo "Disk Usage:"
df -h /
echo ""
echo "Memory Usage:"
free -h
echo ""
echo "CPU Temperature:"
vcgencmd measure_temp
EOF

chmod +x ~/check-trader.sh

# Create restart script
cat > ~/restart-trader.sh << 'EOF'
#!/bin/bash
echo "Restarting Trading Bot..."
sudo systemctl restart trader
sleep 3
sudo systemctl status trader --no-pager
EOF

chmod +x ~/restart-trader.sh

# Set up cron for auto-restart on reboot
echo "⏰ Setting up auto-start on boot..."
(crontab -l 2>/dev/null; echo "@reboot sleep 30 && sudo systemctl start trader") | crontab -

# Performance tuning
echo "🚀 Applying performance tuning..."

# Disable unnecessary services
sudo systemctl disable bluetooth.service || true
sudo systemctl disable avahi-daemon.service || true

# Configure watchdog (auto-reboot on hang)
read -p "Enable hardware watchdog? (auto-reboot on system hang) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo apt install -y watchdog
    echo "bcm2835_wdt" | sudo tee -a /etc/modules
    sudo systemctl enable watchdog
    echo "✅ Watchdog enabled"
fi

echo ""
echo "✅ Raspberry Pi setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Copy your trading bot code to /home/pi/trader-bot/"
echo "2. Install the systemd service file"
echo "3. Start the trading bot with: sudo systemctl start trader"
echo ""
echo "🛠️ Useful commands:"
echo "- Check status: ~/check-trader.sh"
echo "- Restart bot: ~/restart-trader.sh"
echo "- View logs: journalctl -u trader -f"
echo ""
echo "🔄 Please reboot your Pi to apply all changes: sudo reboot"