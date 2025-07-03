#!/bin/bash
#
# Deployment script for Raspberry Pi trading bot
# Usage: ./deploy_to_pi.sh [PI_HOST] [PI_USER]
#

set -e

# Configuration
PI_HOST=${1:-"raspberrypi.local"}
PI_USER=${2:-"pi"}
REMOTE_DIR="/home/${PI_USER}/trader-pi"
SERVICE_NAME="trader"

echo "======================================="
echo "Deploying Trading Bot to Raspberry Pi"
echo "Host: ${PI_HOST}"
echo "User: ${PI_USER}"
echo "Remote Dir: ${REMOTE_DIR}"
echo "======================================="

# Create remote directory
echo "Creating remote directory..."
ssh ${PI_USER}@${PI_HOST} "mkdir -p ${REMOTE_DIR}"

# Copy files to Pi
echo "Copying files to Pi..."
rsync -avz --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='logs/' \
    --exclude='stats.json' \
    --exclude='.git' \
    ./ ${PI_USER}@${PI_HOST}:${REMOTE_DIR}/

# Install dependencies on Pi
echo "Installing dependencies on Pi..."
ssh ${PI_USER}@${PI_HOST} << 'EOF'
    cd ~/trader-pi
    
    # Update system
    sudo apt-get update
    
    # Install Python dependencies
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements-pi.txt
    
    # Create logs directory
    mkdir -p logs
    
    # Make scripts executable
    chmod +x run.py
    chmod +x deploy_to_pi.sh
EOF

# Install systemd service
echo "Installing systemd service..."
scp trader.service ${PI_USER}@${PI_HOST}:/tmp/
ssh ${PI_USER}@${PI_HOST} << EOF
    sudo mv /tmp/trader.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable ${SERVICE_NAME}.service
    echo "Service installed. Use 'sudo systemctl start ${SERVICE_NAME}' to start the bot."
EOF

echo "======================================="
echo "Deployment complete!"
echo ""
echo "To manage the trading bot:"
echo "  Start:   sudo systemctl start ${SERVICE_NAME}"
echo "  Stop:    sudo systemctl stop ${SERVICE_NAME}"
echo "  Status:  sudo systemctl status ${SERVICE_NAME}"
echo "  Logs:    sudo journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "To run manually:"
echo "  ssh ${PI_USER}@${PI_HOST}"
echo "  cd ${REMOTE_DIR}"
echo "  python3 run.py"
echo "======================================="