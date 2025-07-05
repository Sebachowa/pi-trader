#!/bin/bash
# Simple deploy script
echo "🚀 Deploying to Raspberry Pi..."

# Pull latest changes
echo "📥 Pulling latest code..."
ssh sebachowa@100.107.63.44 "cd ~/code/pi-trader && git pull origin main"

# Restart service
echo "🔄 Restarting bot..."
ssh sebachowa@100.107.63.44 "sudo systemctl restart trader || echo 'Service not configured yet'"

echo "✅ Done! Check logs with:"
echo "ssh sebachowa@100.107.63.44 'sudo journalctl -u trader -f'"