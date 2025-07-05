#!/bin/bash
# Setup GitHub Actions runner on Raspberry Pi

echo "🚀 Setting up GitHub Actions Runner on Raspberry Pi"
echo "=================================================="

# Go to your repo: https://github.com/YOUR_USERNAME/pi-trader
# Click: Settings → Actions → Runners → New self-hosted runner
# Choose: Linux → ARM64

echo "📋 Follow these steps:"
echo ""
echo "1. Go to: https://github.com/YOUR_USERNAME/pi-trader/settings/actions/runners/new"
echo "2. Select: Linux and ARM64"
echo "3. Copy the download and configure commands shown there"
echo ""
echo "4. SSH to your Pi and run:"
echo "   cd ~"
echo "   mkdir actions-runner && cd actions-runner"
echo "   # Paste the download command from GitHub"
echo "   # Paste the configure command from GitHub"
echo ""
echo "5. Install as service:"
echo "   sudo ./svc.sh install"
echo "   sudo ./svc.sh start"
echo ""
echo "6. Push to GitHub and it will deploy automatically!"
echo ""
echo "Benefits:"
echo "✅ No SSH keys needed in GitHub"
echo "✅ No public IP needed"
echo "✅ Pi connects to GitHub (not reverse)"
echo "✅ Keep using Tailscale for remote access"
echo "✅ Super secure!"