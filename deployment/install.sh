#!/bin/bash
#
# Autonomous Trading System Installation Script
# This script sets up the trading system on a Linux server
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
INSTALL_DIR="/opt/autonomous-trader"
SERVICE_USER="trader"
PYTHON_VERSION="3.11"

print_banner() {
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║        AUTONOMOUS TRADING SYSTEM - INSTALLATION               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo -e "${RED}This script must be run as root${NC}"
        exit 1
    fi
}

check_os() {
    if [[ ! -f /etc/os-release ]]; then
        echo -e "${RED}Cannot detect OS version${NC}"
        exit 1
    fi
    
    . /etc/os-release
    
    if [[ "$ID" != "ubuntu" && "$ID" != "debian" ]]; then
        echo -e "${YELLOW}Warning: This script is tested on Ubuntu/Debian. Your OS: $ID${NC}"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

install_dependencies() {
    echo -e "${YELLOW}Installing system dependencies...${NC}"
    
    apt-get update
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
        build-essential \
        git \
        curl \
        wget \
        htop \
        supervisor \
        nginx \
        postgresql \
        postgresql-contrib \
        redis-server \
        certbot \
        python3-certbot-nginx
        
    # Install Docker
    if ! command -v docker &> /dev/null; then
        echo "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        rm get-docker.sh
    fi
    
    # Install Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "Installing Docker Compose..."
        curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
    fi
    
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

create_user() {
    echo -e "${YELLOW}Creating system user...${NC}"
    
    if ! id "$SERVICE_USER" &>/dev/null; then
        useradd -m -s /bin/bash -d /home/$SERVICE_USER $SERVICE_USER
        usermod -aG docker $SERVICE_USER
    fi
    
    echo -e "${GREEN}✓ User '$SERVICE_USER' created${NC}"
}

setup_directories() {
    echo -e "${YELLOW}Setting up directories...${NC}"
    
    # Create installation directory
    mkdir -p $INSTALL_DIR
    
    # Create data directories
    mkdir -p $INSTALL_DIR/{data,logs,backups,config}
    
    # Set permissions
    chown -R $SERVICE_USER:$SERVICE_USER $INSTALL_DIR
    chmod 755 $INSTALL_DIR
    
    echo -e "${GREEN}✓ Directories created${NC}"
}

install_application() {
    echo -e "${YELLOW}Installing application...${NC}"
    
    # Copy application files
    if [ -d "../autonomous_trading" ]; then
        cp -r ../autonomous_trading $INSTALL_DIR/
        cp -r ../nautilus_challenge $INSTALL_DIR/
        cp ../requirements.txt $INSTALL_DIR/
        cp ../docker-compose.yml $INSTALL_DIR/
        cp ../Dockerfile $INSTALL_DIR/
    else
        echo -e "${RED}Application files not found. Please run from deployment directory.${NC}"
        exit 1
    fi
    
    # Create virtual environment
    sudo -u $SERVICE_USER python${PYTHON_VERSION} -m venv $INSTALL_DIR/venv
    
    # Install Python dependencies
    sudo -u $SERVICE_USER $INSTALL_DIR/venv/bin/pip install --upgrade pip
    sudo -u $SERVICE_USER $INSTALL_DIR/venv/bin/pip install -r $INSTALL_DIR/requirements.txt
    
    echo -e "${GREEN}✓ Application installed${NC}"
}

setup_database() {
    echo -e "${YELLOW}Setting up database...${NC}"
    
    # Create database user and database
    sudo -u postgres psql <<EOF
CREATE USER trader WITH PASSWORD 'changeme';
CREATE DATABASE trading OWNER trader;
GRANT ALL PRIVILEGES ON DATABASE trading TO trader;
EOF
    
    echo -e "${GREEN}✓ Database configured${NC}"
}

configure_nginx() {
    echo -e "${YELLOW}Configuring Nginx...${NC}"
    
    cat > /etc/nginx/sites-available/autobot <<EOF
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    location /grafana/ {
        proxy_pass http://localhost:3000/;
        proxy_set_header Host \$http_host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
    
    ln -sf /etc/nginx/sites-available/autobot /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
    nginx -t
    systemctl restart nginx
    
    echo -e "${GREEN}✓ Nginx configured${NC}"
}

setup_systemd() {
    echo -e "${YELLOW}Setting up systemd service...${NC}"
    
    # Copy service file
    cp autobot.service /etc/systemd/system/
    
    # Create environment file
    cat > $INSTALL_DIR/.env <<EOF
# Trading Configuration
TRADING_MODE=paper

# API Keys (add your own)
BINANCE_API_KEY=
BINANCE_API_SECRET=

# Telegram Notifications
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Database
POSTGRES_USER=trader
POSTGRES_PASSWORD=changeme
POSTGRES_DB=trading

# Redis
REDIS_PASSWORD=changeme

# Monitoring
GRAFANA_USER=admin
GRAFANA_PASSWORD=changeme
EOF
    
    chown $SERVICE_USER:$SERVICE_USER $INSTALL_DIR/.env
    chmod 600 $INSTALL_DIR/.env
    
    # Reload systemd
    systemctl daemon-reload
    systemctl enable autobot.service
    
    echo -e "${GREEN}✓ Systemd service configured${NC}"
}

setup_monitoring() {
    echo -e "${YELLOW}Setting up monitoring...${NC}"
    
    # Create Prometheus configuration
    mkdir -p $INSTALL_DIR/monitoring
    
    cat > $INSTALL_DIR/monitoring/prometheus.yml <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'autobot'
    static_configs:
      - targets: ['localhost:8080']
EOF
    
    # Start monitoring stack with Docker Compose
    cd $INSTALL_DIR
    docker-compose up -d prometheus grafana
    
    echo -e "${GREEN}✓ Monitoring configured${NC}"
}

setup_ssl() {
    echo -e "${YELLOW}Setting up SSL (optional)...${NC}"
    
    read -p "Do you want to setup SSL with Let's Encrypt? (y/N) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your domain name: " domain
        
        if [ ! -z "$domain" ]; then
            certbot --nginx -d $domain --non-interactive --agree-tos --email admin@$domain
            echo -e "${GREEN}✓ SSL configured for $domain${NC}"
        fi
    fi
}

setup_firewall() {
    echo -e "${YELLOW}Setting up firewall...${NC}"
    
    # Install UFW if not present
    apt-get install -y ufw
    
    # Configure firewall rules
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow 22/tcp    # SSH
    ufw allow 80/tcp    # HTTP
    ufw allow 443/tcp   # HTTPS
    ufw allow 8080/tcp  # Monitoring dashboard
    ufw allow 3000/tcp  # Grafana
    
    # Enable firewall
    echo "y" | ufw enable
    
    echo -e "${GREEN}✓ Firewall configured${NC}"
}

create_backup_script() {
    echo -e "${YELLOW}Creating backup script...${NC}"
    
    cat > $INSTALL_DIR/backup.sh <<'EOF'
#!/bin/bash
# Daily backup script

BACKUP_DIR="/opt/autonomous-trader/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$DATE.tar.gz"

# Create backup
tar -czf $BACKUP_FILE \
    /opt/autonomous-trader/data \
    /opt/autonomous-trader/logs \
    /opt/autonomous-trader/config \
    --exclude='*.log'

# Backup database
pg_dump -U trader trading | gzip > "$BACKUP_DIR/db_backup_$DATE.sql.gz"

# Keep only last 30 days of backups
find $BACKUP_DIR -type f -name "backup_*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -type f -name "db_backup_*.sql.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_FILE"
EOF
    
    chmod +x $INSTALL_DIR/backup.sh
    chown $SERVICE_USER:$SERVICE_USER $INSTALL_DIR/backup.sh
    
    # Add to crontab
    echo "0 2 * * * $SERVICE_USER $INSTALL_DIR/backup.sh >> $INSTALL_DIR/logs/backup.log 2>&1" > /etc/cron.d/autobot-backup
    
    echo -e "${GREEN}✓ Backup script created${NC}"
}

post_install_message() {
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║              INSTALLATION COMPLETED SUCCESSFULLY               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    echo "Next steps:"
    echo "1. Edit configuration: sudo nano $INSTALL_DIR/.env"
    echo "2. Add your API keys and credentials"
    echo "3. Start the service: sudo systemctl start autobot"
    echo "4. Check status: sudo systemctl status autobot"
    echo "5. View logs: sudo journalctl -u autobot -f"
    echo ""
    echo "Web interfaces:"
    echo "- Trading Dashboard: http://localhost:8080"
    echo "- Grafana: http://localhost:3000 (admin/changeme)"
    echo ""
    echo -e "${YELLOW}IMPORTANT: Change all default passwords before running in production!${NC}"
}

# Main installation flow
main() {
    print_banner
    check_root
    check_os
    
    echo -e "${YELLOW}This will install the Autonomous Trading System.${NC}"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    
    install_dependencies
    create_user
    setup_directories
    install_application
    setup_database
    configure_nginx
    setup_systemd
    setup_monitoring
    setup_ssl
    setup_firewall
    create_backup_script
    
    post_install_message
}

# Run main function
main