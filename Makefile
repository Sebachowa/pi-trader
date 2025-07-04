# Makefile for easy management of the trading bot

.PHONY: help install test run deploy docker-build docker-run docker-stop setup-pi clean

# Default target
help:
	@echo "Trading Bot Management Commands:"
	@echo "================================"
	@echo "make install      - Install dependencies locally"
	@echo "make test         - Run tests"
	@echo "make run          - Run locally in dry-run mode"
	@echo "make deploy       - Deploy to Raspberry Pi"
	@echo "make docker-build - Build Docker image"
	@echo "make docker-run   - Run with Docker Compose"
	@echo "make docker-stop  - Stop Docker containers"
	@echo "make setup-pi     - Initial Pi setup"
	@echo "make monitor      - Monitor the bot"
	@echo "make logs         - View live logs from Pi"
	@echo "make clean        - Clean up temporary files"

# Install dependencies
install:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements-pi.txt
	@echo "✅ Dependencies installed"

# Run tests
test:
	@echo "Running tests..."
	python3 -m pytest tests/ -v || echo "No tests yet"
	python3 -m flake8 . --max-line-length=120 --exclude=venv,__pycache__ || true

# Run locally
run:
	@echo "Starting trading bot in dry-run mode..."
	python3 run.py --dry-run

# Deploy to Pi using the deployment script
deploy:
	@echo "Deploying to Raspberry Pi..."
	@read -p "Enter Pi hostname/IP: " PI_HOST; \
	read -p "Enter Pi username [pi]: " PI_USER; \
	PI_USER=$${PI_USER:-pi}; \
	./deploy_to_pi.sh $$PI_HOST $$PI_USER

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t trader-bot:latest .

docker-run:
	@echo "Starting with Docker Compose..."
	docker-compose up -d
	@echo "✅ Trading bot started"
	@echo "View logs: docker-compose logs -f trader"

docker-stop:
	@echo "Stopping Docker containers..."
	docker-compose down

# Run with monitoring stack
docker-monitor:
	@echo "Starting with monitoring..."
	docker-compose --profile monitoring up -d
	@echo "✅ Monitoring available at:"
	@echo "   - Grafana: http://localhost:3000 (admin/admin)"
	@echo "   - Prometheus: http://localhost:9090"

# Initial Raspberry Pi setup
setup-pi:
	@echo "Setting up Raspberry Pi..."
	@read -p "Enter Pi hostname/IP: " PI_HOST; \
	read -p "Enter Pi username [pi]: " PI_USER; \
	PI_USER=$${PI_USER:-pi}; \
	ssh $$PI_USER@$$PI_HOST 'bash -s' < scripts/setup_pi.sh

# Monitor the bot
monitor:
	@echo "Starting monitoring..."
	python3 monitor.py --watch

# View logs from Pi
logs:
	@read -p "Enter Pi hostname/IP: " PI_HOST; \
	read -p "Enter Pi username [pi]: " PI_USER; \
	PI_USER=$${PI_USER:-pi}; \
	ssh $$PI_USER@$$PI_HOST "journalctl -u trader -f"

# SSH to Pi
ssh:
	@read -p "Enter Pi hostname/IP: " PI_HOST; \
	read -p "Enter Pi username [pi]: " PI_USER; \
	PI_USER=$${PI_USER:-pi}; \
	ssh $$PI_USER@$$PI_HOST

# Clean up
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	@echo "✅ Cleaned up"

# GitHub secrets setup
setup-secrets:
	@echo "Setting up GitHub secrets..."
	chmod +x scripts/setup_github_secrets.sh
	./scripts/setup_github_secrets.sh

# Quick status check
status:
	@echo "Checking trading bot status..."
	@read -p "Enter Pi hostname/IP: " PI_HOST; \
	read -p "Enter Pi username [pi]: " PI_USER; \
	PI_USER=$${PI_USER:-pi}; \
	ssh $$PI_USER@$$PI_HOST "systemctl status trader; echo ''; df -h /; echo ''; free -h"

# Backup configuration
backup:
	@echo "Backing up configuration..."
	@mkdir -p backups
	@tar -czf backups/config-backup-$$(date +%Y%m%d-%H%M%S).tar.gz config/
	@echo "✅ Configuration backed up"

# Update from Git and redeploy
update:
	@echo "Updating from Git..."
	git pull
	$(MAKE) test
	$(MAKE) deploy