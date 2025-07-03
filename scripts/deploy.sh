#!/bin/bash

# Deployment script for Raspberry Pi
# This script handles deployment, rollback, and service management

set -euo pipefail

# Configuration
SERVICE_NAME="${SERVICE_NAME:-trader}"
DEPLOY_PATH="${DEPLOY_PATH:-/home/pi/trader}"
BACKUP_PATH="${BACKUP_PATH:-/home/pi/backups}"
LOG_PATH="${LOG_PATH:-/var/log/$SERVICE_NAME}"
ROLLBACK_KEEP="${ROLLBACK_KEEP:-5}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Check if running as appropriate user
check_user() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root"
        exit 1
    fi
}

# Create necessary directories
setup_directories() {
    log "Setting up directories..."
    mkdir -p "$BACKUP_PATH"
    sudo mkdir -p "$LOG_PATH"
    sudo chown $USER:$USER "$LOG_PATH"
}

# Backup current deployment
backup_current() {
    if [ -d "$DEPLOY_PATH" ] && [ "$(ls -A $DEPLOY_PATH)" ]; then
        log "Creating backup of current deployment..."
        local backup_name="${SERVICE_NAME}-$(date +%Y%m%d-%H%M%S).tar.gz"
        tar -czf "$BACKUP_PATH/$backup_name" -C "$DEPLOY_PATH" .
        log "Backup created: $backup_name"
        
        # Clean old backups
        log "Cleaning old backups (keeping last $ROLLBACK_KEEP)..."
        cd "$BACKUP_PATH"
        ls -t ${SERVICE_NAME}-*.tar.gz 2>/dev/null | tail -n +$((ROLLBACK_KEEP + 1)) | xargs -r rm
    else
        warning "No existing deployment to backup"
    fi
}

# Deploy new version
deploy() {
    local source_path="${1:-}"
    
    if [ -z "$source_path" ]; then
        error "Source path not provided"
        exit 1
    fi
    
    if [ ! -d "$source_path" ]; then
        error "Source path does not exist: $source_path"
        exit 1
    fi
    
    log "Deploying from $source_path to $DEPLOY_PATH..."
    
    # Stop service
    if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
        log "Stopping $SERVICE_NAME service..."
        sudo systemctl stop "$SERVICE_NAME"
    fi
    
    # Deploy files
    mkdir -p "$DEPLOY_PATH"
    rsync -av --delete --exclude='.env' --exclude='node_modules' --exclude='logs' "$source_path/" "$DEPLOY_PATH/"
    
    # Install dependencies
    log "Installing production dependencies..."
    cd "$DEPLOY_PATH"
    npm ci --production
    
    # Set up environment
    if [ -f "$DEPLOY_PATH/.env.production" ] && [ ! -f "$DEPLOY_PATH/.env" ]; then
        log "Setting up environment configuration..."
        cp "$DEPLOY_PATH/.env.production" "$DEPLOY_PATH/.env"
    fi
    
    # Set permissions
    chmod +x "$DEPLOY_PATH/scripts/"*.sh 2>/dev/null || true
    
    # Run migrations if available
    if [ -f "$DEPLOY_PATH/scripts/migrate.sh" ]; then
        log "Running migrations..."
        "$DEPLOY_PATH/scripts/migrate.sh"
    fi
    
    # Start service
    log "Starting $SERVICE_NAME service..."
    sudo systemctl start "$SERVICE_NAME"
    
    # Wait for service to stabilize
    sleep 5
    
    # Health check
    if [ -f "$DEPLOY_PATH/scripts/health_check.sh" ]; then
        log "Running health check..."
        if "$DEPLOY_PATH/scripts/health_check.sh"; then
            log "Deployment successful!"
        else
            error "Health check failed! Rolling back..."
            rollback
            exit 1
        fi
    else
        warning "No health check script found"
    fi
}

# Rollback to previous version
rollback() {
    log "Starting rollback..."
    
    # Find latest backup
    local latest_backup=$(ls -t "$BACKUP_PATH/${SERVICE_NAME}-"*.tar.gz 2>/dev/null | head -n1)
    
    if [ -z "$latest_backup" ]; then
        error "No backup found for rollback"
        exit 1
    fi
    
    log "Rolling back to: $(basename $latest_backup)"
    
    # Stop service
    if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
        sudo systemctl stop "$SERVICE_NAME"
    fi
    
    # Restore backup
    rm -rf "$DEPLOY_PATH"/*
    tar -xzf "$latest_backup" -C "$DEPLOY_PATH"
    
    # Reinstall dependencies
    cd "$DEPLOY_PATH"
    npm ci --production
    
    # Start service
    sudo systemctl start "$SERVICE_NAME"
    
    log "Rollback completed"
}

# List available backups
list_backups() {
    log "Available backups:"
    ls -lht "$BACKUP_PATH/${SERVICE_NAME}-"*.tar.gz 2>/dev/null || echo "No backups found"
}

# Service management
service_status() {
    sudo systemctl status "$SERVICE_NAME"
}

service_logs() {
    local lines="${1:-50}"
    sudo journalctl -u "$SERVICE_NAME" -n "$lines" -f
}

service_restart() {
    log "Restarting $SERVICE_NAME service..."
    sudo systemctl restart "$SERVICE_NAME"
}

# Main script logic
main() {
    local command="${1:-help}"
    
    case "$command" in
        deploy)
            check_user
            setup_directories
            backup_current
            deploy "${2:-}"
            ;;
        rollback)
            check_user
            rollback
            ;;
        backup)
            check_user
            setup_directories
            backup_current
            ;;
        list-backups)
            list_backups
            ;;
        status)
            service_status
            ;;
        logs)
            service_logs "${2:-50}"
            ;;
        restart)
            service_restart
            ;;
        help|*)
            echo "Usage: $0 {deploy|rollback|backup|list-backups|status|logs|restart|help}"
            echo ""
            echo "Commands:"
            echo "  deploy <source>  - Deploy from source directory"
            echo "  rollback        - Rollback to previous version"
            echo "  backup          - Create backup of current deployment"
            echo "  list-backups    - List available backups"
            echo "  status          - Show service status"
            echo "  logs [lines]    - Show service logs (default: 50 lines)"
            echo "  restart         - Restart the service"
            echo "  help            - Show this help message"
            ;;
    esac
}

# Run main function
main "$@"