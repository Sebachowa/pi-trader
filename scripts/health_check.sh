#!/bin/bash

# Health check script for trader service
# Returns 0 if healthy, 1 if unhealthy

set -euo pipefail

# Configuration
SERVICE_NAME="${SERVICE_NAME:-trader}"
SERVICE_PORT="${SERVICE_PORT:-3000}"
HEALTH_ENDPOINT="${HEALTH_ENDPOINT:-/health}"
MAX_RETRIES="${MAX_RETRIES:-5}"
RETRY_DELAY="${RETRY_DELAY:-3}"
TIMEOUT="${TIMEOUT:-10}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[HEALTH CHECK]${NC} $1"
}

error() {
    echo -e "${RED}[HEALTH CHECK] ERROR:${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[HEALTH CHECK] WARNING:${NC} $1"
}

# Check if service is running
check_service_status() {
    log "Checking if $SERVICE_NAME service is active..."
    
    if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
        log "Service is active"
        return 0
    else
        error "Service is not active"
        sudo systemctl status "$SERVICE_NAME" --no-pager || true
        return 1
    fi
}

# Check if port is listening
check_port() {
    log "Checking if port $SERVICE_PORT is listening..."
    
    if ss -tuln | grep -q ":$SERVICE_PORT "; then
        log "Port $SERVICE_PORT is listening"
        return 0
    else
        error "Port $SERVICE_PORT is not listening"
        return 1
    fi
}

# Check HTTP endpoint
check_http_endpoint() {
    local url="http://localhost:$SERVICE_PORT$HEALTH_ENDPOINT"
    log "Checking HTTP endpoint: $url"
    
    local response
    local http_code
    
    # Make HTTP request
    response=$(curl -s -w "\n%{http_code}" --connect-timeout "$TIMEOUT" "$url" 2>/dev/null || echo "000")
    http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "200" ]; then
        log "HTTP endpoint returned 200 OK"
        if [ -n "$body" ]; then
            log "Response: $body"
        fi
        return 0
    elif [ "$http_code" = "000" ]; then
        error "Failed to connect to HTTP endpoint"
        return 1
    else
        error "HTTP endpoint returned $http_code"
        if [ -n "$body" ]; then
            error "Response: $body"
        fi
        return 1
    fi
}

# Check process resources
check_resources() {
    log "Checking process resources..."
    
    local pid=$(sudo systemctl show -p MainPID --value "$SERVICE_NAME")
    
    if [ "$pid" = "0" ] || [ -z "$pid" ]; then
        error "Could not find process PID"
        return 1
    fi
    
    # Check CPU usage
    local cpu_usage=$(ps -p "$pid" -o %cpu --no-headers | tr -d ' ')
    log "CPU usage: ${cpu_usage}%"
    
    # Check memory usage
    local mem_usage=$(ps -p "$pid" -o %mem --no-headers | tr -d ' ')
    log "Memory usage: ${mem_usage}%"
    
    # Check if CPU usage is too high
    if (( $(echo "$cpu_usage > 90" | bc -l) )); then
        warning "CPU usage is very high: ${cpu_usage}%"
    fi
    
    # Check if memory usage is too high
    if (( $(echo "$mem_usage > 80" | bc -l) )); then
        warning "Memory usage is very high: ${mem_usage}%"
    fi
    
    return 0
}

# Check disk space
check_disk_space() {
    log "Checking disk space..."
    
    local usage=$(df -h /home/pi | awk 'NR==2 {print $5}' | sed 's/%//')
    log "Disk usage: ${usage}%"
    
    if [ "$usage" -gt 90 ]; then
        error "Disk space is critically low: ${usage}%"
        return 1
    elif [ "$usage" -gt 80 ]; then
        warning "Disk space is getting low: ${usage}%"
    fi
    
    return 0
}

# Check application logs for errors
check_logs() {
    log "Checking recent logs for errors..."
    
    local error_count=$(sudo journalctl -u "$SERVICE_NAME" --since "5 minutes ago" --no-pager | grep -c -i "error" || true)
    
    if [ "$error_count" -gt 10 ]; then
        error "Found $error_count errors in recent logs"
        warning "Recent errors:"
        sudo journalctl -u "$SERVICE_NAME" --since "5 minutes ago" --no-pager | grep -i "error" | tail -5
        return 1
    elif [ "$error_count" -gt 0 ]; then
        warning "Found $error_count errors in recent logs"
    else
        log "No errors found in recent logs"
    fi
    
    return 0
}

# Main health check with retries
run_health_check() {
    local checks_passed=0
    local total_checks=6
    
    # Run all checks
    if check_service_status; then
        ((checks_passed++))
    fi
    
    if check_port; then
        ((checks_passed++))
    fi
    
    # HTTP endpoint check with retries
    local http_healthy=false
    for i in $(seq 1 "$MAX_RETRIES"); do
        if check_http_endpoint; then
            http_healthy=true
            ((checks_passed++))
            break
        else
            if [ "$i" -lt "$MAX_RETRIES" ]; then
                warning "HTTP check failed, retrying in $RETRY_DELAY seconds... ($i/$MAX_RETRIES)"
                sleep "$RETRY_DELAY"
            fi
        fi
    done
    
    if [ "$http_healthy" = false ]; then
        error "HTTP endpoint check failed after $MAX_RETRIES attempts"
    fi
    
    if check_resources; then
        ((checks_passed++))
    fi
    
    if check_disk_space; then
        ((checks_passed++))
    fi
    
    if check_logs; then
        ((checks_passed++))
    fi
    
    # Summary
    echo ""
    if [ "$checks_passed" -eq "$total_checks" ]; then
        log "Health check PASSED ($checks_passed/$total_checks checks)"
        return 0
    else
        error "Health check FAILED ($checks_passed/$total_checks checks)"
        return 1
    fi
}

# Performance metrics
get_metrics() {
    log "Collecting performance metrics..."
    
    local pid=$(sudo systemctl show -p MainPID --value "$SERVICE_NAME")
    
    if [ "$pid" != "0" ] && [ -n "$pid" ]; then
        echo "Process Metrics:"
        echo "  PID: $pid"
        echo "  CPU: $(ps -p "$pid" -o %cpu --no-headers | tr -d ' ')%"
        echo "  Memory: $(ps -p "$pid" -o %mem --no-headers | tr -d ' ')%"
        echo "  Uptime: $(ps -p "$pid" -o etime --no-headers | tr -d ' ')"
        echo ""
    fi
    
    echo "System Metrics:"
    echo "  Load Average: $(uptime | awk -F'load average:' '{print $2}')"
    echo "  Memory: $(free -h | awk 'NR==2 {print "Used: " $3 " / Total: " $2}')"
    echo "  Disk: $(df -h /home/pi | awk 'NR==2 {print "Used: " $3 " / Total: " $2 " (" $5 ")"}')"
    echo "  Temperature: $(vcgencmd measure_temp 2>/dev/null || echo "N/A")"
}

# Main script
main() {
    local command="${1:-check}"
    
    case "$command" in
        check)
            run_health_check
            ;;
        metrics)
            get_metrics
            ;;
        help|*)
            echo "Usage: $0 {check|metrics|help}"
            echo ""
            echo "Commands:"
            echo "  check    - Run health checks (default)"
            echo "  metrics  - Show performance metrics"
            echo "  help     - Show this help message"
            ;;
    esac
}

# Run main function
main "$@"