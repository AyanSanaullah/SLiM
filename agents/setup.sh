#!/bin/bash

# Setup script for Google ADK Agents
# This script sets up the environment and dependencies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    log_info "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log_info "Python version: ${PYTHON_VERSION}"
    
    log_success "Python check completed"
}

# Create virtual environment
create_venv() {
    log_info "Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    log_info "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    log_success "Dependencies installed successfully"
}

# Set up Google Cloud credentials
setup_credentials() {
    log_info "Setting up Google Cloud credentials..."
    
    if [ ! -f "credentials/service-account.json" ]; then
        log_warning "Service account credentials not found at credentials/service-account.json"
        log_info "Please download your service account key from Google Cloud Console and place it at credentials/service-account.json"
        log_info "You can also set the GOOGLE_APPLICATION_CREDENTIALS environment variable"
    else
        log_success "Service account credentials found"
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    directories=(
        "data/user_data"
        "models/user_models"
        "logs/user_logs"
        "backups/user_models"
        "exports/user_models"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    log_success "Directories created successfully"
}

# Set up environment variables
setup_env() {
    log_info "Setting up environment variables..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account.json

# Application Configuration
ADK_CONFIG_PATH=config/adk_config.yaml
VERTEX_CONFIG_PATH=config/vertex_config.yaml
LOG_LEVEL=INFO
FLASK_DEBUG=False
PORT=8080

# Redis Configuration (for local development)
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
EOF
        log_success "Environment file created"
    else
        log_info "Environment file already exists"
    fi
}

# Test installation
test_installation() {
    log_info "Testing installation..."
    
    # Test Python imports
    python3 -c "
import sys
try:
    from adk_agents.user_agent_manager import UserAgentManager
    from adk_agents.vertex_client import VertexAIClient
    import flask
    # import yaml
    print('✓ All required modules imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"
    
    log_success "Installation test completed"
}

# Main setup function
main() {
    log_info "Starting setup for Google ADK Agents..."
    
    check_python
    create_venv
    install_dependencies
    setup_credentials
    create_directories
    setup_env
    test_installation
    
    log_success "Setup completed successfully!"
    
    log_info "Next steps:"
    log_info "1. Update the .env file with your Google Cloud project details"
    log_info "2. Place your service account credentials at credentials/service-account.json"
    log_info "3. Run: source venv/bin/activate"
    log_info "4. Run: python app.py (for local development)"
    log_info "5. Run: ./deploy.sh (for deployment to Google Cloud)"
}

# Run main function
main "$@"
