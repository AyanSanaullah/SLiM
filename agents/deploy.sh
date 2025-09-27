#!/bin/bash

# Deployment script for Google ADK Agents
# This script handles deployment to Google Cloud Platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"your-project-id"}
LOCATION=${GOOGLE_CLOUD_LOCATION:-"us-central1"}
SERVICE_NAME="shellhacks-adk-agents"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
REGION="us-central1"

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

# Check if required tools are installed
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    log_success "Dependencies check completed"
}

# Authenticate with Google Cloud
authenticate_gcloud() {
    log_info "Authenticating with Google Cloud..."
    
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_info "No active authentication found. Please authenticate:"
        gcloud auth login
    fi
    
    gcloud config set project ${PROJECT_ID}
    log_success "Google Cloud authentication completed"
}

# Enable required APIs
enable_apis() {
    log_info "Enabling required Google Cloud APIs..."
    
    apis=(
        "aiplatform.googleapis.com"
        "cloudbuild.googleapis.com"
        "run.googleapis.com"
        "storage.googleapis.com"
        "logging.googleapis.com"
        "monitoring.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        log_info "Enabling ${api}..."
        gcloud services enable ${api}
    done
    
    log_success "APIs enabled successfully"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    # Build image
    log_info "Building Docker image: ${IMAGE_NAME}"
    docker build --platform linux/amd64 -f docker/Dockerfile -t ${IMAGE_NAME} .
    
    # Configure Docker to use gcloud as credential helper
    gcloud auth configure-docker
    
    # Push image
    log_info "Pushing image to Google Container Registry..."
    docker push ${IMAGE_NAME}
    
    log_success "Docker image built and pushed successfully"
}

# Deploy to Cloud Run
deploy_to_cloud_run() {
    log_info "Deploying to Google Cloud Run..."
    
    # Deploy service
    gcloud run deploy ${SERVICE_NAME} \
        --image ${IMAGE_NAME} \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --port 8080 \
        --memory 4Gi \
        --cpu 2 \
        --max-instances 10 \
        --min-instances 1 \
        --timeout 300 \
        --concurrency 80 \
        --set-env-vars GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${LOCATION}
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')
    
    log_success "Service deployed to Cloud Run"
    log_info "Service URL: ${SERVICE_URL}"
}

# Set up monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create log sink for Cloud Run
    gcloud logging sinks create shellhacks-agents-sink \
        bigquery.googleapis.com/projects/${PROJECT_ID}/datasets/logs \
        --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="'${SERVICE_NAME}'"'
    
    log_success "Monitoring setup completed"
}

# Main deployment function
main() {
    log_info "Starting deployment of Google ADK Agents..."
    
    # Parse command line arguments
    DEPLOYMENT_TARGET=${1:-"cloud-run"}
    
    case ${DEPLOYMENT_TARGET} in
        "cloud-run")
            check_dependencies
            authenticate_gcloud
            enable_apis
            build_and_push_image
            deploy_to_cloud_run
            setup_monitoring
            ;;
        *)
            log_error "Invalid deployment target: ${DEPLOYMENT_TARGET}"
            log_info "Valid targets: cloud-run"
            exit 1
            ;;
    esac
    
    log_success "Deployment completed successfully!"
    
    # Display useful information
    log_info "Useful commands:"
    log_info "  View logs: gcloud logging read 'resource.type=\"cloud_run_revision\"' --limit 50"
    log_info "  View service: gcloud run services list"
    log_info "  Scale service: gcloud run services update ${SERVICE_NAME} --min-instances 2"
}

# Run main function
main "$@"
