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
    
    if ! command -v kubectl &> /dev/null; then
        log_warning "kubectl is not installed. Kubernetes deployment will be skipped."
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
        "container.googleapis.com"
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
    docker build -t ${IMAGE_NAME} .
    
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
        --set-env-vars GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${LOCATION} \
        --set-cloudsql-instances ${PROJECT_ID}:${LOCATION}:your-instance-name
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')
    
    log_success "Service deployed to Cloud Run"
    log_info "Service URL: ${SERVICE_URL}"
}

# Deploy to Google Kubernetes Engine
deploy_to_gke() {
    log_info "Deploying to Google Kubernetes Engine..."
    
    # Check if cluster exists
    if ! gcloud container clusters describe shellhacks-cluster --region ${REGION} &> /dev/null; then
        log_info "Creating GKE cluster..."
        gcloud container clusters create shellhacks-cluster \
            --region ${REGION} \
            --num-nodes 3 \
            --machine-type e2-standard-4 \
            --enable-autoscaling \
            --min-nodes 1 \
            --max-nodes 10 \
            --enable-autorepair \
            --enable-autoupgrade \
            --enable-network-policy
    fi
    
    # Get cluster credentials
    gcloud container clusters get-credentials shellhacks-cluster --region ${REGION}
    
    # Apply Kubernetes manifests
    log_info "Applying Kubernetes manifests..."
    kubectl apply -f k8s/
    
    # Wait for deployment
    kubectl rollout status deployment/${SERVICE_NAME}
    
    log_success "Service deployed to GKE"
}

# Deploy to Vertex AI Agent Engine
deploy_to_vertex_ai() {
    log_info "Deploying to Vertex AI Agent Engine..."
    
    # Create agent configuration
    cat > agent_config.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
data:
  adk_config.yaml: |
$(cat config/adk_config.yaml | sed 's/^/    /')
  vertex_config.yaml: |
$(cat config/vertex_config.yaml | sed 's/^/    /')
EOF
    
    # Apply configuration
    kubectl apply -f agent_config.yaml
    
    # Deploy agent to Vertex AI
    gcloud ai models upload \
        --region=${LOCATION} \
        --display-name=${SERVICE_NAME} \
        --container-image-uri=${IMAGE_NAME} \
        --container-ports=8080 \
        --container-health-route=/health \
        --container-predict-route=/predict
    
    log_success "Agent deployed to Vertex AI"
}

# Set up monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create monitoring dashboard
    gcloud monitoring dashboards create \
        --config-from-file=monitoring/dashboard.json
    
    # Create alerting policies
    gcloud alpha monitoring policies create \
        --policy-from-file=monitoring/alert-policies.yaml
    
    log_success "Monitoring setup completed"
}

# Set up logging
setup_logging() {
    log_info "Setting up logging..."
    
    # Create log sink
    gcloud logging sinks create shellhacks-agents-sink \
        bigquery.googleapis.com/projects/${PROJECT_ID}/datasets/logs \
        --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="'${SERVICE_NAME}'"'
    
    log_success "Logging setup completed"
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
            setup_logging
            ;;
        "gke")
            check_dependencies
            authenticate_gcloud
            enable_apis
            build_and_push_image
            deploy_to_gke
            ;;
        "vertex-ai")
            check_dependencies
            authenticate_gcloud
            enable_apis
            build_and_push_image
            deploy_to_vertex_ai
            ;;
        "all")
            check_dependencies
            authenticate_gcloud
            enable_apis
            build_and_push_image
            deploy_to_cloud_run
            deploy_to_gke
            deploy_to_vertex_ai
            setup_monitoring
            setup_logging
            ;;
        *)
            log_error "Invalid deployment target: ${DEPLOYMENT_TARGET}"
            log_info "Valid targets: cloud-run, gke, vertex-ai, all"
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
