#!/bin/bash
# Deploy SRE Analytics to Staging Environment

set -e  # Exit on error

echo "🚀 Starting deployment to STAGING environment..."
echo ""

# Configuration
NAMESPACE="${NAMESPACE:-sre-analytics-staging}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"
DOCKER_IMAGE="${DOCKER_USERNAME}/sre-analytics:${IMAGE_TAG}"

# Deployment method (kubernetes, docker-compose, or helm)
DEPLOY_METHOD="${DEPLOY_METHOD:-kubernetes}"

echo "📋 Deployment Configuration:"
echo "  Environment: STAGING"
echo "  Namespace: ${NAMESPACE}"
echo "  Image: ${DOCKER_IMAGE}"
echo "  Method: ${DEPLOY_METHOD}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Deploy using Kubernetes
deploy_kubernetes() {
    echo "☸️  Deploying to Kubernetes..."

    if ! command_exists kubectl; then
        echo "❌ kubectl not found. Please install kubectl."
        exit 1
    fi

    # Create namespace if it doesn't exist
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

    # Apply Kubernetes manifests
    if [ -d "k8s" ]; then
        echo "📦 Applying Kubernetes manifests..."
        kubectl apply -f k8s/ -n "${NAMESPACE}"

        # Update image tag
        kubectl set image deployment/sre-analytics \
            sre-analytics="${DOCKER_IMAGE}" \
            -n "${NAMESPACE}"

        # Wait for rollout
        echo "⏳ Waiting for deployment to complete..."
        kubectl rollout status deployment/sre-analytics -n "${NAMESPACE}" --timeout=5m

        echo "✅ Kubernetes deployment complete"
    else
        echo "⚠️  No k8s/ directory found"
        exit 1
    fi
}

# Deploy using Helm
deploy_helm() {
    echo "⛵ Deploying with Helm..."

    if ! command_exists helm; then
        echo "❌ helm not found. Please install Helm."
        exit 1
    fi

    if [ -d "helm/sre-analytics" ]; then
        helm upgrade --install sre-analytics ./helm/sre-analytics \
            --namespace "${NAMESPACE}" \
            --create-namespace \
            --set image.repository="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/sre-analytics" \
            --set image.tag="${IMAGE_TAG}" \
            --set environment="staging" \
            --wait \
            --timeout=5m

        echo "✅ Helm deployment complete"
    else
        echo "⚠️  No helm/sre-analytics/ directory found"
        exit 1
    fi
}

# Deploy using Docker Compose
deploy_docker_compose() {
    echo "🐳 Deploying with Docker Compose..."

    if ! command_exists docker; then
        echo "❌ docker not found. Please install Docker."
        exit 1
    fi

    if [ -f "docker/docker-compose.staging.yml" ]; then
        docker compose -f docker/docker-compose.staging.yml pull
        docker compose -f docker/docker-compose.staging.yml up -d --remove-orphans

        echo "✅ Docker Compose deployment complete"
    else
        echo "⚠️  No docker/docker-compose.staging.yml found"
        exit 1
    fi
}

# Main deployment logic
case "${DEPLOY_METHOD}" in
    kubernetes)
        deploy_kubernetes
        ;;
    helm)
        deploy_helm
        ;;
    docker-compose)
        deploy_docker_compose
        ;;
    *)
        echo "❌ Unknown deployment method: ${DEPLOY_METHOD}"
        echo "   Supported: kubernetes, helm, docker-compose"
        exit 1
        ;;
esac

echo ""
echo "🎉 Staging deployment completed successfully!"
echo ""

# Run smoke tests
if [ -f "scripts/deployment/smoke-tests.sh" ]; then
    echo "🧪 Running smoke tests..."
    bash scripts/deployment/smoke-tests.sh staging
else
    echo "⚠️  No smoke tests found - skipping"
fi

echo ""
echo "📊 Deployment Summary:"
echo "  ✅ Environment: STAGING"
echo "  ✅ Image: ${DOCKER_IMAGE}"
echo "  ✅ Method: ${DEPLOY_METHOD}"
echo ""
echo "🔗 Access staging at: https://staging.sre-analytics.example.com"
echo "   (Update URL in .github/workflows/deploy.yml environment configuration)"
