#!/bin/bash
# Smoke Tests for SRE Analytics Deployment

set -e

ENVIRONMENT="${1:-staging}"
MAX_RETRIES=30
RETRY_DELAY=5

echo "🧪 Running smoke tests for ${ENVIRONMENT} environment..."
echo ""

# Determine base URL based on environment
case "${ENVIRONMENT}" in
    staging)
        BASE_URL="${STAGING_URL:-http://localhost:5001}"
        ;;
    production)
        BASE_URL="${PRODUCTION_URL:-https://sre-analytics.example.com}"
        ;;
    *)
        echo "❌ Unknown environment: ${ENVIRONMENT}"
        exit 1
        ;;
esac

echo "🔗 Testing endpoint: ${BASE_URL}"
echo ""

# Test 1: Health Check
test_health() {
    echo "1️⃣  Testing /health endpoint..."

    for i in $(seq 1 $MAX_RETRIES); do
        if curl -f -s "${BASE_URL}/health" > /dev/null 2>&1; then
            HEALTH_RESPONSE=$(curl -s "${BASE_URL}/health")
            STATUS=$(echo "${HEALTH_RESPONSE}" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

            if [ "${STATUS}" = "healthy" ]; then
                echo "   ✅ Health check PASSED"
                echo "   Response: ${HEALTH_RESPONSE}"
                return 0
            fi
        fi

        echo "   ⏳ Attempt $i/$MAX_RETRIES - waiting for service..."
        sleep $RETRY_DELAY
    done

    echo "   ❌ Health check FAILED after $MAX_RETRIES attempts"
    return 1
}

# Test 2: Readiness Check
test_readiness() {
    echo ""
    echo "2️⃣  Testing /health/ready endpoint..."

    RESPONSE=$(curl -s -w "\n%{http_code}" "${BASE_URL}/health/ready")
    HTTP_CODE=$(echo "${RESPONSE}" | tail -n1)
    BODY=$(echo "${RESPONSE}" | sed '$d')

    if [ "${HTTP_CODE}" = "200" ]; then
        echo "   ✅ Readiness check PASSED"
        echo "   Response: ${BODY}"
        return 0
    else
        echo "   ❌ Readiness check FAILED (HTTP ${HTTP_CODE})"
        echo "   Response: ${BODY}"
        return 1
    fi
}

# Test 3: Liveness Check
test_liveness() {
    echo ""
    echo "3️⃣  Testing /health/live endpoint..."

    RESPONSE=$(curl -s -w "\n%{http_code}" "${BASE_URL}/health/live")
    HTTP_CODE=$(echo "${RESPONSE}" | tail -n1)
    BODY=$(echo "${RESPONSE}" | sed '$d')

    if [ "${HTTP_CODE}" = "200" ]; then
        echo "   ✅ Liveness check PASSED"
        echo "   Response: ${BODY}"
        return 0
    else
        echo "   ❌ Liveness check FAILED (HTTP ${HTTP_CODE})"
        echo "   Response: ${BODY}"
        return 1
    fi
}

# Test 4: Home Page
test_homepage() {
    echo ""
    echo "4️⃣  Testing homepage (/)..."

    RESPONSE=$(curl -s -w "\n%{http_code}" "${BASE_URL}/")
    HTTP_CODE=$(echo "${RESPONSE}" | tail -n1)

    if [ "${HTTP_CODE}" = "200" ]; then
        echo "   ✅ Homepage PASSED"
        return 0
    else
        echo "   ❌ Homepage FAILED (HTTP ${HTTP_CODE})"
        return 1
    fi
}

# Test 5: API Response Time
test_response_time() {
    echo ""
    echo "5️⃣  Testing API response time..."

    START_TIME=$(date +%s%N)
    curl -s "${BASE_URL}/health" > /dev/null 2>&1
    END_TIME=$(date +%s%N)

    RESPONSE_TIME=$(( (END_TIME - START_TIME) / 1000000 ))  # Convert to milliseconds

    if [ ${RESPONSE_TIME} -lt 1000 ]; then
        echo "   ✅ Response time: ${RESPONSE_TIME}ms (< 1000ms threshold)"
        return 0
    else
        echo "   ⚠️  Response time: ${RESPONSE_TIME}ms (slower than expected)"
        return 0  # Not a failure, just a warning
    fi
}

# Run all tests
FAILED_TESTS=0

test_health || FAILED_TESTS=$((FAILED_TESTS + 1))
test_readiness || FAILED_TESTS=$((FAILED_TESTS + 1))
test_liveness || FAILED_TESTS=$((FAILED_TESTS + 1))
test_homepage || FAILED_TESTS=$((FAILED_TESTS + 1))
test_response_time || FAILED_TESTS=$((FAILED_TESTS + 1))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ${FAILED_TESTS} -eq 0 ]; then
    echo "✅ All smoke tests PASSED!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 0
else
    echo "❌ ${FAILED_TESTS} smoke test(s) FAILED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 1
fi
