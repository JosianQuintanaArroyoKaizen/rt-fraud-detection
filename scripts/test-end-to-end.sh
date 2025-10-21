#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ENVIRONMENT=${1:-dev}
REGION=${AWS_REGION:-eu-central-1}
PRODUCER_STACK="${ENVIRONMENT}-rtfd-lambda-producer"
CONSUMER_STACK="${ENVIRONMENT}-rtfd-lambda-consumer"
FOUNDATION_STACK="${ENVIRONMENT}-rtfd-streaming-foundation"

echo -e "${BLUE}Testing End-to-End Data Flow${NC}"

PRODUCER_FUNCTION=$(aws cloudformation describe-stacks \
    --stack-name ${PRODUCER_STACK} \
    --region ${REGION} \
    --query 'Stacks[0].Outputs[?OutputKey==`ProducerLambdaName`].OutputValue' \
    --output text)

DATA_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name ${FOUNDATION_STACK} \
    --region ${REGION} \
    --query 'Stacks[0].Outputs[?OutputKey==`DataBucketName`].OutputValue' \
    --output text)

echo -e "${YELLOW}Step 1: Invoking producer to send test transactions...${NC}"
aws lambda invoke \
    --function-name ${PRODUCER_FUNCTION} \
    --region ${REGION} \
    --log-type Tail \
    response.json > /dev/null

echo -e "${GREEN}Producer invoked${NC}"
cat response.json | jq .

echo -e "${YELLOW}Step 2: Waiting 10 seconds for consumer to process...${NC}"
sleep 10

echo -e "${YELLOW}Step 3: Checking S3 for processed data...${NC}"
aws s3 ls s3://${DATA_BUCKET}/processed/ --recursive --human-readable | tail -5

echo -e "${GREEN}Latest processed file:${NC}"
LATEST_FILE=$(aws s3 ls s3://${DATA_BUCKET}/processed/ --recursive | sort | tail -1 | awk '{print $4}')

if [ -n "$LATEST_FILE" ]; then
    echo "s3://${DATA_BUCKET}/${LATEST_FILE}"
    echo ""
    echo -e "${YELLOW}File contents (first 50 lines):${NC}"
    aws s3 cp s3://${DATA_BUCKET}/${LATEST_FILE} - | jq . | head -50
else
    echo -e "${YELLOW}No processed files found yet. Consumer may still be processing.${NC}"
fi

echo ""
echo -e "${BLUE}End-to-End Test Complete${NC}"