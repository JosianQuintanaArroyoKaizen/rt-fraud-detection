#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
ENVIRONMENT=${1:-dev}
REGION=${AWS_REGION:-eu-central-1}
STACK_NAME="${ENVIRONMENT}-rtfd-streaming-foundation"

# Generate unique suffix
SUFFIX=$(${SCRIPT_DIR}/generate-suffix.sh)

echo -e "${GREEN}Deploying Streaming Foundation Stack${NC}"
echo "Environment: ${ENVIRONMENT}"
echo "Region: ${REGION}"
echo "Stack Name: ${STACK_NAME}"
echo "Unique Suffix: ${SUFFIX}"
echo ""

# Validate template
echo -e "${YELLOW}Validating CloudFormation template...${NC}"
aws cloudformation validate-template \
    --template-body file://${SCRIPT_DIR}/01-streaming-foundation.yaml \
    --region ${REGION}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Template validation successful${NC}"
else
    echo -e "${RED}Template validation failed${NC}"
    exit 1
fi

# Deploy stack
echo -e "${YELLOW}Deploying stack...${NC}"
aws cloudformation deploy \
    --template-file ${SCRIPT_DIR}/01-streaming-foundation.yaml \
    --stack-name ${STACK_NAME} \
    --parameter-overrides \
        Environment=${ENVIRONMENT} \
        ProjectName=rtfd \
        UniqueSuffix=${SUFFIX} \
        KinesisShardCount=3 \
        KinesisRetentionHours=24 \
    --capabilities CAPABILITY_NAMED_IAM \
    --region ${REGION} \
    --tags \
        Environment=${ENVIRONMENT} \
        Project=rtfd \
        ManagedBy=CloudFormation

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Stack deployment successful${NC}"
    
    # Get outputs
    echo -e "${YELLOW}Stack Outputs:${NC}"
    aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME} \
        --region ${REGION} \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
        --output table
    
    # Save outputs to file
    OUTPUTS_FILE="${SCRIPT_DIR}/outputs/${ENVIRONMENT}-outputs.json"
    mkdir -p ${SCRIPT_DIR}/outputs
    aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME} \
        --region ${REGION} \
        --query 'Stacks[0].Outputs' \
        > ${OUTPUTS_FILE}
    
    echo -e "${GREEN}Outputs saved to: ${OUTPUTS_FILE}${NC}"
else
    echo -e "${RED}Stack deployment failed${NC}"
    exit 1
fi