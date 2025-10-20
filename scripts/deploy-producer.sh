#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ENVIRONMENT=${1:-dev}
REGION=${AWS_REGION:-eu-central-1}
STACK_NAME="${ENVIRONMENT}-rtfd-lambda-producer"
FOUNDATION_STACK="${ENVIRONMENT}-rtfd-streaming-foundation"

echo -e "${YELLOW}Deploying Producer Lambda${NC}"

# Package Lambda
./scripts/package-producer-lambda.sh

# Deploy CloudFormation stack
echo -e "${YELLOW}Deploying Lambda infrastructure...${NC}"
aws cloudformation deploy \
    --template-file cloudformation/02-lambda-producer.yaml \
    --stack-name ${STACK_NAME} \
    --parameter-overrides \
        Environment=${ENVIRONMENT} \
        FoundationStackName=${FOUNDATION_STACK} \
    --capabilities CAPABILITY_IAM \
    --region ${REGION}

# Get Lambda function name
FUNCTION_NAME=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME} \
    --region ${REGION} \
    --query 'Stacks[0].Outputs[?OutputKey==`ProducerLambdaName`].OutputValue' \
    --output text)

# Get deployment bucket from foundation stack
DEPLOYMENT_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name ${FOUNDATION_STACK} \
    --region ${REGION} \
    --query 'Stacks[0].Outputs[?OutputKey==`DataBucketName`].OutputValue' \
    --output text)

echo -e "${YELLOW}Uploading Lambda package to S3...${NC}"
S3_KEY="lambda-deployments/producer-lambda-$(date +%s).zip"
aws s3 cp build/producer-lambda.zip s3://${DEPLOYMENT_BUCKET}/${S3_KEY} \
    --region ${REGION}

echo -e "${YELLOW}Updating Lambda code from S3...${NC}"
aws lambda update-function-code \
    --function-name ${FUNCTION_NAME} \
    --s3-bucket ${DEPLOYMENT_BUCKET} \
    --s3-key ${S3_KEY} \
    --region ${REGION}

echo -e "${GREEN}Producer Lambda deployed successfully${NC}"
echo "Function name: ${FUNCTION_NAME}"