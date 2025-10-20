#!/bin/bash

set -e

ENVIRONMENT=${1:-dev}
REGION=${AWS_REGION:-eu-central-1}
STACK_NAME="${ENVIRONMENT}-rtfd-lambda-producer"

FUNCTION_NAME=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME} \
    --region ${REGION} \
    --query 'Stacks[0].Outputs[?OutputKey==`ProducerLambdaName`].OutputValue' \
    --output text)

echo "Invoking Lambda: ${FUNCTION_NAME}"

aws lambda invoke \
    --function-name ${FUNCTION_NAME} \
    --region ${REGION} \
    --log-type Tail \
    --query 'LogResult' \
    --output text \
    response.json | base64 -d

echo ""
echo "Response:"
cat response.json | jq .