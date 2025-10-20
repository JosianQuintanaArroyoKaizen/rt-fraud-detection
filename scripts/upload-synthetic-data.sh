#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ENVIRONMENT=${1:-dev}
OUTPUTS_FILE="cloudformation/outputs/${ENVIRONMENT}-outputs.json"

if [ ! -f "$OUTPUTS_FILE" ]; then
    echo "Error: Outputs file not found: $OUTPUTS_FILE"
    exit 1
fi

BUCKET_NAME=$(cat $OUTPUTS_FILE | jq -r '.[] | select(.OutputKey=="DataBucketName") | .OutputValue')

echo -e "${YELLOW}Uploading synthetic data to S3...${NC}"
echo "Bucket: ${BUCKET_NAME}"
echo "Source: data/synthetic_labeled/mifid_synthetic.parquet"

aws s3 cp \
    data/synthetic_labeled/mifid_synthetic.parquet \
    s3://${BUCKET_NAME}/synthetic/mifid_synthetic.parquet

echo -e "${GREEN}Upload complete${NC}"
echo "S3 URI: s3://${BUCKET_NAME}/synthetic/mifid_synthetic.parquet"