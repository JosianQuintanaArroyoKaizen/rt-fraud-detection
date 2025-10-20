#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Packaging producer Lambda...${NC}"

# Create build directory
BUILD_DIR="build/producer"
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}

# Copy Lambda code
cp src/producer/lambda_function.py ${BUILD_DIR}/

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r src/producer/requirements.txt -t ${BUILD_DIR}/ --platform manylinux2014_x86_64 --only-binary=:all:

# Create ZIP
cd ${BUILD_DIR}
zip -r ../producer-lambda.zip . -q
cd ../..

echo -e "${GREEN}Package created: build/producer-lambda.zip${NC}"
ls -lh build/producer-lambda.zip