#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Packaging consumer Lambda...${NC}"

BUILD_DIR="build/consumer"
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}

cp src/consumer/lambda_function.py ${BUILD_DIR}/

echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r src/consumer/requirements.txt -t ${BUILD_DIR}/ --platform manylinux2014_x86_64 --only-binary=:all:

cd ${BUILD_DIR}
zip -r ../consumer-lambda.zip . -q
cd ../..

echo -e "${GREEN}Package created: build/consumer-lambda.zip${NC}"
ls -lh build/consumer-lambda.zip