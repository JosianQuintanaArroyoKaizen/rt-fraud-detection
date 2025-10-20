#!/bin/bash

# Generate a short unique suffix for resource naming
# Uses first 6 chars of hash of current timestamp + username

TIMESTAMP=$(date +%s)
USERNAME=$(whoami)
SUFFIX=$(echo -n "${TIMESTAMP}${USERNAME}" | sha256sum | cut -c1-6)

echo $SUFFIX