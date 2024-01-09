#!/bin/bash

# The URL of the Git repository
REPO_URL="https://github.com/akshay-antony/MOFGPT.git"

# Clone the repository
git clone $REPO_URL

# Extract the repository name from the URL
REPO_NAME=$(basename $REPO_URL .git)

# Change to the repository directory
cd $REPO_NAME

# Check if requirements.txt exists and then install dependencies
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found."
fi
