#!/bin/bash

GREEN='\033[0;32m'
NC='\033[0m' # No Color
echo -e "${GREEN}Setting up the environment...${NC}"

# Move to the OpenPI directory
cd ~/VLABench/third_party/openpi

# Create virtual environment
uv venv --python 3.10 examples/vlabench/.venv

# Activate the environment
source examples/vlabench/.venv/bin/activate


# Install dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
uv pip sync examples/vlabench/requirements.txt

# Install OpenPI-Client
echo -e "${GREEN}Installing openpi-client...${NC}"
uv pip install -e packages/openpi-client


echo -e "${GREEN}Installing VLABench...${NC}"
uv pip install -e ~/VLABench

# echo -e "${GREEN}Installing OpenPI...${NC}"
# uv pip install -e .
