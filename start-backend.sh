#!/bin/bash

echo "🐍 Starting AI DevLab OS Backend..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies if they don't exist
if [ ! -f "venv/installed" ]; then
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
    touch venv/installed
fi

# Create storage directories
mkdir -p storage/models storage/datasets storage/temp

echo -e "${GREEN}✓ Backend setup complete!${NC}"
echo -e "${BLUE}Starting server...${NC}"

# Start the FastAPI server
python main.py