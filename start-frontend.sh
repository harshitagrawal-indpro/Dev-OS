#!/bin/bash

echo "⚛️ Starting AI DevLab OS Frontend..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

cd frontend

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}Installing dependencies...${NC}"
    npm install
fi

echo -e "${GREEN}✓ Frontend setup complete!${NC}"
echo -e "${BLUE}Starting development server...${NC}"

# Start the Vite development server
npm run dev