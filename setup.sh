#!/bin/bash

echo "🚀 Setting up AI DevLab OS..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js not found. Please install Node.js 16 or higher."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    print_error "npm not found. Please install npm."
    exit 1
fi

print_status "Setting up project structure..."

# Create necessary directories
mkdir -p backend/storage/{models,datasets,temp}
mkdir -p frontend/src/components

print_success "Project directories created"

# Setup backend
print_status "Setting up backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install backend dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

print_success "Backend setup complete!"

# Mark virtual environment as installed
touch venv/installed

cd ..

# Setup frontend
print_status "Setting up frontend..."
cd frontend

# Install frontend dependencies
if [ ! -d "node_modules" ]; then
    print_status "Installing Node.js dependencies..."
    npm install
fi

print_success "Frontend setup complete!"

cd ..

print_success "🎉 AI DevLab OS setup complete!"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "1. Start the backend: ${YELLOW}./start-backend.sh${NC}"
echo -e "2. Start the frontend: ${YELLOW}./start-frontend.sh${NC}"
echo -e "3. Or start both: ${YELLOW}./start-all.sh${NC}"
echo ""
echo -e "${BLUE}URLs:${NC}"
echo -e "📊 Backend API: ${YELLOW}http://localhost:8000${NC}"
echo -e "🌐 Frontend App: ${YELLOW}http://localhost:3000${NC}"
echo -e "📚 API Docs: ${YELLOW}http://localhost:8000/docs${NC}"
echo ""

# Check for optional dependencies
print_status "Checking optional dependencies..."

cd backend
source venv/bin/activate

python3 -c "
import sys
try:
    import pandas
    print('✓ pandas: Available')
except ImportError:
    print('✗ pandas: Not available')

try:
    import sklearn
    print('✓ scikit-learn: Available')
except ImportError:
    print('✗ scikit-learn: Not available')

try:
    import xgboost
    print('✓ XGBoost: Available')
except ImportError:
    print('✗ XGBoost: Not available')

try:
    import matplotlib
    print('✓ matplotlib: Available')
except ImportError:
    print('✗ matplotlib: Not available')
"

cd ..

echo ""
print_warning "If any dependencies are missing, they will be automatically handled by the backend."
print_success "Setup completed successfully!"