# AI DevLab OS 🚀

**Complete AI Development Platform** - Zero-friction machine learning from data upload to production deployment.

![AI DevLab OS](https://img.shields.io/badge/AI-DevLab%20OS-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge)
![React](https://img.shields.io/badge/React-18+-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-red?style=for-the-badge)

## ✨ Features

### 🔥 Core Platform
- **Smart Dataset Analysis** - Upload any dataset and get instant AI-powered insights
- **Auto Model Training** - Automated ML with hyperparameter optimization  
- **One-Click API Deployment** - Production-ready APIs with monitoring
- **Colab-Style Workspace** - Interactive Python notebooks with AI suggestions
- **AI-Powered UI Builder** - Generate interfaces from natural language
- **Enterprise Security** - Built-in auth, rate limiting, and encryption

### 🛠️ Technology Stack
- **Backend**: FastAPI, Python 3.8+, scikit-learn, XGBoost, pandas
- **Frontend**: React 18, TypeScript, Tailwind CSS, Lucide Icons
- **ML Libraries**: scikit-learn, XGBoost, pandas, numpy, matplotlib
- **Development**: Vite, ESLint, PostCSS, Uvicorn

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Git** (for cloning)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ai-devlab-os
```

### 2. Run Setup Script
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Start the Platform
```bash
# Option 1: Start everything at once
./start-all.sh

# Option 2: Start separately
./start-backend.sh    # Terminal 1
./start-frontend.sh   # Terminal 2
```

### 4. Access the Platform
- 🌐 **Frontend**: http://localhost:3000
- 📊 **Backend API**: http://localhost:8000  
- 📚 **API Docs**: http://localhost:8000/docs

## 📁 Project Structure

```
ai-devlab-os/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── storage/           # Data storage
│       ├── models/        # Trained models
│       ├── datasets/      # Uploaded datasets
│       └── temp/          # Temporary files
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── App.tsx       # Main app component
│   │   ├── main.tsx      # Entry point
│   │   └── index.css     # Global styles
│   ├── package.json      # Node dependencies
│   ├── vite.config.ts    # Vite configuration
│   ├── tailwind.config.js # Tailwind CSS config
│   └── index.html        # HTML template
├── start-all.sh           # Start both services
├── start-backend.sh       # Start backend only
├── start-frontend.sh      # Start frontend only
├── setup.sh              # Initial setup script
└── README.md             # This file
```

## 🔧 Manual Setup

If the automated setup doesn't work, follow these manual steps:

### Backend Setup
```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create storage directories
mkdir -p storage/models storage/datasets storage/temp

# Start backend
python main.py
```

### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## 🐛 Troubleshooting

### Common Issues

#### 2. Node.js Dependencies Issues
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

#### 3. Port Already in Use
```bash
# Kill processes on ports 3000 and 8000
sudo lsof -ti:3000 | xargs kill -9  # Linux/Mac
sudo lsof -ti:8000 | xargs kill -9  # Linux/Mac

# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

#### 4. CORS Issues
Make sure the backend is running on port 8000 and frontend on port 3000. The backend is configured to accept requests from localhost:3000.

#### 5. TypeScript Errors
```bash
cd frontend
npx tsc --noEmit  # Check for type errors
```

### Error Messages and Solutions

| Error | Solution |
|-------|----------|
| `Cannot find module './components/...` | Create missing component files in `frontend/src/components/` |
| `pandas could not be resolved` | Install pandas: `pip install pandas` |
| `Port 3000 is already in use` | Kill process or use different port |
| `ECONNREFUSED 127.0.0.1:8000` | Make sure backend is running |
| `Unknown at rule @tailwind` | Install Tailwind CSS: `npm install tailwindcss` |

## 📖 API Documentation

### Core Endpoints

#### Dataset Management
- `POST /api/upload-dataset` - Upload and analyze dataset
- `GET /api/datasets` - List all datasets
- `GET /api/dataset/{id}` - Get dataset details
- `GET /api/dataset/{id}/eda` - Perform exploratory data analysis

#### Model Training
- `POST /api/train-model` - Start model training
- `GET /api/training-jobs` - List training jobs
- `GET /api/models` - List trained models
- `DELETE /api/model/{id}` - Delete model

#### Prediction & Deployment
- `POST /api/predict` - Make predictions
- `POST /api/execute-code` - Execute Python code

#### System
- `GET /api/health` - Health check
- `GET /api/system-info` - System information

### Example API Usage

#### Upload Dataset
```bash
curl -X POST "http://localhost:8000/api/upload-dataset" \
  -F "file=@dataset.csv"
```

#### Train Model
```bash
curl -X POST "http://localhost:8000/api/train-model" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "your-dataset-id",
    "target_column": "target",
    "task_type": "classification",
    "test_size": 0.2
  }'
```

#### Make Prediction
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "your-model-id",
    "data": {
      "feature1": "value1",
      "feature2": "value2"
    }
  }'
```

## 🔒 Security Features

- **CORS Protection** - Configured for localhost development
- **Input Validation** - Pydantic models for API validation
- **File Upload Security** - File type and size restrictions
- **Code Execution Sandboxing** - Subprocess isolation with timeouts
- **Error Handling** - Secure error messages without sensitive data

## 🚀 Production Deployment

### Docker Deployment (Coming Soon)
```dockerfile
# Dockerfile example
FROM python:3.11-slim
WORKDIR /app
COPY backend/ .
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
```bash
# .env file
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_URL=http://localhost:3000
DEBUG=false
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines
- Follow TypeScript/Python type hints
- Use ESLint for frontend code formatting
- Add tests for new features
- Update documentation for API changes

## 📊 Performance Optimization

### Backend Optimization
- Use async/await for I/O operations
- Implement database connection pooling (when adding database)
- Add caching for frequent requests
- Optimize model loading and prediction

### Frontend Optimization
- Code splitting with Vite
- Lazy loading for components
- Image optimization
- Bundle size monitoring

## 🔮 Roadmap

### Version 1.1 (Next Release)
- [ ] Database integration (PostgreSQL/SQLite)
- [ ] User authentication and authorization
- [ ] Model versioning and experiment tracking
- [ ] Advanced hyperparameter optimization with Optuna
- [ ] Real-time collaboration features

### Version 1.2 (Future)
- [ ] Docker containerization
- [ ] Kubernetes deployment manifests
- [ ] Advanced visualization dashboard
- [ ] Plugin system for custom algorithms
- [ ] Multi-tenant support

### Version 2.0 (Long-term)
- [ ] Distributed training with Ray/Dask
- [ ] GPU acceleration support
- [ ] AutoML pipeline automation
- [ ] Model marketplace
- [ ] Cloud provider integrations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI** - Modern web framework for Python APIs
- **React** - Frontend library for building user interfaces
- **Tailwind CSS** - Utility-first CSS framework
- **scikit-learn** - Machine learning library for Python
- **Lucide React** - Beautiful and consistent icon pack

## 📞 Support

- 📧 **Email**: support@aidevlab.os
- 💬 **Discord**: [Join our community](https://discord.gg/aidevlab)
- 📖 **Documentation**: [docs.aidevlab.os](https://docs.aidevlab.os)
- 🐛 **Issues**: [GitHub Issues](https://github.com/username/ai-devlab-os/issues)

---

**Built with ❤️ for the AI community**

Made by developers, for developers. Star ⭐ this repo if you find it useful!1. Python Dependencies Missing
```bash
cd backend
source venv/bin/activate
pip install pandas scikit-learn xgboost matplotlib seaborn
```

####