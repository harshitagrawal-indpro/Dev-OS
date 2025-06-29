import React, { useState, useEffect } from 'react';
import { 
  Rocket, 
  Globe, 
  Copy, 
  Settings, 
  Shield, 
  BarChart3, 
  AlertCircle,
  CheckCircle,
  ExternalLink,
  Play,
  Send
} from 'lucide-react';

interface Model {
  model_id: string;
  model_name: string;
  score: number;
  task_type: string;
  created_at: string;
  target_column: string;
}

interface DeployedAPI {
  id: string;
  name: string;
  model: string;
  endpoint: string;
  status: 'active' | 'inactive' | 'deploying';
  requests: number;
  latency: number;
  uptime: string;
}

export const APIDeployment: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState('');
  const [models, setModels] = useState<Model[]>([]);
  const [testInput, setTestInput] = useState('{}');
  const [testResult, setTestResult] = useState<any>(null);
  const [testLoading, setTestLoading] = useState(false);
  const [deploymentConfig, setDeploymentConfig] = useState({
    name: 'my-model-api',
    version: 'v1.0.0',
    autoscaling: true,
    authentication: true,
    rateLimit: '1000/hour'
  });

  // Mock deployed APIs - in real app, this would come from backend
  const deployedAPIs: DeployedAPI[] = [
    {
      id: '1',
      name: 'Customer Churn API',
      model: 'XGBoost Classifier',
      endpoint: 'http://localhost:8000/api/predict',
      status: 'active',
      requests: 1247,
      latency: 45,
      uptime: '99.9%'
    },
    {
      id: '2',
      name: 'Sentiment Analysis API',
      model: 'BERT Model',
      endpoint: 'http://localhost:8000/api/predict',
      status: 'active',
      requests: 2156,
      latency: 120,
      uptime: '99.7%'
    }
  ];

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/models');
      if (response.ok) {
        const data = await response.json();
        setModels(data.models);
        if (data.models.length > 0) {
          setSelectedModel(data.models[0].model_id);
          generateTestInput(data.models[0]);
        }
      }
    } catch (error) {
      console.error('Error loading models:', error);
    }
  };

  const generateTestInput = (model: Model) => {
    // Generate sample input based on model type
    const sampleInputs = {
      classification: {
        age: 35,
        income: 75000,
        tenure: 24,
        satisfaction_score: 8.5
      },
      regression: {
        age: 35,
        income: 75000,
        tenure: 24,
        satisfaction_score: 8.5
      }
    };

    const input = sampleInputs[model.task_type as keyof typeof sampleInputs] || sampleInputs.classification;
    setTestInput(JSON.stringify(input, null, 2));
  };

  const testAPI = async () => {
    if (!selectedModel) {
      alert('Please select a model first');
      return;
    }

    setTestLoading(true);
    
    try {
      const inputData = JSON.parse(testInput);
      
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: selectedModel,
          data: inputData
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to make prediction');
      }

      const result = await response.json();
      setTestResult(result);
    } catch (error) {
      console.error('Error testing API:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setTestResult({ error: errorMessage });
    } finally {
      setTestLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    alert('Copied to clipboard!');
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-600 bg-green-50';
      case 'deploying': return 'text-blue-600 bg-blue-50';
      case 'inactive': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return CheckCircle;
      case 'deploying': return Play;
      case 'inactive': return AlertCircle;
      default: return AlertCircle;
    }
  };

  const generateCurlCommand = () => {
    return `curl -X POST "http://localhost:8000/api/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_id": "${selectedModel}",
    "data": ${testInput}
  }'`;
  };

  const generatePythonCode = () => {
    return `import requests
import json

url = "http://localhost:8000/api/predict"
headers = {"Content-Type": "application/json"}
data = {
    "model_id": "${selectedModel}",
    "data": ${testInput}
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(result)`;
  };

  const generateJavaScriptCode = () => {
    return `fetch('http://localhost:8000/api/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    model_id: '${selectedModel}',
    data: ${testInput}
  })
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));`;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">API Deployment</h1>
          <p className="text-gray-600">Deploy your AI models as production-ready APIs</p>
        </div>
        <div className="flex space-x-3">
          <button 
            onClick={() => window.open('http://localhost:8000/docs', '_blank')}
            className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            View Swagger Docs
          </button>
          <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            Deploy New API
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* API Testing */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Test Your API</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Select Model</label>
                <select 
                  value={selectedModel}
                  onChange={(e) => {
                    setSelectedModel(e.target.value);
                    const model = models.find(m => m.model_id === e.target.value);
                    if (model) generateTestInput(model);
                  }}
                  className="w-full p-3 border border-gray-300 rounded-lg"
                >
                  <option value="">Choose a model...</option>
                  {models.map((model) => (
                    <option key={model.model_id} value={model.model_id}>
                      {model.model_name} - {model.task_type} (Score: {(model.score * 100).toFixed(1)}%)
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Request Body (JSON)</label>
                <textarea
                  value={testInput}
                  onChange={(e) => setTestInput(e.target.value)}
                  rows={6}
                  className="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
                  placeholder='{"feature1": "value1", "feature2": "value2"}'
                />
              </div>

              <div className="flex space-x-3">
                <button 
                  onClick={testAPI}
                  disabled={testLoading || !selectedModel}
                  className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Send className="w-4 h-4" />
                  <span>{testLoading ? 'Testing...' : 'Test API'}</span>
                </button>
                <button 
                  onClick={() => copyToClipboard(`http://localhost:8000/api/predict`)}
                  className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Copy Endpoint
                </button>
              </div>
            </div>

            {/* API Response */}
            <div className="mt-6">
              <h4 className="text-md font-semibold text-gray-900 mb-3">Response</h4>
              <div className="bg-gray-900 rounded-lg p-4 min-h-32">
                {testLoading ? (
                  <div className="flex items-center justify-center h-20">
                    <div className="animate-spin w-6 h-6 border-4 border-green-500 border-t-transparent rounded-full"></div>
                  </div>
                ) : testResult ? (
                  <pre className="text-sm text-gray-300 overflow-x-auto">
                    <code>{JSON.stringify(testResult, null, 2)}</code>
                  </pre>
                ) : (
                  <div className="flex items-center justify-center h-20 text-gray-500">
                    Click "Test API" to see the response
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Code Examples */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Code Examples</h3>
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-gray-900 mb-2">cURL</h4>
                <div className="bg-gray-900 rounded-lg p-4 relative">
                  <button 
                    onClick={() => copyToClipboard(generateCurlCommand())}
                    className="absolute top-2 right-2 p-2 text-gray-400 hover:text-white"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                  <pre className="text-sm text-gray-300 overflow-x-auto pr-10">
                    <code>{generateCurlCommand()}</code>
                  </pre>
                </div>
              </div>

              <div>
                <h4 className="font-medium text-gray-900 mb-2">Python</h4>
                <div className="bg-gray-900 rounded-lg p-4 relative">
                  <button 
                    onClick={() => copyToClipboard(generatePythonCode())}
                    className="absolute top-2 right-2 p-2 text-gray-400 hover:text-white"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                  <pre className="text-sm text-gray-300 overflow-x-auto pr-10">
                    <code>{generatePythonCode()}</code>
                  </pre>
                </div>
              </div>

              <div>
                <h4 className="font-medium text-gray-900 mb-2">JavaScript</h4>
                <div className="bg-gray-900 rounded-lg p-4 relative">
                  <button 
                    onClick={() => copyToClipboard(generateJavaScriptCode())}
                    className="absolute top-2 right-2 p-2 text-gray-400 hover:text-white"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                  <pre className="text-sm text-gray-300 overflow-x-auto pr-10">
                    <code>{generateJavaScriptCode()}</code>
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Deployment Sidebar */}
        <div className="space-y-6">
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Deploy</h3>
            <div className="space-y-4">
              <button className="w-full p-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center space-x-2">
                <Rocket className="w-5 h-5" />
                <span>Deploy Selected Model</span>
              </button>
              <div className="text-center">
                <p className="text-sm text-gray-600">Estimated deployment time: <strong>2-3 minutes</strong></p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">API Statistics</h3>
            <div className="space-y-4">
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-gray-900">{deployedAPIs.filter(api => api.status === 'active').length}</p>
                <p className="text-sm text-gray-600">Active APIs</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-gray-900">{deployedAPIs.reduce((sum, api) => sum + api.requests, 0).toLocaleString()}</p>
                <p className="text-sm text-gray-600">Total Requests</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-gray-900">99.8%</p>
                <p className="text-sm text-gray-600">Avg Uptime</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Security & Performance</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-2">
                  <Shield className="w-4 h-4 text-green-500" />
                  <span className="text-sm">HTTPS Enabled</span>
                </div>
                <CheckCircle className="w-4 h-4 text-green-500" />
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-2">
                  <Settings className="w-4 h-4 text-blue-500" />
                  <span className="text-sm">Auto-scaling</span>
                </div>
                <CheckCircle className="w-4 h-4 text-green-500" />
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-2">
                  <BarChart3 className="w-4 h-4 text-purple-500" />
                  <span className="text-sm">Rate Limiting</span>
                </div>
                <CheckCircle className="w-4 h-4 text-green-500" />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Deployed APIs */}
      <div className="bg-white rounded-xl p-6 border border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Deployed APIs</h3>
          <div className="flex items-center space-x-2">
            <Globe className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-600">Production endpoints</span>
          </div>
        </div>
        <div className="space-y-4">
          {deployedAPIs.map((api) => {
            const StatusIcon = getStatusIcon(api.status);
            return (
              <div key={api.id} className="flex items-center space-x-4 p-4 border border-gray-200 rounded-lg">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${getStatusColor(api.status)}`}>
                  <StatusIcon className="w-4 h-4" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <h4 className="font-medium text-gray-900">{api.name}</h4>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(api.status)}`}>
                      {api.status}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mb-2">{api.model}</p>
                  <p className="text-xs text-gray-500 mb-2">{api.endpoint}</p>
                  <div className="flex items-center space-x-4 text-xs text-gray-500">
                    <span>{api.requests.toLocaleString()} requests</span>
                    <span>{api.latency}ms avg latency</span>
                    <span>{api.uptime} uptime</span>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button 
                    onClick={() => copyToClipboard(api.endpoint)}
                    className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                  <button className="p-2 text-gray-600 hover:bg-gray-50 rounded-lg transition-colors">
                    <Settings className="w-4 h-4" />
                  </button>
                  <button 
                    onClick={() => window.open(api.endpoint.replace('/api/predict', '/docs'), '_blank')}
                    className="p-2 text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
                  >
                    <ExternalLink className="w-4 h-4" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};