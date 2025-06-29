import React, { useState, useEffect } from 'react';
import { 
  Send, 
  Copy, 
  History, 
  BookOpen, 
  Settings, 
  CheckCircle,
  XCircle,
  Clock,
  Code,
  Download,
  Activity,
  Zap,
  Globe,
  Target,
  Award,
  RefreshCw
} from 'lucide-react';

interface APIRequest {
  id: string;
  method: string;
  endpoint: string;
  status: number;
  responseTime: number;
  timestamp: string;
  success: boolean;
  data?: any;
}

interface Model {
  model_id: string;
  model_name: string;
  score: number;
  task_type: string;
  feature_columns: string[];
}

interface PredictionResult {
  prediction: any;
  confidence?: number;
  model_name: string;
  task_type: string;
  timestamp: string;
  status: string;
}

type CodeLanguage = 'javascript' | 'python' | 'curl';

export default function APITesting() {
  const [selectedEndpoint, setSelectedEndpoint] = useState('/api/predict');
  const [requestMethod, setRequestMethod] = useState('POST');
  const [requestBody, setRequestBody] = useState('');
  const [response, setResponse] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [headers, setHeaders] = useState(`{
  "Content-Type": "application/json"
}`);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [recentRequests, setRecentRequests] = useState<APIRequest[]>([]);
  const [apiStats, setApiStats] = useState({
    totalRequests: 0,
    successRate: 100,
    avgResponseTime: 0,
    lastRequest: null as string | null
  });
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);

  const endpoints = [
    '/api/predict',
    '/api/models',
    '/api/datasets',
    '/api/training-jobs',
    '/api/health',
    '/api/status/realtime'
  ];

  const samplePredictions: Record<string, Record<string, number>> = {
    'iris_dataset': {
      'sepal length (cm)': 5.1,
      'sepal width (cm)': 3.5,
      'petal length (cm)': 1.4,
      'petal width (cm)': 0.2
    },
    'wine_dataset': {
      'alcohol': 13.2,
      'malic_acid': 2.3,
      'ash': 2.4,
      'alcalinity_of_ash': 15.6,
      'magnesium': 127.0
    },
    'cancer_dataset': {
      'mean radius': 14.1,
      'mean texture': 19.4,
      'mean perimeter': 91.2,
      'mean area': 626.0,
      'mean smoothness': 0.09
    }
  };

  useEffect(() => {
    loadModels();
    setupWebSocket();
    loadInitialStats();
    
    return () => {
      if (websocket) {
        websocket.close();
      }
    };
  }, []);

  const setupWebSocket = () => {
    try {
      const ws = new WebSocket('ws://localhost:8000/ws');
      
      ws.onopen = () => {
        console.log('API Testing WebSocket connected');
        setWebsocket(ws);
      };
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type === 'prediction_made') {
            updateApiStats();
          }
        } catch (error) {
          console.log('API Testing WebSocket message:', event.data);
        }
      };
      
      ws.onclose = () => {
        console.log('API Testing WebSocket disconnected');
        setTimeout(setupWebSocket, 3000);
      };
    } catch (error) {
      console.error('Failed to setup API testing WebSocket:', error);
    }
  };

  const loadModels = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/models');
      if (response.ok) {
        const data = await response.json();
        setModels(data.models || []);
        
        // Auto-select first model and generate sample data
        if (data.models && data.models.length > 0) {
          const firstModel = data.models[0];
          setSelectedModel(firstModel.model_id);
          generateSampleData(firstModel);
        }
      }
    } catch (error) {
      console.error('Error loading models:', error);
    }
  };

  const generateSampleData = (model: Model) => {
    // Generate realistic sample data based on model
    let sampleData: Record<string, number> = {};
    
    // Use predefined samples if available
    const datasetKey = Object.keys(samplePredictions).find(key => 
      model.model_name.toLowerCase().includes(key.replace('_dataset', ''))
    );
    
    if (datasetKey) {
      sampleData = samplePredictions[datasetKey];
    } else {
      // Generate generic sample data
      model.feature_columns?.forEach((feature, index) => {
        sampleData[feature] = Math.random() * 100;
      });
    }

    const requestData = {
      model_id: model.model_id,
      data: sampleData
    };

    setRequestBody(JSON.stringify(requestData, null, 2));
  };

  const loadInitialStats = () => {
    // Initialize with some demo stats
    setApiStats({
      totalRequests: Math.floor(Math.random() * 500) + 100,
      successRate: 95 + Math.random() * 4,
      avgResponseTime: 25 + Math.random() * 20,
      lastRequest: new Date().toISOString()
    });

    // Add some demo requests
    const demoRequests: APIRequest[] = [
      {
        id: '1',
        method: 'POST',
        endpoint: '/api/predict',
        status: 200,
        responseTime: 45,
        timestamp: '2 minutes ago',
        success: true
      },
      {
        id: '2',
        method: 'GET',
        endpoint: '/api/models',
        status: 200,
        responseTime: 23,
        timestamp: '5 minutes ago',
        success: true
      },
      {
        id: '3',
        method: 'GET',
        endpoint: '/api/health',
        status: 200,
        responseTime: 12,
        timestamp: '8 minutes ago',
        success: true
      }
    ];
    setRecentRequests(demoRequests);
  };

  const updateApiStats = () => {
    setApiStats(prev => ({
      ...prev,
      totalRequests: prev.totalRequests + 1,
      lastRequest: new Date().toISOString()
    }));
  };

  const sendRequest = async () => {
    setLoading(true);
    const startTime = Date.now();
    
    try {
      const url = `http://localhost:8000${selectedEndpoint}`;
      const options: RequestInit = {
        method: requestMethod,
        headers: JSON.parse(headers),
      };

      if (requestMethod !== 'GET' && requestBody.trim()) {
        options.body = requestBody;
      }

      const response = await fetch(url, options);
      const responseTime = Date.now() - startTime;
      const data = await response.json();
      
      const formattedResponse = JSON.stringify(data, null, 2);
      setResponse(formattedResponse);

      // Add to recent requests
      const newRequest: APIRequest = {
        id: Date.now().toString(),
        method: requestMethod,
        endpoint: selectedEndpoint,
        status: response.status,
        responseTime,
        timestamp: 'Just now',
        success: response.ok,
        data
      };

      setRecentRequests(prev => [newRequest, ...prev.slice(0, 9)]);
      updateApiStats();

      // Show success notification for predictions
      if (selectedEndpoint === '/api/predict' && response.ok && data.prediction !== undefined) {
        const predictionText = typeof data.prediction === 'number' 
          ? data.prediction.toFixed(2)
          : data.prediction.toString();
        
        alert(`🎯 Prediction Result: ${predictionText}\n${data.confidence ? `Confidence: ${(data.confidence * 100).toFixed(1)}%` : ''}\nModel: ${data.model_name}`);
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setResponse(`Error: ${errorMessage}`);
      
      const newRequest: APIRequest = {
        id: Date.now().toString(),
        method: requestMethod,
        endpoint: selectedEndpoint,
        status: 0,
        responseTime: Date.now() - startTime,
        timestamp: 'Just now',
        success: false
      };
      
      setRecentRequests(prev => [newRequest, ...prev.slice(0, 9)]);
    } finally {
      setLoading(false);
    }
  };

  const quickTest = async (endpoint: string, method: string = 'GET') => {
    setSelectedEndpoint(endpoint);
    setRequestMethod(method);
    
    if (method === 'GET') {
      setRequestBody('');
    }
    
    // Auto-send request after brief delay
    setTimeout(() => {
      sendRequest();
    }, 300);
  };

  const testWithModel = (model: Model) => {
    setSelectedModel(model.model_id);
    setSelectedEndpoint('/api/predict');
    setRequestMethod('POST');
    generateSampleData(model);
    
    setTimeout(() => {
      sendRequest();
    }, 500);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    alert('Copied to clipboard!');
  };

  const generateCode = (language: CodeLanguage): string => {
    const examples: Record<CodeLanguage, string> = {
      javascript: `const response = await fetch('http://localhost:8000${selectedEndpoint}', {
  method: '${requestMethod}',
  headers: ${headers},
  body: ${requestMethod !== 'GET' ? `JSON.stringify(${requestBody})` : 'null'}
});
const data = await response.json();
console.log(data);`,
      python: `import requests
import json

url = 'http://localhost:8000${selectedEndpoint}'
headers = ${headers}
${requestMethod !== 'GET' ? `data = ${requestBody}

response = requests.${requestMethod.toLowerCase()}(url, headers=headers, json=data)` : `response = requests.${requestMethod.toLowerCase()}(url, headers=headers)`}
result = response.json()
print(result)`,
      curl: `curl -X ${requestMethod} \\
  'http://localhost:8000${selectedEndpoint}' \\
  -H 'Content-Type: application/json' \\
  ${requestMethod !== 'GET' ? `-d '${requestBody.replace(/\n/g, ' ')}'` : ''}`
    };
    
    return examples[language];
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">API Testing Suite</h1>
          <p className="text-gray-600">Test your AI models with real-time response monitoring</p>
        </div>
        <div className="flex space-x-3">
          <button className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
            <BookOpen className="w-4 h-4" />
            <span>Documentation</span>
          </button>
          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            <Activity className="w-4 h-4" />
            <span>Live Testing</span>
          </button>
        </div>
      </div>

      {/* Quick Test Actions */}
      <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-xl p-6 border border-green-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">🚀 Quick API Tests</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button
            onClick={() => quickTest('/api/health')}
            disabled={loading}
            className="p-4 bg-white border-2 border-green-200 rounded-lg hover:border-green-400 transition-all disabled:opacity-50"
          >
            <div className="flex items-center space-x-3 mb-2">
              <CheckCircle className="w-6 h-6 text-green-600" />
              <div className="text-left">
                <h4 className="font-semibold text-gray-900">Health Check</h4>
                <p className="text-sm text-gray-600">Test API availability</p>
              </div>
            </div>
          </button>

          <button
            onClick={() => quickTest('/api/models')}
            disabled={loading}
            className="p-4 bg-white border-2 border-blue-200 rounded-lg hover:border-blue-400 transition-all disabled:opacity-50"
          >
            <div className="flex items-center space-x-3 mb-2">
              <Globe className="w-6 h-6 text-blue-600" />
              <div className="text-left">
                <h4 className="font-semibold text-gray-900">List Models</h4>
                <p className="text-sm text-gray-600">Get available models</p>
              </div>
            </div>
          </button>

          <button
            onClick={() => quickTest('/api/status/realtime')}
            disabled={loading}
            className="p-4 bg-white border-2 border-purple-200 rounded-lg hover:border-purple-400 transition-all disabled:opacity-50"
          >
            <div className="flex items-center space-x-3 mb-2">
              <Activity className="w-6 h-6 text-purple-600" />
              <div className="text-left">
                <h4 className="font-semibold text-gray-900">Real-time Status</h4>
                <p className="text-sm text-gray-600">Live system metrics</p>
              </div>
            </div>
          </button>
        </div>
      </div>

      {/* Model Testing Section */}
      {models.length > 0 && (
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">🎯 Test AI Models</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {models.slice(0, 3).map((model) => (
              <button
                key={model.model_id}
                onClick={() => testWithModel(model)}
                disabled={loading}
                className="p-4 bg-white border-2 border-gray-200 rounded-lg hover:border-purple-400 transition-all disabled:opacity-50 text-left"
              >
                <div className="flex items-center space-x-3 mb-2">
                  <Target className="w-6 h-6 text-purple-600" />
                  <div>
                    <h4 className="font-semibold text-gray-900">{model.model_name}</h4>
                    <p className="text-sm text-gray-600">{(model.score * 100).toFixed(1)}% accuracy</p>
                  </div>
                  {model.score > 0.95 && <Award className="w-4 h-4 text-yellow-500" />}
                </div>
                <div className="text-xs text-gray-500 capitalize">{model.task_type}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Request Configuration */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Request Configuration</h3>
            
            <div className="space-y-4">
              <div className="flex space-x-4">
                <div className="w-32">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Method</label>
                  <select 
                    value={requestMethod}
                    onChange={(e) => setRequestMethod(e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg"
                  >
                    <option value="GET">GET</option>
                    <option value="POST">POST</option>
                    <option value="PUT">PUT</option>
                    <option value="DELETE">DELETE</option>
                  </select>
                </div>
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Endpoint</label>
                  <select 
                    value={selectedEndpoint}
                    onChange={(e) => setSelectedEndpoint(e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg"
                  >
                    {endpoints.map((endpoint) => (
                      <option key={endpoint} value={endpoint}>{endpoint}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Headers</label>
                <textarea
                  value={headers}
                  onChange={(e) => setHeaders(e.target.value)}
                  rows={3}
                  className="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
                />
              </div>

              {requestMethod !== 'GET' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Request Body</label>
                  <div className="relative">
                    <textarea
                      value={requestBody}
                      onChange={(e) => setRequestBody(e.target.value)}
                      rows={8}
                      className="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
                      placeholder='{"model_id": "your-model-id", "data": {...}}'
                    />
                    {selectedEndpoint === '/api/predict' && models.length > 0 && (
                      <div className="absolute top-2 right-2">
                        <select
                          value={selectedModel}
                          onChange={(e) => {
                            const model = models.find(m => m.model_id === e.target.value);
                            if (model) {
                              setSelectedModel(e.target.value);
                              generateSampleData(model);
                            }
                          }}
                          className="text-xs border border-gray-300 rounded px-2 py-1"
                        >
                          <option value="">Select Model</option>
                          {models.map((model) => (
                            <option key={model.model_id} value={model.model_id}>
                              {model.model_name}
                            </option>
                          ))}
                        </select>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <button
                onClick={sendRequest}
                disabled={loading}
                className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                {loading ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
                <span>{loading ? 'Sending...' : 'Send Request'}</span>
              </button>
            </div>
          </div>

          {/* Response */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Response</h3>
              <div className="flex space-x-2">
                <button 
                  onClick={() => copyToClipboard(response)}
                  className="p-2 text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
                  disabled={!response}
                >
                  <Copy className="w-4 h-4" />
                </button>
                <button className="p-2 text-gray-600 hover:bg-gray-50 rounded-lg transition-colors">
                  <Download className="w-4 h-4" />
                </button>
              </div>
            </div>
            
            <div className="bg-gray-900 rounded-lg p-4 min-h-64">
              {loading ? (
                <div className="flex items-center justify-center h-32">
                  <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
                </div>
              ) : response ? (
                <pre className="text-sm text-gray-300 overflow-x-auto whitespace-pre-wrap">
                  <code>{response}</code>
                </pre>
              ) : (
                <div className="flex items-center justify-center h-32 text-gray-500">
                  No response yet. Send a request to see results.
                </div>
              )}
            </div>
          </div>

          {/* Code Generation */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Code Generation</h3>
            <div className="flex space-x-2 mb-4">
              {(['javascript', 'python', 'curl'] as CodeLanguage[]).map((lang) => (
                <button
                  key={lang}
                  className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm"
                  onClick={() => copyToClipboard(generateCode(lang))}
                >
                  {lang === 'javascript' ? 'JavaScript' : lang === 'python' ? 'Python' : 'cURL'}
                </button>
              ))}
            </div>
            <div className="bg-gray-900 rounded-lg p-4">
              <pre className="text-sm text-gray-300 overflow-x-auto">
                <code>{generateCode('javascript')}</code>
              </pre>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* API Stats */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">📊 Live API Stats</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Total Requests</span>
                <span className="text-sm font-medium text-gray-900">{apiStats.totalRequests.toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Success Rate</span>
                <span className="text-sm font-medium text-green-600">{apiStats.successRate.toFixed(1)}%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Avg Response</span>
                <span className="text-sm font-medium text-blue-600">{apiStats.avgResponseTime.toFixed(0)}ms</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Status</span>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-sm font-medium text-green-600">Operational</span>
                </div>
              </div>
            </div>
          </div>

          {/* Request History */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center space-x-2 mb-4">
              <History className="w-5 h-5 text-gray-400" />
              <h3 className="text-lg font-semibold text-gray-900">Recent Requests</h3>
            </div>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {recentRequests.map((request) => (
                <div key={request.id} className="p-3 border border-gray-200 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        request.success ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
                      }`}>
                        {request.method}
                      </span>
                      <span className="text-xs text-gray-500">{request.status}</span>
                    </div>
                    {request.success ? (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    ) : (
                      <XCircle className="w-4 h-4 text-red-500" />
                    )}
                  </div>
                  <p className="text-sm text-gray-900 mb-1 truncate">{request.endpoint}</p>
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>{request.responseTime}ms</span>
                    <span>{request.timestamp}</span>
                  </div>
                </div>
              ))}
              {recentRequests.length === 0 && (
                <div className="text-center py-4">
                  <Clock className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                  <p className="text-sm text-gray-500">No requests yet</p>
                </div>
              )}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
            <div className="space-y-2">
              <button 
                onClick={() => quickTest('/api/health')}
                className="w-full p-3 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center space-x-2">
                  <Zap className="w-4 h-4 text-green-500" />
                  <span className="text-sm">Quick Health Check</span>
                </div>
              </button>
              <button 
                onClick={() => window.open('http://localhost:8000/docs', '_blank')}
                className="w-full p-3 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center space-x-2">
                  <BookOpen className="w-4 h-4 text-blue-500" />
                  <span className="text-sm">API Documentation</span>
                </div>
              </button>
              <button className="w-full p-3 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center space-x-2">
                  <Code className="w-4 h-4 text-purple-500" />
                  <span className="text-sm">Generate SDK</span>
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}