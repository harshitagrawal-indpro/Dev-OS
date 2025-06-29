import React, { useState } from 'react';
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
  Download
} from 'lucide-react';

interface APIRequest {
  id: string;
  method: string;
  endpoint: string;
  status: number;
  responseTime: number;
  timestamp: string;
  success: boolean;
}

type CodeLanguage = 'javascript' | 'python' | 'curl';

export const APITesting: React.FC = () => {
  const [selectedEndpoint, setSelectedEndpoint] = useState('/api/predict');
  const [requestMethod, setRequestMethod] = useState('POST');
  const [requestBody, setRequestBody] = useState(`{
  "model_id": "your-model-id",
  "data": {
    "age": 35,
    "income": 75000,
    "tenure": 24,
    "satisfaction_score": 8.5
  }
}`);
  const [response, setResponse] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [headers, setHeaders] = useState(`{
  "Content-Type": "application/json",
  "Authorization": "Bearer your-api-key",
  "X-API-Version": "v1"
}`);

  const endpoints = [
    '/api/predict',
    '/api/models',
    '/api/datasets',
    '/api/upload-dataset',
    '/api/train-model'
  ];

  const recentRequests: APIRequest[] = [
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
      method: 'POST',
      endpoint: '/api/train-model',
      status: 500,
      responseTime: 1200,
      timestamp: '10 minutes ago',
      success: false
    }
  ];

  const sendRequest = async () => {
    setLoading(true);
    
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
      const data = await response.json();
      
      setResponse(JSON.stringify(data, null, 2));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setResponse(`Error: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
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
data = ${requestBody}

response = requests.${requestMethod.toLowerCase()}(url, headers=headers, json=data)
result = response.json()
print(result)`,
      curl: `curl -X ${requestMethod} \\
  'http://localhost:8000${selectedEndpoint}' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer your-api-key' \\
  ${requestMethod !== 'GET' ? `-d '${requestBody.replace(/\n/g, ' ')}'` : ''}`
    };
    
    return examples[language];
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">API Testing</h1>
          <p className="text-gray-600">Test your deployed APIs with an integrated testing suite</p>
        </div>
        <div className="flex space-x-3">
          <button className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
            <BookOpen className="w-4 h-4" />
            <span>Documentation</span>
          </button>
          <button className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </button>
        </div>
      </div>

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
                  rows={4}
                  className="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
                />
              </div>

              {requestMethod !== 'GET' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Request Body</label>
                  <textarea
                    value={requestBody}
                    onChange={(e) => setRequestBody(e.target.value)}
                    rows={6}
                    className="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
                  />
                </div>
              )}

              <button
                onClick={sendRequest}
                disabled={loading}
                className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                <Send className="w-4 h-4" />
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
                <pre className="text-sm text-gray-300 overflow-x-auto">
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
          {/* Request History */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center space-x-2 mb-4">
              <History className="w-5 h-5 text-gray-400" />
              <h3 className="text-lg font-semibold text-gray-900">Recent Requests</h3>
            </div>
            <div className="space-y-3">
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
                  <p className="text-sm text-gray-900 mb-1">{request.endpoint}</p>
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>{request.responseTime}ms</span>
                    <span>{request.timestamp}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* API Status */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">API Status</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Response Time</span>
                <span className="text-sm font-medium text-green-600">45ms</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Uptime</span>
                <span className="text-sm font-medium text-green-600">99.9%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Requests Today</span>
                <span className="text-sm font-medium text-gray-900">1,247</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Success Rate</span>
                <span className="text-sm font-medium text-green-600">98.7%</span>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
            <div className="space-y-2">
              <button className="w-full p-3 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center space-x-2">
                  <Code className="w-4 h-4 text-blue-500" />
                  <span className="text-sm">Generate Client SDK</span>
                </div>
              </button>
              <button className="w-full p-3 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center space-x-2">
                  <BookOpen className="w-4 h-4 text-green-500" />
                  <span className="text-sm">Export Collection</span>
                </div>
              </button>
              <button className="w-full p-3 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center space-x-2">
                  <Settings className="w-4 h-4 text-gray-500" />
                  <span className="text-sm">Configure Monitoring</span>
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};