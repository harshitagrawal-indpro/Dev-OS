import React, { useState, useEffect } from 'react';
import { 
  Code, 
  Play, 
  Save, 
  Download, 
  Upload, 
  Terminal, 
  FileText,
  Lightbulb,
  Zap,
  GitBranch,
  Brain,
  RefreshCw
} from 'lucide-react';

interface ExecutionResult {
  output: string;
  error: string;
  return_code: number;
}

export const CodeWorkspace: React.FC = () => {
  const [code, setCode] = useState(`# AI DevLab Notebook - Customer Churn Analysis
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create sample data
np.random.seed(42)
n_samples = 1000

# Generate synthetic customer data
data = {
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.normal(50000, 20000, n_samples),
    'tenure': np.random.randint(1, 60, n_samples),
    'satisfaction_score': np.random.uniform(1, 10, n_samples),
    'monthly_charges': np.random.uniform(30, 120, n_samples)
}

# Create target variable (churn) based on some logic
data['churn'] = (
    (data['satisfaction_score'] < 5) | 
    (data['monthly_charges'] > 100) | 
    (data['tenure'] < 12)
).astype(int)

df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['churn'].mean():.2%}")

# Feature engineering
df['tenure_years'] = df['tenure'] / 12
df['income_per_charge'] = df['income'] / df['monthly_charges']

# Prepare features
X = df[['age', 'income', 'tenure_years', 'satisfaction_score', 'income_per_charge']]
y = df['churn']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nFeature Importance:")
print(feature_importance)`);

  const [output, setOutput] = useState('');
  const [executing, setExecuting] = useState(false);
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');

  const executeCode = async () => {
    setExecuting(true);
    setOutput('Executing code...\n');
    
    try {
      const response = await fetch('http://localhost:8000/api/execute-code', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code: code,
          dataset_id: selectedDataset || null
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to execute code');
      }

      const result: ExecutionResult = await response.json();
      
      let outputText = '';
      if (result.output) {
        outputText += result.output;
      }
      if (result.error) {
        outputText += `\nERROR:\n${result.error}`;
      }
      if (result.return_code !== 0) {
        outputText += `\nExit code: ${result.return_code}`;
      }
      
      setOutput(outputText || 'Code executed successfully with no output.');
    } catch (error) {
      console.error('Error executing code:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setOutput(`Error: ${errorMessage}`);
    } finally {
      setExecuting(false);
    }
  };

  const saveNotebook = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'notebook.py';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const loadNotebook = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        setCode(content);
      };
      reader.readAsText(file);
    }
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/datasets');
      if (response.ok) {
        const data = await response.json();
        setDatasets(data.datasets);
      }
    } catch (error) {
      console.error('Error loading datasets:', error);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Code Workspace</h1>
          <p className="text-gray-600">Colab-style notebook with AI-powered suggestions</p>
        </div>
        <div className="flex space-x-3">
          <input
            type="file"
            accept=".py,.ipynb,.txt"
            onChange={loadNotebook}
            className="hidden"
            id="file-upload"
          />
          <label htmlFor="file-upload" className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors cursor-pointer">
            <Upload className="w-4 h-4" />
            <span>Import</span>
          </label>
          <button 
            onClick={saveNotebook}
            className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <Download className="w-4 h-4" />
            <span>Export</span>
          </button>
          <button 
            onClick={saveNotebook}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Save className="w-4 h-4" />
            <span>Save</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Notebook Area */}
        <div className="lg:col-span-3 space-y-4">
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <div className="flex space-x-2">
                <button className="px-4 py-2 rounded-lg text-sm font-medium bg-blue-50 text-blue-700 border border-blue-200">
                  Main Notebook
                </button>
              </div>
              <div className="flex items-center space-x-2">
                <select
                  value={selectedDataset}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                  className="text-sm border border-gray-300 rounded px-2 py-1"
                >
                  <option value="">No dataset selected</option>
                  {datasets.map((dataset: any) => (
                    <option key={dataset.dataset_id} value={dataset.dataset_id}>
                      {dataset.filename}
                    </option>
                  ))}
                </select>
                <button 
                  onClick={executeCode}
                  disabled={executing}
                  className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
                >
                  {executing ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                  <span>{executing ? 'Running...' : 'Run All'}</span>
                </button>
              </div>
            </div>

            {/* Code Editor */}
            <div className="p-4">
              <div className="mb-4">
                <div className="flex items-center space-x-2 mb-2">
                  <Code className="w-4 h-4 text-gray-400" />
                  <span className="text-sm font-medium text-gray-700">Python Code</span>
                </div>
                <textarea
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  className="w-full h-96 font-mono text-sm p-4 border border-gray-300 rounded-lg bg-gray-50 resize-none"
                  placeholder="Start coding..."
                />
              </div>

              {/* Output */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <Terminal className="w-4 h-4 text-gray-400" />
                    <span className="text-sm font-medium text-gray-700">Output</span>
                  </div>
                  <button 
                    onClick={() => setOutput('')}
                    className="text-xs text-gray-500 hover:text-gray-700"
                  >
                    Clear
                  </button>
                </div>
                <div className="bg-black text-green-400 font-mono text-sm p-4 rounded-lg h-48 overflow-y-auto">
                  <pre className="whitespace-pre-wrap">{output || 'No output yet. Run some code to see results.'}</pre>
                  {executing && (
                    <div className="flex items-center mt-2">
                      <div className="animate-spin w-4 h-4 border-2 border-green-400 border-t-transparent rounded-full mr-2"></div>
                      <span>Executing code...</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* AI Assistant Panel */}
        <div className="space-y-6">
          {/* AI Code Generator */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center space-x-2 mb-4">
              <Brain className="w-5 h-5 text-purple-500" />
              <h3 className="text-lg font-semibold text-gray-900">AI Assistant</h3>
            </div>
            <div className="space-y-3">
              <button className="w-full p-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:from-purple-600 hover:to-pink-600 transition-all">
                <div className="flex items-center space-x-2">
                  <Zap className="w-4 h-4" />
                  <span className="text-sm">Generate Enhanced Code</span>
                </div>
              </button>
              <button className="w-full p-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center space-x-2">
                  <Lightbulb className="w-4 h-4" />
                  <span className="text-sm">Explain Current Code</span>
                </div>
              </button>
              <button className="w-full p-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center space-x-2">
                  <FileText className="w-4 h-4" />
                  <span className="text-sm">Generate Documentation</span>
                </div>
              </button>
            </div>
          </div>

          {/* Package Manager */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Packages</h3>
            <div className="space-y-2">
              {[
                'pandas==2.1.3',
                'scikit-learn==1.3.2',
                'numpy==1.25.2',
                'matplotlib==3.8.2',
                'seaborn==0.13.0',
                'xgboost==2.0.2'
              ].map((pkg, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <span className="text-sm font-mono text-gray-700">{pkg}</span>
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                </div>
              ))}
            </div>
            <button className="w-full mt-4 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors text-sm">
              Install Package
            </button>
          </div>

          {/* Version Control */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center space-x-2 mb-4">
              <GitBranch className="w-5 h-5 text-gray-400" />
              <h3 className="text-lg font-semibold text-gray-900">Version Control</h3>
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="font-medium text-gray-900 text-sm">Current Version</p>
                  <p className="text-xs text-gray-600">v1.2.3 • 2 hours ago</p>
                </div>
                <button className="text-blue-600 hover:text-blue-700 text-sm">
                  View Changes
                </button>
              </div>
              <button className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm">
                Create Checkpoint
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};