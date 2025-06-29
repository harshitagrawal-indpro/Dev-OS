import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  Zap, 
  Settings, 
  Play, 
  Pause, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  TrendingUp,
  Database,
  BarChart3,
  RefreshCw,
  Download,
  Eye
} from 'lucide-react';

interface Dataset {
  dataset_id: string;
  filename: string;
  shape: [number, number];
  columns: string[];
}

interface TrainingJob {
  job_id: string;
  status: 'queued' | 'training' | 'completed' | 'failed';
  progress: number;
  accuracy: number;
  model_name: string;
  dataset_name: string;
  created_at: string;
}

interface TrainingConfig {
  dataset_id: string;
  target_column: string;
  task_type: 'classification' | 'regression';
  test_size: number;
  algorithms: string[];
  cv_folds: number;
  max_time: number;
}

export const ModelTraining: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [targetColumn, setTargetColumn] = useState<string>('');
  const [taskType, setTaskType] = useState<'classification' | 'regression'>('classification');
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [config, setConfig] = useState<TrainingConfig>({
    dataset_id: '',
    target_column: '',
    task_type: 'classification',
    test_size: 0.2,
    algorithms: ['RandomForest', 'XGBoost', 'LogisticRegression'],
    cv_folds: 5,
    max_time: 3600
  });

  useEffect(() => {
    loadDatasets();
    loadTrainingJobs();
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

  const loadTrainingJobs = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/training-jobs');
      if (response.ok) {
        const data = await response.json();
        setTrainingJobs(data.jobs);
      }
    } catch (error) {
      console.error('Error loading training jobs:', error);
    }
  };

  const startTraining = async () => {
    if (!selectedDataset || !targetColumn) {
      alert('Please select a dataset and target column');
      return;
    }

    setIsTraining(true);
    
    try {
      const response = await fetch('http://localhost:8000/api/train-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dataset_id: selectedDataset,
          target_column: targetColumn,
          task_type: taskType,
          test_size: config.test_size,
          algorithms: config.algorithms,
          cv_folds: config.cv_folds,
          max_time: config.max_time
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to start training');
      }

      const data = await response.json();
      alert(`Training started! Job ID: ${data.job_id}`);
      loadTrainingJobs();
    } catch (error) {
      console.error('Error starting training:', error);
      alert('Error starting training. Please try again.');
    } finally {
      setIsTraining(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return CheckCircle;
      case 'training': return RefreshCw;
      case 'failed': return AlertCircle;
      case 'queued': return Clock;
      default: return Clock;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-50';
      case 'training': return 'text-blue-600 bg-blue-50';
      case 'failed': return 'text-red-600 bg-red-50';
      case 'queued': return 'text-yellow-600 bg-yellow-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const selectedDatasetInfo = datasets.find(d => d.dataset_id === selectedDataset);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Model Training</h1>
          <p className="text-gray-600">Train AI models with automated hyperparameter optimization</p>
        </div>
        <div className="flex space-x-3">
          <button className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
            <Eye className="w-4 h-4" />
            <span>View Experiments</span>
          </button>
          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            <Download className="w-4 h-4" />
            <span>Export Results</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Training Configuration */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Training Configuration</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Dataset</label>
                <select 
                  value={selectedDataset}
                  onChange={(e) => {
                    setSelectedDataset(e.target.value);
                    setConfig({...config, dataset_id: e.target.value});
                  }}
                  className="w-full p-3 border border-gray-300 rounded-lg"
                >
                  <option value="">Select a dataset...</option>
                  {datasets.map((dataset) => (
                    <option key={dataset.dataset_id} value={dataset.dataset_id}>
                      {dataset.filename} ({dataset.shape[0]} rows, {dataset.shape[1]} columns)
                    </option>
                  ))}
                </select>
              </div>

              {selectedDatasetInfo && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Target Column</label>
                  <select 
                    value={targetColumn}
                    onChange={(e) => {
                      setTargetColumn(e.target.value);
                      setConfig({...config, target_column: e.target.value});
                    }}
                    className="w-full p-3 border border-gray-300 rounded-lg"
                  >
                    <option value="">Select target column...</option>
                    {selectedDatasetInfo.columns.map((column) => (
                      <option key={column} value={column}>{column}</option>
                    ))}
                  </select>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Task Type</label>
                <select 
                  value={taskType}
                  onChange={(e) => {
                    setTaskType(e.target.value as 'classification' | 'regression');
                    setConfig({...config, task_type: e.target.value as 'classification' | 'regression'});
                  }}
                  className="w-full p-3 border border-gray-300 rounded-lg"
                >
                  <option value="classification">Classification</option>
                  <option value="regression">Regression</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Test Size</label>
                  <input
                    type="range"
                    min="0.1"
                    max="0.5"
                    step="0.05"
                    value={config.test_size}
                    onChange={(e) => setConfig({...config, test_size: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                  <div className="text-sm text-gray-600 text-center">{(config.test_size * 100).toFixed(0)}%</div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">CV Folds</label>
                  <input
                    type="number"
                    min="3"
                    max="10"
                    value={config.cv_folds}
                    onChange={(e) => setConfig({...config, cv_folds: parseInt(e.target.value)})}
                    className="w-full p-3 border border-gray-300 rounded-lg"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Algorithms to Try</label>
                <div className="grid grid-cols-3 gap-2">
                  {['RandomForest', 'XGBoost', 'LogisticRegression', 'SVM', 'GradientBoosting', 'NeuralNetwork'].map((algo) => (
                    <label key={algo} className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={config.algorithms.includes(algo)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setConfig({...config, algorithms: [...config.algorithms, algo]});
                          } else {
                            setConfig({...config, algorithms: config.algorithms.filter(a => a !== algo)});
                          }
                        }}
                        className="rounded border-gray-300"
                      />
                      <span className="text-sm text-gray-700">{algo}</span>
                    </label>
                  ))}
                </div>
              </div>

              <button
                onClick={startTraining}
                disabled={isTraining || !selectedDataset || !targetColumn}
                className="w-full flex items-center justify-center space-x-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isTraining ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    <span>Starting Training...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    <span>Start Training</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Training Jobs */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Training Jobs</h3>
              <button 
                onClick={loadTrainingJobs}
                className="p-2 text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>
            
            <div className="space-y-4">
              {trainingJobs.map((job) => {
                const StatusIcon = getStatusIcon(job.status);
                return (
                  <div key={job.job_id} className="flex items-center space-x-4 p-4 border border-gray-200 rounded-lg">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${getStatusColor(job.status)}`}>
                      <StatusIcon className={`w-4 h-4 ${job.status === 'training' ? 'animate-spin' : ''}`} />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <h4 className="font-medium text-gray-900">{job.model_name}</h4>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(job.status)}`}>
                          {job.status}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{job.dataset_name}</p>
                      <div className="flex items-center justify-between text-sm text-gray-500">
                        <span>Accuracy: {job.accuracy ? `${(job.accuracy * 100).toFixed(1)}%` : 'N/A'}</span>
                        <span>{job.created_at}</span>
                      </div>
                      {job.status === 'training' && (
                        <div className="mt-2">
                          <div className="flex justify-between text-xs text-gray-500 mb-1">
                            <span>Progress</span>
                            <span>{job.progress}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${job.progress}%` }}
                            ></div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
              {trainingJobs.length === 0 && (
                <div className="text-center py-8">
                  <Brain className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                  <p className="text-gray-500">No training jobs yet</p>
                  <p className="text-sm text-gray-400">Start by configuring and launching your first training job</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">AutoML Features</h3>
            <div className="space-y-3">
              <div className="flex items-center space-x-3 p-3 bg-blue-50 rounded-lg">
                <Zap className="w-5 h-5 text-blue-600" />
                <div>
                  <p className="font-medium text-blue-900">Hyperparameter Tuning</p>
                  <p className="text-sm text-blue-700">Automated optimization</p>
                </div>
              </div>
              <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
                <TrendingUp className="w-5 h-5 text-green-600" />
                <div>
                  <p className="font-medium text-green-900">Feature Engineering</p>
                  <p className="text-sm text-green-700">Smart transformations</p>
                </div>
              </div>
              <div className="flex items-center space-x-3 p-3 bg-purple-50 rounded-lg">
                <BarChart3 className="w-5 h-5 text-purple-600" />
                <div>
                  <p className="font-medium text-purple-900">Model Comparison</p>
                  <p className="text-sm text-purple-700">Best algorithm selection</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Training Stats</h3>
            <div className="space-y-4">
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-gray-900">{trainingJobs.filter(j => j.status === 'completed').length}</p>
                <p className="text-sm text-gray-600">Completed Models</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-gray-900">{trainingJobs.filter(j => j.status === 'training').length}</p>
                <p className="text-sm text-gray-600">Currently Training</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-gray-900">
                  {trainingJobs.length > 0 
                    ? (trainingJobs
                        .filter(j => j.accuracy)
                        .reduce((sum, j) => sum + j.accuracy, 0) / 
                       trainingJobs.filter(j => j.accuracy).length * 100
                      ).toFixed(1) + '%'
                    : 'N/A'
                  }
                </p>
                <p className="text-sm text-gray-600">Avg Accuracy</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
            <div className="space-y-2">
              <button className="w-full p-3 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center space-x-2">
                  <Settings className="w-4 h-4 text-blue-500" />
                  <span className="text-sm">Advanced Settings</span>
                </div>
              </button>
              <button className="w-full p-3 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center space-x-2">
                  <Database className="w-4 h-4 text-green-500" />
                  <span className="text-sm">Import Pre-trained Model</span>
                </div>
              </button>
              <button className="w-full p-3 text-left border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center space-x-2">
                  <Brain className="w-4 h-4 text-purple-500" />
                  <span className="text-sm">Neural Architecture Search</span>
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};