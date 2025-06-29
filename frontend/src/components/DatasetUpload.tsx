import React, { useState } from 'react';
import { Upload, FileText, Image, Database, AlertCircle, CheckCircle, BarChart3, Brain, Zap } from 'lucide-react';

interface DatasetInfo {
  dataset_id: string;
  filename: string;
  shape: [number, number];
  columns: string[];
  preview: any[];
  dtypes: Record<string, string>;
}

interface EDAResults {
  statistics: any;
  charts: Array<{type: string, title: string, data: string}>;
  insights: string[];
  data_quality_score: number;
}

export const DatasetUpload: React.FC = () => {
  const [dragActive, setDragActive] = useState(false);
  const [uploadedDataset, setUploadedDataset] = useState<DatasetInfo | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [edaResults, setEdaResults] = useState<EDAResults | null>(null);
  const [loadingEDA, setLoadingEDA] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileUpload = async (file: File) => {
    setAnalyzing(true);
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/api/upload-dataset', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload dataset');
      }

      const data = await response.json();
      setUploadedDataset(data);
      setAnalyzing(false);
      
      // Automatically start EDA
      performEDA(data.dataset_id);
    } catch (error) {
      console.error('Error uploading dataset:', error);
      setAnalyzing(false);
      alert('Error uploading dataset. Please try again.');
    }
  };

  const performEDA = async (datasetId: string) => {
    setLoadingEDA(true);
    
    try {
      const response = await fetch(`http://localhost:8000/api/dataset/${datasetId}/eda`);
      
      if (!response.ok) {
        throw new Error('Failed to perform EDA');
      }

      const edaData = await response.json();
      setEdaResults(edaData);
    } catch (error) {
      console.error('Error performing EDA:', error);
      alert('Error performing analysis. Please try again.');
    } finally {
      setLoadingEDA(false);
    }
  };

  const startTraining = async () => {
    if (!uploadedDataset) return;
    
    // For demo, assume target column is the last column
    const targetColumn = uploadedDataset.columns[uploadedDataset.columns.length - 1];
    
    try {
      const response = await fetch('http://localhost:8000/api/train-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dataset_id: uploadedDataset.dataset_id,
          target_column: targetColumn,
          task_type: 'classification', // Auto-detect in real implementation
          test_size: 0.2
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to start training');
      }

      const data = await response.json();
      alert(`Training started! Job ID: ${data.job_id}`);
      
      // Redirect to training page or show training status
      // You could emit an event or use a global state manager here
    } catch (error) {
      console.error('Error starting training:', error);
      alert('Error starting training. Please try again.');
    }
  };

  const insights = edaResults?.insights || [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dataset Upload & Analysis</h1>
        <p className="text-gray-600">Upload your dataset and get AI-powered insights instantly</p>
      </div>

      {!uploadedDataset ? (
        <div className="bg-white rounded-xl p-8 border border-gray-200">
          <div
            className={`border-2 border-dashed rounded-xl p-12 text-center transition-all ${
              dragActive 
                ? 'border-blue-500 bg-blue-50' 
                : 'border-gray-300 hover:border-gray-400'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Upload Your Dataset</h3>
            <p className="text-gray-600 mb-4">Drag and drop your files here, or click to browse</p>
            
            <div className="flex justify-center space-x-4 mb-6">
              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <FileText className="w-4 h-4" />
                <span>CSV, Excel</span>
              </div>
              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <Image className="w-4 h-4" />
                <span>Images</span>
              </div>
              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <Database className="w-4 h-4" />
                <span>JSON, Parquet</span>
              </div>
            </div>
            
            <input
              type="file"
              id="file-upload"
              className="hidden"
              accept=".csv,.xlsx,.json,.parquet,.jpg,.png"
              onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
            />
            <label
              htmlFor="file-upload"
              className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors cursor-pointer"
            >
              Select Files
            </label>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* File Info */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center space-x-4 mb-4">
              <div className="w-12 h-12 bg-green-50 rounded-lg flex items-center justify-center">
                <CheckCircle className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900">{uploadedDataset.filename}</h3>
                <p className="text-gray-600">Dataset ID: {uploadedDataset.dataset_id}</p>
              </div>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-gray-900">{uploadedDataset.shape[0].toLocaleString()}</p>
                <p className="text-sm text-gray-600">Rows</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-gray-900">{uploadedDataset.shape[1]}</p>
                <p className="text-sm text-gray-600">Columns</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-gray-900">{edaResults?.data_quality_score?.toFixed(0) || 'N/A'}%</p>
                <p className="text-sm text-gray-600">Quality Score</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-gray-900">Classification</p>
                <p className="text-sm text-gray-600">Problem Type</p>
              </div>
            </div>
          </div>

          {/* EDA Results */}
          {loadingEDA ? (
            <div className="bg-white rounded-xl p-6 border border-gray-200">
              <div className="flex items-center justify-center h-32">
                <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
                <span className="ml-3 text-gray-600">Analyzing dataset...</span>
              </div>
            </div>
          ) : edaResults ? (
            <>
              {/* AI Insights */}
              <div className="bg-white rounded-xl p-6 border border-gray-200">
                <div className="flex items-center space-x-2 mb-4">
                  <Brain className="w-5 h-5 text-purple-600" />
                  <h3 className="text-lg font-semibold text-gray-900">AI-Powered Insights</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {insights.slice(0, 3).map((insight, index) => (
                    <div key={index} className="p-4 border border-gray-200 rounded-lg">
                      <div className="flex items-center space-x-3 mb-2">
                        <Zap className="w-5 h-5 text-blue-600" />
                        <h4 className="font-medium text-gray-900">Insight {index + 1}</h4>
                      </div>
                      <p className="text-sm text-gray-600">{insight}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Visualizations */}
              {edaResults.charts.length > 0 && (
                <div className="bg-white rounded-xl p-6 border border-gray-200">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Visualizations</h3>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {edaResults.charts.map((chart, index) => (
                      <div key={index} className="text-center">
                        <h4 className="font-medium text-gray-900 mb-3">{chart.title}</h4>
                        <img 
                          src={chart.data} 
                          alt={chart.title}
                          className="w-full rounded-lg border border-gray-200"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Statistics Summary */}
              <div className="bg-white rounded-xl p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Dataset Statistics</h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-3">Missing Values</h4>
                    <div className="space-y-2">
                      {Object.entries(edaResults.statistics.missing_values).slice(0, 5).map(([col, missing]) => (
                        <div key={col} className="flex justify-between">
                          <span className="text-sm text-gray-600">{col}</span>
                          <span className="text-sm font-medium text-gray-900">{missing as number}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900 mb-3">Data Types</h4>
                    <div className="space-y-2">
                      {Object.entries(uploadedDataset.dtypes).slice(0, 5).map(([col, dtype]) => (
                        <div key={col} className="flex justify-between">
                          <span className="text-sm text-gray-600">{col}</span>
                          <span className="text-sm font-medium text-gray-900">{dtype}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </>
          ) : null}

          {/* Feature Preview */}
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Preview</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    {uploadedDataset.columns.map((column, index) => (
                      <th key={index} className="text-left py-3 px-4 font-medium text-gray-900">
                        {column}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {uploadedDataset.preview.slice(0, 5).map((row, rowIndex) => (
                    <tr key={rowIndex} className="border-b border-gray-100">
                      {uploadedDataset.columns.map((column, colIndex) => (
                        <td key={colIndex} className="py-3 px-4 text-gray-600">
                          {typeof row[column] === 'object' ? JSON.stringify(row[column]) : String(row[column] ?? 'N/A')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Next Steps */}
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-6 border border-blue-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Ready for Model Training</h3>
            <p className="text-gray-600 mb-4">Your dataset is analyzed and ready for AI model training. We recommend starting with classification algorithms.</p>
            <div className="flex space-x-3">
              <button 
                onClick={startTraining}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Start Auto-Training
              </button>
              <button className="px-6 py-2 border border-gray-300 rounded-lg hover:bg-white transition-colors">
                Custom Training
              </button>
            </div>
          </div>
        </div>
      )}

      {analyzing && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-8 text-center">
            <div className="animate-spin w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full mx-auto mb-4"></div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Processing Dataset</h3>
            <p className="text-gray-600">Uploading and analyzing your data...</p>
          </div>
        </div>
      )}
    </div>
  );
};