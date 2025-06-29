import React, { useState, useEffect } from 'react';
import { 
  Archive, 
  Download, 
  Upload, 
  Share2, 
  Trash2, 
  Star, 
  Clock, 
  Database,
  GitBranch,
  Tag,
  FileText,
  Settings,
  Filter,
  Search
} from 'lucide-react';

interface StoredModel {
  model_id: string;
  model_name: string;
  score: number;
  task_type: string;
  created_at: string;
  target_column: string;
}

export const ModelStorage: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [sortBy, setSortBy] = useState('recent');
  const [models, setModels] = useState<StoredModel[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/models');
      if (response.ok) {
        const data = await response.json();
        setModels(data.models);
      }
    } catch (error) {
      console.error('Error loading models:', error);
    } finally {
      setLoading(false);
    }
  };

  const deleteModel = async (modelId: string) => {
    if (!confirm('Are you sure you want to delete this model?')) return;

    try {
      const response = await fetch(`http://localhost:8000/api/model/${modelId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        setModels(models.filter(m => m.model_id !== modelId));
        alert('Model deleted successfully');
      }
    } catch (error) {
      console.error('Error deleting model:', error);
      alert('Error deleting model');
    }
  };

  const filteredModels = models.filter(model => {
    const matchesSearch = model.model_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         model.task_type.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesSearch;
  });

  const sortedModels = [...filteredModels].sort((a, b) => {
    switch (sortBy) {
      case 'recent':
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      case 'accuracy':
        return b.score - a.score;
      case 'name':
        return a.model_name.localeCompare(b.model_name);
      default:
        return 0;
    }
  });

  const getTaskTypeColor = (taskType: string) => {
    switch (taskType) {
      case 'classification':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'regression':
        return 'text-green-600 bg-green-50 border-green-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Model Storage</h1>
          <p className="text-gray-600">Manage your trained models with versioning and metadata</p>
        </div>
        <div className="flex space-x-3">
          <button className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
            <Upload className="w-4 h-4" />
            <span>Import Model</span>
          </button>
          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            <Archive className="w-4 h-4" />
            <span>New Model</span>
          </button>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="bg-white rounded-xl p-6 border border-gray-200">
        <div className="flex items-center space-x-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search models..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          
          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-gray-400" />
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg"
            >
              <option value="all">All Models</option>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>
          
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg"
          >
            <option value="recent">Most Recent</option>
            <option value="accuracy">Highest Accuracy</option>
            <option value="name">Name A-Z</option>
          </select>
        </div>
      </div>

      {/* Models Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {sortedModels.map((model) => (
          <div key={model.model_id} className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-shadow">
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-2">
                  <h3 className="text-lg font-semibold text-gray-900">{model.model_name}</h3>
                  <button className="p-1 hover:bg-gray-100 rounded transition-colors">
                    <Star className="w-4 h-4 text-gray-400" />
                  </button>
                </div>
                <p className="text-sm text-gray-600 mb-3">Target: {model.target_column}</p>
                <div className="flex items-center space-x-2 mb-3">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getTaskTypeColor(model.task_type)}`}>
                    {model.task_type}
                  </span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <p className="text-lg font-bold text-gray-900">{(model.score * 100).toFixed(1)}%</p>
                <p className="text-xs text-gray-600">Accuracy</p>
              </div>
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <p className="text-lg font-bold text-gray-900">
                  {new Date(model.created_at).toLocaleDateString()}
                </p>
                <p className="text-xs text-gray-600">Created</p>
              </div>
            </div>

            <div className="flex items-center justify-between text-xs text-gray-500 mb-4">
              <div className="flex items-center space-x-1">
                <GitBranch className="w-3 h-3" />
                <span>v1.0.0</span>
              </div>
              <div className="flex items-center space-x-1">
                <Clock className="w-3 h-3" />
                <span>{new Date(model.created_at).toLocaleDateString()}</span>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <button className="flex-1 flex items-center justify-center space-x-2 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                <Download className="w-4 h-4" />
                <span>Download</span>
              </button>
              <button className="p-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
                <Share2 className="w-4 h-4 text-gray-600" />
              </button>
              <button className="p-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
                <Settings className="w-4 h-4 text-gray-600" />
              </button>
              <button 
                onClick={() => deleteModel(model.model_id)}
                className="p-2 border border-red-300 rounded-lg hover:bg-red-50 transition-colors"
              >
                <Trash2 className="w-4 h-4 text-red-600" />
              </button>
            </div>
          </div>
        ))}
      </div>

      {sortedModels.length === 0 && (
        <div className="text-center py-12">
          <Archive className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">No models found</h3>
          <p className="text-gray-600 mb-4">
            {searchTerm ? 'No models match your search criteria.' : 'Start by training your first model.'}
          </p>
          <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            Train New Model
          </button>
        </div>
      )}

      {/* Storage Statistics */}
      <div className="bg-white rounded-xl p-6 border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Storage Statistics</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <Database className="w-8 h-8 text-blue-600 mx-auto mb-2" />
            <p className="text-2xl font-bold text-gray-900">{models.length}</p>
            <p className="text-sm text-gray-600">Total Models</p>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <Archive className="w-8 h-8 text-green-600 mx-auto mb-2" />
            <p className="text-2xl font-bold text-gray-900">
              {models.filter(m => m.task_type === 'classification').length}
            </p>
            <p className="text-sm text-gray-600">Classification</p>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <GitBranch className="w-8 h-8 text-purple-600 mx-auto mb-2" />
            <p className="text-2xl font-bold text-gray-900">
              {models.filter(m => m.task_type === 'regression').length}
            </p>
            <p className="text-sm text-gray-600">Regression</p>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <FileText className="w-8 h-8 text-orange-600 mx-auto mb-2" />
            <p className="text-2xl font-bold text-gray-900">
              {models.length > 0 ? (models.reduce((sum, m) => sum + m.score, 0) / models.length * 100).toFixed(0) : 0}%
            </p>
            <p className="text-sm text-gray-600">Avg Accuracy</p>
          </div>
        </div>
      </div>
    </div>
  );
};