import React from 'react';
import { 
  TrendingUp, 
  Database, 
  Brain, 
  Rocket, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  BarChart3,
  Activity,
  Zap
} from 'lucide-react';

export const Dashboard: React.FC = () => {
  const stats = [
    { label: 'Active Models', value: '24', trend: '+12%', icon: Brain, color: 'blue' },
    { label: 'Deployed APIs', value: '18', trend: '+8%', icon: Rocket, color: 'green' },
    { label: 'Datasets', value: '47', trend: '+15%', icon: Database, color: 'purple' },
    { label: 'Predictions', value: '15.2K', trend: '+23%', icon: Activity, color: 'orange' },
  ];

  const recentActivity = [
    { type: 'model', title: 'Customer Churn Predictor', status: 'deployed', time: '2 hours ago' },
    { type: 'dataset', title: 'Sales Data Q4 2024', status: 'analyzed', time: '4 hours ago' },
    { type: 'api', title: 'Sentiment Analysis API', status: 'testing', time: '6 hours ago' },
    { type: 'training', title: 'Image Classifier v2', status: 'training', time: '1 day ago' },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deployed': return 'text-green-600 bg-green-50';
      case 'training': return 'text-blue-600 bg-blue-50';
      case 'testing': return 'text-orange-600 bg-orange-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'deployed': return CheckCircle;
      case 'training': return Clock;
      case 'testing': return AlertCircle;
      default: return Clock;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">AI DevLab Dashboard</h1>
          <p className="text-gray-600">Monitor your AI models, APIs, and datasets in real-time</p>
        </div>
        <div className="flex space-x-3">
          <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            New Project
          </button>
          <button className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
            Import Model
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <div key={stat.label} className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-shadow">
              <div className="flex items-center justify-between mb-4">
                <div className={`w-12 h-12 rounded-lg bg-${stat.color}-50 flex items-center justify-center`}>
                  <Icon className={`w-6 h-6 text-${stat.color}-600`} />
                </div>
                <span className="text-sm font-medium text-green-600">{stat.trend}</span>
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
                <p className="text-sm text-gray-600">{stat.label}</p>
              </div>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Activity */}
        <div className="bg-white rounded-xl p-6 border border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Recent Activity</h3>
            <button className="text-sm text-blue-600 hover:text-blue-700">View all</button>
          </div>
          <div className="space-y-4">
            {recentActivity.map((activity, index) => {
              const StatusIcon = getStatusIcon(activity.status);
              return (
                <div key={index} className="flex items-center space-x-4 p-3 rounded-lg hover:bg-gray-50 transition-colors">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${getStatusColor(activity.status)}`}>
                    <StatusIcon className="w-4 h-4" />
                  </div>
                  <div className="flex-1">
                    <p className="font-medium text-gray-900">{activity.title}</p>
                    <p className="text-sm text-gray-500">{activity.time}</p>
                  </div>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(activity.status)}`}>
                    {activity.status}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Performance Chart */}
        <div className="bg-white rounded-xl p-6 border border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Model Performance</h3>
            <div className="flex items-center space-x-2">
              <BarChart3 className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-600">Last 7 days</span>
            </div>
          </div>
          <div className="h-48 flex items-center justify-center bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg">
            <div className="text-center">
              <TrendingUp className="w-12 h-12 text-blue-600 mx-auto mb-2" />
              <p className="text-sm text-gray-600">Performance metrics visualization</p>
              <p className="text-xs text-gray-500">Connect your models to see real-time data</p>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-xl p-6 border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="p-4 rounded-lg border-2 border-dashed border-gray-300 hover:border-blue-500 hover:bg-blue-50 transition-all group">
            <Database className="w-8 h-8 text-gray-400 group-hover:text-blue-600 mx-auto mb-2" />
            <p className="font-medium text-gray-900">Upload Dataset</p>
            <p className="text-sm text-gray-500">Start with your data</p>
          </button>
          
          <button className="p-4 rounded-lg border-2 border-dashed border-gray-300 hover:border-green-500 hover:bg-green-50 transition-all group">
            <Zap className="w-8 h-8 text-gray-400 group-hover:text-green-600 mx-auto mb-2" />
            <p className="font-medium text-gray-900">Auto-Train Model</p>
            <p className="text-sm text-gray-500">AI-powered training</p>
          </button>
          
          <button className="p-4 rounded-lg border-2 border-dashed border-gray-300 hover:border-purple-500 hover:bg-purple-50 transition-all group">
            <Rocket className="w-8 h-8 text-gray-400 group-hover:text-purple-600 mx-auto mb-2" />
            <p className="font-medium text-gray-900">Deploy API</p>
            <p className="text-sm text-gray-500">One-click deployment</p>
          </button>
        </div>
      </div>
    </div>
  );
};