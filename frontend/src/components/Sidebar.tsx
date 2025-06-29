import React from 'react';
import { 
  BarChart3, 
  Database, 
  Brain, 
  Rocket, 
  Code, 
  Palette, 
  TestTube, 
  Archive,
  Menu,
  Bot
} from 'lucide-react';
import { TabType } from '../App';

interface SidebarProps {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
  isOpen: boolean;
  onToggle: () => void;
}

const navigation = [
  { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
  { id: 'dataset', label: 'Dataset Upload', icon: Database },
  { id: 'training', label: 'Model Training', icon: Brain },
  { id: 'deployment', label: 'API Deployment', icon: Rocket },
  { id: 'workspace', label: 'Code Workspace', icon: Code },
  { id: 'ui-builder', label: 'UI Builder', icon: Palette },
  { id: 'api-testing', label: 'API Testing', icon: TestTube },
  { id: 'storage', label: 'Model Storage', icon: Archive },
];

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab, isOpen, onToggle }) => {
  return (
    <div className={`fixed left-0 top-0 h-full bg-white border-r border-gray-200 transition-all duration-300 z-50 ${isOpen ? 'w-64' : 'w-16'}`}>
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className={`flex items-center space-x-3 ${isOpen ? 'opacity-100' : 'opacity-0'} transition-opacity duration-200`}>
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-gray-900">AI DevLab OS</h1>
              <p className="text-xs text-gray-500">Zero-friction AI platform</p>
            </div>
          </div>
          <button
            onClick={onToggle}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <Menu className="w-4 h-4 text-gray-600" />
          </button>
        </div>
      </div>
      
      <nav className="p-4 space-y-2">
        {navigation.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id as TabType)}
              className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-all duration-200 ${
                isActive 
                  ? 'bg-blue-50 text-blue-700 border border-blue-200' 
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
              }`}
            >
              <Icon className={`w-5 h-5 ${isActive ? 'text-blue-600' : 'text-gray-500'}`} />
              <span className={`font-medium ${isOpen ? 'opacity-100' : 'opacity-0'} transition-opacity duration-200`}>
                {item.label}
              </span>
            </button>
          );
        })}
      </nav>
    </div>
  );
};