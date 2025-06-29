import React, { useState } from 'react';
import { 
  Palette, 
  Layout, 
  Type, 
  Square, 
  Circle, 
  Image, 
  MousePointer, 
  Smartphone,
  Monitor,
  Tablet,
  Eye,
  Code,
  Download
} from 'lucide-react';

interface UIComponent {
  id: string;
  type: string;
  label: string;
  icon: React.ComponentType<any>;
  properties: Record<string, any>;
}

export const UIBuilder: React.FC = () => {
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [viewport, setViewport] = useState<'mobile' | 'tablet' | 'desktop'>('desktop');
  const [components, setComponents] = useState<UIComponent[]>([]);

  const componentLibrary = [
    { id: 'button', type: 'button', label: 'Button', icon: Square },
    { id: 'input', type: 'input', label: 'Input Field', icon: Type },
    { id: 'card', type: 'card', label: 'Card', icon: Layout },
    { id: 'image', type: 'image', label: 'Image', icon: Image },
    { id: 'text', type: 'text', label: 'Text', icon: Type },
    { id: 'form', type: 'form', label: 'Form', icon: Square },
  ];

  const [generatedCode] = useState(`import React, { useState } from 'react';

const PredictionInterface = () => {
  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: input })
    });
    const data = await response.json();
    setResult(data.prediction);
  };

  return (
    <div className="max-w-md mx-auto bg-white rounded-xl shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4">AI Prediction</h2>
      <div className="space-y-4">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter your data..."
          className="w-full p-3 border rounded-lg"
        />
        <button
          onClick={handlePredict}
          className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700"
        >
          Get Prediction
        </button>
        {result && (
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="font-medium">Prediction: {result}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionInterface;`);

  const templates = [
    {
      name: 'ML Prediction Form',
      description: 'Clean form for model predictions',
      preview: 'https://images.pexels.com/photos/270348/pexels-photo-270348.jpeg?auto=compress&cs=tinysrgb&w=300&h=200'
    },
    {
      name: 'Data Dashboard',
      description: 'Analytics dashboard with charts',
      preview: 'https://images.pexels.com/photos/265087/pexels-photo-265087.jpeg?auto=compress&cs=tinysrgb&w=300&h=200'
    },
    {
      name: 'API Documentation',
      description: 'Interactive API documentation',
      preview: 'https://images.pexels.com/photos/270632/pexels-photo-270632.jpeg?auto=compress&cs=tinysrgb&w=300&h=200'
    }
  ];

  const getViewportIcon = () => {
    switch (viewport) {
      case 'mobile': return Smartphone;
      case 'tablet': return Tablet;
      case 'desktop': return Monitor;
      default: return Monitor;
    }
  };

  const getViewportDimensions = () => {
    switch (viewport) {
      case 'mobile': return 'w-80 h-96';
      case 'tablet': return 'w-96 h-80';
      case 'desktop': return 'w-full h-96';
      default: return 'w-full h-96';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">UI Builder</h1>
          <p className="text-gray-600">Create beautiful interfaces for your AI models</p>
        </div>
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 bg-gray-100 rounded-lg p-1">
            {(['mobile', 'tablet', 'desktop'] as const).map((size) => {
              const Icon = size === 'mobile' ? Smartphone : size === 'tablet' ? Tablet : Monitor;
              return (
                <button
                  key={size}
                  onClick={() => setViewport(size)}
                  className={`p-2 rounded-md transition-colors ${
                    viewport === size ? 'bg-white shadow-sm' : 'hover:bg-gray-200'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                </button>
              );
            })}
          </div>
          <button className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
            <Eye className="w-4 h-4" />
            <span>Preview</span>
          </button>
          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            <Download className="w-4 h-4" />
            <span>Export</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Component Library */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Components</h3>
            <div className="space-y-2">
              {componentLibrary.map((component) => {
                const Icon = component.icon;
                return (
                  <button
                    key={component.id}
                    onClick={() => setSelectedComponent(component.id)}
                    className={`w-full flex items-center space-x-3 p-3 rounded-lg border-2 border-dashed transition-all ${
                      selectedComponent === component.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-300 hover:border-gray-400'
                    }`}
                  >
                    <Icon className="w-5 h-5 text-gray-500" />
                    <span className="text-sm font-medium text-gray-900">{component.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Templates</h3>
            <div className="space-y-3">
              {templates.map((template, index) => (
                <div key={index} className="border border-gray-200 rounded-lg overflow-hidden">
                  <img 
                    src={template.preview} 
                    alt={template.name}
                    className="w-full h-24 object-cover"
                  />
                  <div className="p-3">
                    <h4 className="font-medium text-gray-900 text-sm">{template.name}</h4>
                    <p className="text-xs text-gray-600 mb-2">{template.description}</p>
                    <button className="text-xs text-blue-600 hover:text-blue-700">
                      Use Template
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Canvas */}
        <div className="lg:col-span-3">
          <div className="bg-white rounded-xl p-6 border border-gray-200 h-full">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Design Canvas</h3>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600">Viewport:</span>
                <div className="flex items-center space-x-1">
                  {React.createElement(getViewportIcon(), { className: "w-4 h-4 text-gray-400" })}
                  <span className="text-sm font-medium text-gray-700 capitalize">{viewport}</span>
                </div>
              </div>
            </div>
            
            <div className="flex justify-center">
              <div className={`${getViewportDimensions()} border-2 border-dashed border-gray-300 rounded-lg bg-gray-50 relative overflow-hidden`}>
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <MousePointer className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-500 mb-2">Drag components here</p>
                    <p className="text-sm text-gray-400">or start with a template</p>
                  </div>
                </div>
                
                {/* Sample Preview */}
                <div className="absolute inset-4 bg-white rounded-lg shadow-lg p-6 opacity-30">
                  <div className="space-y-4">
                    <div className="h-8 bg-gray-200 rounded"></div>
                    <div className="h-12 bg-blue-100 rounded"></div>
                    <div className="h-8 bg-gray-100 rounded"></div>
                    <div className="h-10 bg-blue-500 rounded"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Properties Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Properties</h3>
            {selectedComponent ? (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Width</label>
                  <input
                    type="text"
                    placeholder="100%"
                    className="w-full p-2 border border-gray-300 rounded-lg text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Height</label>
                  <input
                    type="text"
                    placeholder="auto"
                    className="w-full p-2 border border-gray-300 rounded-lg text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Background</label>
                  <input
                    type="color"
                    defaultValue="#ffffff"
                    className="w-full h-10 border border-gray-300 rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Border Radius</label>
                  <input
                    type="range"
                    min="0"
                    max="20"
                    defaultValue="8"
                    className="w-full"
                  />
                </div>
              </div>
            ) : (
              <p className="text-sm text-gray-500">Select a component to edit properties</p>
            )}
          </div>

          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Actions</h3>
            <div className="space-y-3">
              <button className="w-full p-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:from-purple-600 hover:to-pink-600 transition-all">
                <div className="flex items-center space-x-2">
                  <Palette className="w-4 h-4" />
                  <span className="text-sm">Generate from Prompt</span>
                </div>
              </button>
              <button className="w-full p-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center space-x-2">
                  <Layout className="w-4 h-4" />
                  <span className="text-sm">Optimize Layout</span>
                </div>
              </button>
              <button className="w-full p-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-center space-x-2">
                  <Code className="w-4 h-4" />
                  <span className="text-sm">Generate Code</span>
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Generated Code Preview */}
      <div className="bg-white rounded-xl p-6 border border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Generated Code</h3>
          <button className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors">
            <Download className="w-4 h-4" />
            <span>Download React Component</span>
          </button>
        </div>
        <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm text-gray-300">
            <code>{generatedCode}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};