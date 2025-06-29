import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { Dashboard } from './components/Dashboard';
import { DatasetUpload } from './components/DatasetUpload';
import { ModelTraining } from './components/ModelTraining';
import { APIDeployment } from './components/APIDeployment';
import { CodeWorkspace } from './components/CodeWorkspace';
import { UIBuilder } from './components/UIBuilder';
import { APITesting } from './components/APITesting';
import { ModelStorage } from './components/ModelStorage';
import { LandingPage } from './components/LandingPage';

export type TabType = 'landing' | 'dashboard' | 'dataset' | 'training' | 'deployment' | 'workspace' | 'ui-builder' | 'api-testing' | 'storage';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('landing');
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const renderContent = () => {
    switch (activeTab) {
      case 'landing':
        return <LandingPage onGetStarted={() => setActiveTab('dashboard')} />;
      case 'dashboard':
        return <Dashboard />;
      case 'dataset':
        return <DatasetUpload />;
      case 'training':
        return <ModelTraining />;
      case 'deployment':
        return <APIDeployment />;
      case 'workspace':
        return <CodeWorkspace />;
      case 'ui-builder':
        return <UIBuilder />;
      case 'api-testing':
        return <APITesting />;
      case 'storage':
        return <ModelStorage />;
      default:
        return <LandingPage onGetStarted={() => setActiveTab('dashboard')} />;
    }
  };

  // Show full-screen landing page
  if (activeTab === 'landing') {
    return renderContent();
  }

  // Show dashboard layout for all other tabs
  return (
    <div className="min-h-screen bg-gray-50 flex">
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />
      <div className={`flex-1 transition-all duration-300 ${sidebarOpen ? 'ml-64' : 'ml-16'}`}>
        <Header />
        <main className="p-6">
          {renderContent()}
        </main>
      </div>
    </div>
  );
}

export default App;