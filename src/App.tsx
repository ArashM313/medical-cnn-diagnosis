import React, { useState } from 'react';
import { Header } from './components/Header';
import { DiagnoseTab } from './components/DiagnoseTab';
import { ModelPerformanceTab } from './components/ModelPerformanceTab';
import { Info, Activity } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'diagnose' | 'metrics'>('diagnose');

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col text-slate-900 selection:bg-blue-100 selection:text-blue-900">
      {/* Top Navbar Header */}
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {/* Clinical Disclaimer Notice */}
        <div className="mb-6 p-3 bg-amber-50/80 border border-amber-200 rounded-xl flex items-start gap-2.5 text-xs text-amber-900">
          <Info className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <strong>Investigational Clinical Decision Support Tool:</strong> Designed for medical image analysis
            research and diagnostic education. Visual explanations via Grad-CAM reflect convolutional activation maps
            of the final feature layer (<code className="font-mono text-amber-950 font-semibold">model.features_final</code>).
            Not an autonomous substitute for certified radiological evaluation.
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'diagnose' ? <DiagnoseTab /> : <ModelPerformanceTab />}
      </main>

      {/* Footer */}
      <footer className="mt-auto border-t border-slate-200 bg-white py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-blue-600" />
            <span className="font-medium text-slate-700">Medical Image Disease Classifier</span>
            <span>&middot;</span>
            <span>Chest X-Ray, Chest CT, Brain MRI</span>
          </div>

          <div className="flex items-center gap-4">
            <span className="font-mono text-slate-400">PyTorch 2.x &bull; ResNet18 &bull; MedicalCNN</span>
            <span>&middot;</span>
            <span className="text-slate-400">Grad-CAM Interpretability Engine</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
