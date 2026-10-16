import React from 'react';
import { Activity, Stethoscope, BarChart3, ShieldCheck } from 'lucide-react';

interface HeaderProps {
  activeTab: 'diagnose' | 'metrics';
  setActiveTab: (tab: 'diagnose' | 'metrics') => void;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab }) => {
  return (
    <header className="bg-white border-b border-slate-200 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="p-2.5 bg-blue-600 rounded-lg text-white shadow-xs shrink-0 mt-0.5">
              <Stethoscope className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
                  Medical Image Disease Classifier
                </h1>
                <span className="hidden sm:inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                  <ShieldCheck className="w-3.5 h-3.5" /> PyTorch CNN & Grad-CAM
                </span>
              </div>
              <p className="text-sm text-slate-500 mt-0.5">
                Upload an X-ray, CT, or MRI scan for CNN-based classification with Grad-CAM visual explanation.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 bg-slate-100 p-1 rounded-lg self-start md:self-center">
            <button
              onClick={() => setActiveTab('diagnose')}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all ${
                activeTab === 'diagnose'
                  ? 'bg-white text-blue-700 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <Activity className="w-4 h-4" />
              <span>Diagnose</span>
            </button>
            <button
              onClick={() => setActiveTab('metrics')}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all ${
                activeTab === 'metrics'
                  ? 'bg-white text-blue-700 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              <span>Model Performance</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
