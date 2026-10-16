import React, { useState } from 'react';
import { ModalityId } from '../types';
import { METRICS_DATA, MODALITIES } from '../data/modalityData';
import { ConfusionMatrix } from './ConfusionMatrix';
import { Layers, Cpu, Info } from 'lucide-react';

export const ModelPerformanceTab: React.FC = () => {
  const [selectedModality, setSelectedModality] = useState<ModalityId>('xray');

  const modality = MODALITIES[selectedModality];
  const metrics = METRICS_DATA[selectedModality];

  return (
    <div className="space-y-6">
      {/* Top selector */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <label
            htmlFor="metrics-modality"
            className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1"
          >
            Select model to inspect
          </label>
          <div className="flex flex-wrap gap-2">
            {(Object.keys(MODALITIES) as ModalityId[]).map((id) => {
              const item = MODALITIES[id];
              const isSelected = selectedModality === id;
              return (
                <button
                  key={id}
                  onClick={() => setSelectedModality(id)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    isSelected
                      ? 'bg-blue-600 text-white shadow-xs font-semibold'
                      : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  {item.name}
                </button>
              );
            })}
          </div>
        </div>

        <div className="sm:text-right border-t sm:border-t-0 pt-3 sm:pt-0 border-slate-100">
          <div className="text-xs text-slate-500">Evaluated Architecture</div>
          <div className="text-sm font-semibold text-slate-800 flex sm:justify-end items-center gap-1.5 mt-0.5">
            <Cpu className="w-4 h-4 text-blue-600" />
            <span>{metrics.architectureDetails.type}</span>
          </div>
        </div>
      </div>

      {/* Summary KPI Cards (matching Dash dbc_alert summary) */}
      <div className="bg-gradient-to-r from-blue-50/70 via-slate-50 to-indigo-50/70 border border-blue-200/80 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-md text-xs font-bold tracking-wide uppercase bg-blue-600 text-white">
                {selectedModality.toUpperCase()}
              </span>
              <h3 className="text-lg font-bold text-slate-900">
                Test Set Evaluation Report
              </h3>
            </div>
            <p className="text-sm text-slate-600 mt-1">
              Held-out test split of{' '}
              <strong className="text-slate-900 font-semibold">{metrics.test_set_size}</strong> images
              using weights preserved from epoch{' '}
              <strong className="text-slate-900 font-semibold">{metrics.checkpoint_epoch}</strong>.
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 lg:gap-4">
            <div className="bg-white/80 backdrop-blur-xs p-3 rounded-lg border border-slate-200">
              <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider block">
                Test Accuracy
              </span>
              <span className="text-xl font-bold font-mono text-blue-700">
                {(metrics.test_accuracy * 100).toFixed(2)}%
              </span>
            </div>

            <div className="bg-white/80 backdrop-blur-xs p-3 rounded-lg border border-slate-200">
              <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider block">
                Macro F1
              </span>
              <span className="text-xl font-bold font-mono text-slate-800">
                {metrics.macro_f1.toFixed(3)}
              </span>
            </div>

            <div className="bg-white/80 backdrop-blur-xs p-3 rounded-lg border border-slate-200">
              <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider block">
                ROC-AUC (OVR)
              </span>
              <span className="text-xl font-bold font-mono text-emerald-700">
                {metrics.roc_auc_ovr !== null ? metrics.roc_auc_ovr.toFixed(3) : 'N/A'}
              </span>
            </div>

            <div className="bg-white/80 backdrop-blur-xs p-3 rounded-lg border border-slate-200">
              <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider block">
                Val Acc (Best)
              </span>
              <span className="text-xl font-bold font-mono text-indigo-700">
                {(metrics.checkpoint_val_acc * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        {selectedModality === 'xray' && (
          <div className="mt-4 pt-3 border-t border-blue-200/60 flex items-center gap-2 text-xs text-blue-900 bg-blue-100/40 px-3 py-2 rounded-lg">
            <Info className="w-4 h-4 text-blue-600 shrink-0" />
            <span>
              <strong>Threshold Calibration Applied:</strong> Tuned via validation balanced accuracy sweep to{' '}
              <code className="bg-white px-1.5 py-0.5 rounded border border-blue-200 font-bold">0.53</code>{' '}
              (compensating for class-weighted CrossEntropy loss 2.89x penalty).
            </span>
          </div>
        )}
      </div>

      {/* Main Charts & Tables Grid (matching Dash dbc.Row) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confusion Matrix */}
        <ConfusionMatrix classes={metrics.classes} matrix={metrics.confusion_matrix} />

        {/* Per-Class Metrics Table */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="text-sm font-semibold text-slate-800">
                  Per-Class Precision / Recall / F1
                </h4>
                <p className="text-xs text-slate-500 mt-0.5">
                  Detailed classification performance metrics broken down by target pathology
                </p>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-slate-200 text-sm">
                <thead>
                  <tr className="bg-slate-50">
                    <th className="px-3 py-2.5 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                      Class
                    </th>
                    <th className="px-3 py-2.5 text-right text-xs font-semibold text-slate-600 uppercase tracking-wider">
                      Precision
                    </th>
                    <th className="px-3 py-2.5 text-right text-xs font-semibold text-slate-600 uppercase tracking-wider">
                      Recall
                    </th>
                    <th className="px-3 py-2.5 text-right text-xs font-semibold text-slate-600 uppercase tracking-wider">
                      F1
                    </th>
                    <th className="px-3 py-2.5 text-right text-xs font-semibold text-slate-600 uppercase tracking-wider">
                      Support
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 bg-white font-mono text-xs">
                  {Object.entries(metrics.per_class).map(([className, stats]) => (
                    <tr key={className} className="hover:bg-slate-50/80 transition-colors">
                      <td className="px-3 py-3 font-sans font-medium text-slate-900 whitespace-nowrap">
                        <div className="font-semibold">{className}</div>
                        <div className="text-[11px] text-slate-500 font-normal">
                          {modality.classLabels[className] || className}
                        </div>
                      </td>
                      <td className="px-3 py-3 text-right text-slate-700">
                        {stats.precision.toFixed(3)}
                      </td>
                      <td className="px-3 py-3 text-right text-slate-700">
                        {stats.recall.toFixed(3)}
                      </td>
                      <td className="px-3 py-3 text-right font-bold text-blue-700">
                        {stats.f1.toFixed(3)}
                      </td>
                      <td className="px-3 py-3 text-right text-slate-500">
                        {stats.support}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 text-xs text-slate-500 flex items-center justify-between">
            <span>Support: Number of verified test instances</span>
            <span className="font-mono">Total: {metrics.test_set_size}</span>
          </div>
        </div>
      </div>

      {/* Model Architecture & Training Hyperparameters Specs */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
        <h4 className="text-sm font-semibold text-slate-800 flex items-center gap-2 mb-4">
          <Layers className="w-4 h-4 text-blue-600" />
          <span>Training Pipeline & Architecture Specifications</span>
        </h4>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
            <span className="text-slate-400 font-medium block uppercase text-[10px]">
              Network Backbone
            </span>
            <span className="text-slate-800 font-semibold block mt-1">
              {metrics.architectureDetails.type}
            </span>
            <span className="text-slate-500 text-[11px] block mt-1">
              {metrics.architectureDetails.layers}
            </span>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
            <span className="text-slate-400 font-medium block uppercase text-[10px]">
              Input Resolution & Batch
            </span>
            <span className="text-slate-800 font-semibold block mt-1 font-mono">
              {metrics.architectureDetails.inputResolution}
            </span>
            <span className="text-slate-500 text-[11px] block mt-1 font-mono">
              Batch size: {metrics.architectureDetails.batchSize} | Epochs: {metrics.architectureDetails.trainingEpochs}
            </span>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
            <span className="text-slate-400 font-medium block uppercase text-[10px]">
              Optimization & LR
            </span>
            <span className="text-slate-800 font-semibold block mt-1 font-mono">
              {metrics.architectureDetails.learningRate}
            </span>
            <span className="text-slate-500 text-[11px] block mt-1">
              Patience: {metrics.architectureDetails.patience} epochs early stop
            </span>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
            <span className="text-slate-400 font-medium block uppercase text-[10px]">
              Loss Function
            </span>
            <span className="text-slate-800 font-semibold block mt-1">
              {metrics.architectureDetails.lossFunction}
            </span>
            <span className="text-slate-500 text-[11px] block mt-1">
              Decision Cutoff: {metrics.decision_threshold}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
