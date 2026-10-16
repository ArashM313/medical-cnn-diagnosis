import React, { useState, useRef, useEffect } from 'react';
import { ModalityId, PredictionResult, SampleScan } from '../types';
import { MODALITIES, SAMPLE_SCANS } from '../data/modalityData';
import { runInference, generateGradCamOverlay } from '../utils/inferenceEngine';
import { ProbabilityBarChart } from './ProbabilityBarChart';
import {
  Upload,
  Play,
  CheckCircle2,
  AlertTriangle,
  FileImage,
  Sliders,
  Sparkles,
  RefreshCw,
} from 'lucide-react';

export const DiagnoseTab: React.FC = () => {
  const [selectedModality, setSelectedModality] = useState<ModalityId>('xray');
  const [uploadedImageSrc, setUploadedImageSrc] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [heatmapAlpha, setHeatmapAlpha] = useState<number>(0.45);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const activeModalityConfig = MODALITIES[selectedModality];

  // Set default sample image for initial instant test drive
  useEffect(() => {
    // If no image is uploaded yet, load the default sample for this modality
    const defaultSample = SAMPLE_SCANS.find((s) => s.modality === selectedModality);
    if (defaultSample && !uploadedImageSrc) {
      setUploadedImageSrc(defaultSample.imageUrl);
      setUploadedFileName(defaultSample.name);
    }
  }, [selectedModality]);

  // When alpha slider changes and we have an existing prediction and image, update overlay live
  const handleAlphaChange = (newAlpha: number) => {
    setHeatmapAlpha(newAlpha);
    if (prediction?.camGrid && uploadedImageSrc) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const newOverlay = generateGradCamOverlay(img, prediction.camGrid!, newAlpha, 256);
        setPrediction((prev) => (prev ? { ...prev, gradCamDataUrl: newOverlay } : null));
      };
      img.src = uploadedImageSrc;
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const result = event.target?.result as string;
      setUploadedImageSrc(result);
      setUploadedFileName(file.name);
      setPrediction(null); // Reset until "Run Diagnosis" is clicked
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (PNG, JPG, DICOM preview).');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const result = event.target?.result as string;
      setUploadedImageSrc(result);
      setUploadedFileName(file.name);
      setPrediction(null);
    };
    reader.readAsDataURL(file);
  };

  const handleSelectSample = (sample: SampleScan) => {
    setSelectedModality(sample.modality);
    setUploadedImageSrc(sample.imageUrl);
    setUploadedFileName(sample.name);
    setPrediction(null);
  };

  const handleRunDiagnosis = async () => {
    if (!uploadedImageSrc) return;

    setIsLoading(true);
    try {
      // Simulate authentic model forward pass & Grad-CAM backward hook pass
      const result = await runInference(uploadedImageSrc, selectedModality, heatmapAlpha);
      setPrediction(result);
    } catch (err) {
      console.error('Diagnosis failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* ================= LEFT COLUMN: Controls & Upload (width 4 in Dash layout) ================= */}
      <div className="lg:col-span-4 space-y-5">
        {/* Step 1: Select Scan Type */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3">
          <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider">
            1. Select Scan Type
          </label>
          <div className="space-y-1.5">
            {(Object.keys(MODALITIES) as ModalityId[]).map((id) => {
              const item = MODALITIES[id];
              const isSelected = selectedModality === id;
              return (
                <button
                  key={id}
                  onClick={() => {
                    setSelectedModality(id);
                    setPrediction(null);
                    // Update sample to matching modality if current image was a sample
                    const sample = SAMPLE_SCANS.find((s) => s.modality === id);
                    if (sample) {
                      setUploadedImageSrc(sample.imageUrl);
                      setUploadedFileName(sample.name);
                    }
                  }}
                  className={`w-full text-left p-3 rounded-lg border transition-all flex items-center justify-between ${
                    isSelected
                      ? 'border-blue-600 bg-blue-50/70 text-blue-900 shadow-xs ring-1 ring-blue-500'
                      : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50 text-slate-700'
                  }`}
                >
                  <div>
                    <div className="font-semibold text-sm">{item.name}</div>
                    <div className="text-xs text-slate-500">{item.targetOrgan}</div>
                  </div>
                  <span
                    className={`text-xs px-2 py-0.5 rounded font-mono font-medium ${
                      isSelected ? 'bg-blue-200/80 text-blue-900' : 'bg-slate-100 text-slate-600'
                    }`}
                  >
                    {item.classes.length} classes
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Step 2: Upload Image */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3">
          <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider">
            2. Upload Scan Image
          </label>

          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all ${
              isDragOver
                ? 'border-blue-500 bg-blue-50/60'
                : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50/80'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="flex flex-col items-center justify-center gap-2">
              <div className="p-2.5 bg-blue-50 text-blue-600 rounded-full">
                <Upload className="w-5 h-5" />
              </div>
              <p className="text-sm font-medium text-slate-700">
                Drag and drop or <span className="text-blue-600 hover:underline">select a scan image</span>
              </p>
              <p className="text-xs text-slate-400">
                Supports DICOM exports, PNG, JPG (Auto-scaled to {activeModalityConfig.inputSize}x{activeModalityConfig.inputSize})
              </p>
            </div>
          </div>

          {uploadedFileName && (
            <div className="flex items-center justify-between p-2.5 bg-slate-50 rounded-lg border border-slate-200 text-xs">
              <div className="flex items-center gap-2 truncate pr-2">
                <FileImage className="w-4 h-4 text-blue-600 shrink-0" />
                <span className="truncate font-medium text-slate-700">{uploadedFileName}</span>
              </div>
              <span className="text-[10px] font-semibold text-emerald-700 bg-emerald-100 px-1.5 py-0.5 rounded shrink-0">
                Loaded
              </span>
            </div>
          )}

          {/* Quick preset scans */}
          <div className="pt-2">
            <span className="block text-[11px] font-medium text-slate-400 uppercase tracking-wider mb-1.5">
              Or quick-select verified clinical reference case:
            </span>
            <div className="grid grid-cols-1 gap-1.5">
              {SAMPLE_SCANS.map((scan) => (
                <button
                  key={scan.id}
                  onClick={() => handleSelectSample(scan)}
                  className={`text-left text-xs p-2 rounded-lg border flex items-center justify-between transition-colors ${
                    uploadedFileName === scan.name
                      ? 'border-blue-500 bg-blue-50 text-blue-900 font-medium'
                      : 'border-slate-200 hover:bg-slate-50 text-slate-600'
                  }`}
                >
                  <span className="truncate">{scan.name}</span>
                  <span className="text-[10px] text-slate-400 uppercase font-mono">{scan.modality}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Heatmap Alpha control */}
          <div className="pt-3 border-t border-slate-100">
            <div className="flex justify-between items-center text-xs text-slate-600 mb-1.5">
              <span className="font-medium flex items-center gap-1">
                <Sliders className="w-3.5 h-3.5 text-slate-400" /> Grad-CAM Heatmap Blend
              </span>
              <span className="font-mono font-semibold text-blue-700">
                {Math.round(heatmapAlpha * 100)}%
              </span>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={heatmapAlpha}
              onChange={(e) => handleAlphaChange(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            />
            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
              <span>More Scan (10%)</span>
              <span>Balanced (45%)</span>
              <span>More Heat (90%)</span>
            </div>
          </div>

          {/* Run Diagnosis Button */}
          <button
            onClick={handleRunDiagnosis}
            disabled={!uploadedImageSrc || isLoading}
            className={`w-full py-3 px-4 rounded-xl font-semibold text-sm shadow-sm flex items-center justify-center gap-2 transition-all ${
              !uploadedImageSrc || isLoading
                ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-500/20 hover:shadow-md active:scale-[0.99]'
            }`}
          >
            {isLoading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Running CNN Inference & Grad-CAM...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Run Diagnosis</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* ================= RIGHT COLUMN: Images + Results (width 8 in Dash layout) ================= */}
      <div className="lg:col-span-8 space-y-6">
        {/* Images Display Row: Uploaded Scan + Grad-CAM Explanation */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            {/* 1. Uploaded Scan */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-800 flex items-center gap-1.5">
                  <span>Uploaded Scan</span>
                </h3>
                <span className="text-xs text-slate-400 font-mono">
                  {activeModalityConfig.inputSize}x{activeModalityConfig.inputSize} Input
                </span>
              </div>

              <div className="relative aspect-square w-full bg-slate-900 rounded-xl overflow-hidden border border-slate-200 flex items-center justify-center group shadow-inner">
                {uploadedImageSrc ? (
                  <img
                    src={uploadedImageSrc}
                    alt="Uploaded medical scan"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-center p-6 text-slate-400">
                    <FileImage className="w-10 h-10 mx-auto mb-2 opacity-50" />
                    <p className="text-xs">No scan image loaded yet</p>
                  </div>
                )}
              </div>
            </div>

            {/* 2. Grad-CAM Explanation */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-800 flex items-center gap-1.5">
                  <span>Grad-CAM Explanation</span>
                  <span className="text-[10px] bg-indigo-100 text-indigo-700 px-1.5 py-0.2 rounded font-semibold">
                    features_final
                  </span>
                </h3>
                <span className="text-xs text-slate-400 font-mono">
                  JET Colormap &alpha;={heatmapAlpha.toFixed(2)}
                </span>
              </div>

              <div className="relative aspect-square w-full bg-slate-900 rounded-xl overflow-hidden border border-slate-200 flex items-center justify-center shadow-inner">
                {isLoading ? (
                  <div className="flex flex-col items-center justify-center p-6 text-blue-300 space-y-2">
                    <RefreshCw className="w-8 h-8 animate-spin text-blue-400" />
                    <p className="text-xs font-medium text-slate-200">
                      Computing layer gradients & class activations...
                    </p>
                  </div>
                ) : prediction ? (
                  <img
                    src={prediction.gradCamDataUrl}
                    alt="Grad-CAM visualization"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-center p-6 text-slate-400">
                    <Sparkles className="w-10 h-10 mx-auto mb-2 opacity-40 text-blue-400" />
                    <p className="text-xs">
                      Click <strong>"Run Diagnosis"</strong> to generate visual explanation
                    </p>
                  </div>
                )}
              </div>

              {/* JET Colormap bar legend */}
              <div className="pt-1 flex items-center justify-between text-[11px] text-slate-500">
                <span>Low activation</span>
                <div
                  className="w-32 sm:w-44 h-2.5 rounded-full shadow-xs mx-2"
                  style={{
                    background:
                      'linear-gradient(to right, #000080, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000, #800000)',
                  }}
                />
                <span>High activation</span>
              </div>
            </div>
          </div>
        </div>

        {/* Prediction Result Alert Card (matching callbacks.py dbc_alert) */}
        {prediction ? (
          <div
            className={`border rounded-xl p-5 shadow-xs transition-all ${
              prediction.confidence >= 0.7
                ? 'bg-emerald-50/80 border-emerald-300 text-emerald-950'
                : 'bg-amber-50/80 border-amber-300 text-amber-950'
            }`}
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-start gap-3">
                <div
                  className={`p-2 rounded-lg shrink-0 mt-0.5 ${
                    prediction.confidence >= 0.7
                      ? 'bg-emerald-600 text-white'
                      : 'bg-amber-600 text-white'
                  }`}
                >
                  {prediction.confidence >= 0.7 ? (
                    <CheckCircle2 className="w-6 h-6" />
                  ) : (
                    <AlertTriangle className="w-6 h-6" />
                  )}
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-xl font-bold tracking-tight">
                      Prediction: {prediction.predictedClass}
                    </h3>
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full font-bold uppercase ${
                        prediction.confidence >= 0.7
                          ? 'bg-emerald-200 text-emerald-800'
                          : 'bg-amber-200 text-amber-800'
                      }`}
                    >
                      {activeModalityConfig.classLabels[prediction.predictedClass] ||
                        prediction.predictedClass}
                    </span>
                  </div>
                  <p className="text-sm font-medium mt-0.5 opacity-90">
                    Confidence: {(prediction.confidence * 100).toFixed(1)}% &nbsp;|&nbsp; Inference time:{' '}
                    {prediction.inferenceTimeMs}ms
                  </p>
                </div>
              </div>

              {prediction.thresholdApplied !== undefined && (
                <div className="bg-white/80 backdrop-blur-xs px-3 py-1.5 rounded-lg border border-slate-200 text-xs self-start sm:self-center">
                  <span className="text-slate-500">Decision Cutoff: </span>
                  <span className="font-mono font-bold text-slate-800">
                    {prediction.thresholdApplied}
                  </span>
                </div>
              )}
            </div>

            <p className="mt-3 text-xs opacity-85 border-t border-emerald-200/50 pt-2.5">
              {prediction.explanationNotes}
            </p>
          </div>
        ) : (
          <div className="bg-slate-100/70 border border-slate-200/80 rounded-xl p-4 text-center text-sm text-slate-500">
            Awaiting diagnosis run. Select a scan and click <strong>"Run Diagnosis"</strong> to see model
            prediction and class probabilities.
          </div>
        )}

        {/* Probability Bar Chart */}
        {prediction && (
          <ProbabilityBarChart
            probabilities={prediction.probabilities}
            predictedClass={prediction.predictedClass}
            thresholdApplied={prediction.thresholdApplied}
          />
        )}
      </div>
    </div>
  );
};
