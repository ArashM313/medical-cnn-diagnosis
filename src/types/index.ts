export type ModalityId = 'xray' | 'ct' | 'mri';

export interface ModalityConfig {
  id: ModalityId;
  name: string;
  shortName: string;
  description: string;
  targetOrgan: string;
  classes: string[];
  classLabels: Record<string, string>;
  classDescriptions: Record<string, string>;
  modelArchitecture: string;
  backboneDetails: string;
  decisionThreshold: number;
  inputSize: number;
  defaultSampleUrl: string;
}

export interface PredictionResult {
  predictedClass: string;
  predictedIndex: number;
  confidence: number;
  thresholdApplied?: number;
  probabilities: { className: string; probability: number; percentage: number }[];
  gradCamDataUrl: string;
  camGrid?: number[][]; // Raw normalized activation map (16x16 or 4x4)
  explanationNotes: string;
  inferenceTimeMs: number;
}

export interface PerClassMetric {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ModalityMetrics {
  modality: ModalityId;
  classes: string[];
  test_accuracy: number;
  macro_f1: number;
  roc_auc_ovr: number | null;
  decision_threshold: number;
  confusion_matrix: number[][];
  per_class: Record<string, PerClassMetric>;
  checkpoint_val_acc: number;
  checkpoint_epoch: number;
  test_set_size: number;
  architectureDetails: {
    type: string;
    layers: string;
    inputResolution: string;
    lossFunction: string;
    learningRate: string;
    batchSize: number;
    trainingEpochs: number;
    patience: number;
  };
}

export interface SampleScan {
  id: string;
  name: string;
  modality: ModalityId;
  imageUrl: string;
  groundTruth: string;
  clinicalNotes: string;
}
