import { ModalityConfig, ModalityId, ModalityMetrics, SampleScan } from '../types';

export const MODALITIES: Record<ModalityId, ModalityConfig> = {
  xray: {
    id: 'xray',
    name: 'Chest X-Ray',
    shortName: 'X-Ray',
    description: 'Pediatric and adult chest radiographs screened for bacterial & viral pneumonia.',
    targetOrgan: 'Lungs & Thoracic Cavity',
    classes: ['NORMAL', 'PNEUMONIA'],
    classLabels: {
      NORMAL: 'Normal Lung Parenchyma',
      PNEUMONIA: 'Pneumonia (Consolidation/Infiltrate)'
    },
    classDescriptions: {
      NORMAL: 'Clear lung fields without abnormal focal opacification or effusion.',
      PNEUMONIA: 'Focal or diffuse alveolar opacities, air bronchograms, or lobar consolidation.'
    },
    modelArchitecture: 'MedicalCNN (Custom 4-Block ConvNet)',
    backboneDetails: '4 Conv2d (32->64->128->256) + BatchNorm2d + MaxPool + Dropout(0.25, 0.5) + Linear(512) -> 2',
    decisionThreshold: 0.53, // Calibrated threshold from training/recalibrate.py
    inputSize: 128,
    defaultSampleUrl: '/samples/gradcam_test_xray.png',
  },
  ct: {
    id: 'ct',
    name: 'CT Scan (Chest/Lung)',
    shortName: 'CT Scan',
    description: 'Thoracic computed tomography slices for lung adenocarcinoma, large-cell carcinoma, squamous-cell carcinoma, or normal tissue.',
    targetOrgan: 'Pulmonary Parenchyma & Mediastinum',
    classes: ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma'],
    classLabels: {
      adenocarcinoma: 'Adenocarcinoma',
      'large.cell.carcinoma': 'Large Cell Carcinoma',
      normal: 'Normal Parenchyma',
      'squamous.cell.carcinoma': 'Squamous Cell Carcinoma'
    },
    classDescriptions: {
      adenocarcinoma: 'Glandular malignancy typically found in outer/peripheral lung zones.',
      'large.cell.carcinoma': 'Undifferentiated malignant epithelial tumor with rapid necrotic expansion.',
      normal: 'Homogeneous attenuation of lung lobes without focal mass or spiculation.',
      'squamous.cell.carcinoma': 'Central bronchial lumen cavitating lesion or hilar mass.'
    },
    modelArchitecture: 'TransferResNetCNN (ResNet18 Fine-tuned)',
    backboneDetails: 'Pretrained ResNet18 backbone (layer1-3 frozen, layer4 fine-tuned at lr=1e-4) with replaced FC head -> 4',
    decisionThreshold: 0.50,
    inputSize: 128,
    defaultSampleUrl: '/samples/gradcam_test_ct.png',
  },
  mri: {
    id: 'mri',
    name: 'Brain MRI',
    shortName: 'Brain MRI',
    description: 'Multi-sequence axial cranial magnetic resonance images for intracranial neoplasm differential.',
    targetOrgan: 'Intracranial & Sellar Region',
    classes: ['glioma', 'meningioma', 'notumor', 'pituitary'],
    classLabels: {
      glioma: 'Glioma',
      meningioma: 'Meningioma',
      notumor: 'No Tumor',
      pituitary: 'Pituitary Adenoma'
    },
    classDescriptions: {
      glioma: 'Intra-axial glial origin infiltrating lesion with vasogenic edema.',
      meningioma: 'Extra-axial dural-based mass often showing dural tail enhancement.',
      notumor: 'Symmetrical cerebral hemispheres, preserved ventricles, absence of mass effect.',
      pituitary: 'Sellar/suprasellar mass with expansion into the optochiasmatic cistern.'
    },
    modelArchitecture: 'MedicalCNN (Custom 4-Block ConvNet)',
    backboneDetails: '4 Conv2d (32->64->128->256) + BatchNorm2d + MaxPool + Dropout + Linear(512) -> 4',
    decisionThreshold: 0.50,
    inputSize: 128,
    defaultSampleUrl: '/samples/gradcam_test_mri.png',
  },
};

export const METRICS_DATA: Record<ModalityId, ModalityMetrics> = {
  xray: {
    modality: 'xray',
    classes: ['NORMAL', 'PNEUMONIA'],
    test_accuracy: 0.7644230769230769,
    macro_f1: 0.692986360973977,
    roc_auc_ovr: 0.9301720359412669,
    decision_threshold: 0.53,
    confusion_matrix: [
      [88, 146],
      [1, 389],
    ],
    per_class: {
      NORMAL: {
        precision: 0.9887640449438202,
        recall: 0.37606837606837606,
        f1: 0.544891640866873,
        support: 234,
      },
      PNEUMONIA: {
        precision: 0.7271028037383177,
        recall: 0.9974358974358974,
        f1: 0.841081081081081,
        support: 390,
      },
    },
    checkpoint_val_acc: 0.9783163265306123,
    checkpoint_epoch: 20,
    test_set_size: 624,
    architectureDetails: {
      type: 'MedicalCNN (From Scratch)',
      layers: 'Conv(32)->Conv(64)->Conv(128)->Conv(256) + BatchNorm + MaxPool + Dropout',
      inputResolution: '128 x 128 (Grayscale/RGB)',
      lossFunction: 'CrossEntropyLoss (Weighted: 2.89x for class imbalance)',
      learningRate: '1e-3 (Adam optimizer with ReduceLROnPlateau)',
      batchSize: 32,
      trainingEpochs: 30,
      patience: 8,
    },
  },
  ct: {
    modality: 'ct',
    classes: ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma'],
    test_accuracy: 0.7111111111111111,
    macro_f1: 0.7184546782252357,
    roc_auc_ovr: 0.9359024874094177,
    decision_threshold: 0.50,
    confusion_matrix: [
      [64, 9, 21, 26],
      [1, 40, 6, 4],
      [2, 1, 51, 0],
      [4, 1, 16, 69],
    ],
    per_class: {
      adenocarcinoma: {
        precision: 0.9014084507042254,
        recall: 0.5333333333333333,
        f1: 0.6701570680628273,
        support: 120,
      },
      'large.cell.carcinoma': {
        precision: 0.7843137254901961,
        recall: 0.7843137254901961,
        f1: 0.7843137254901961,
        support: 51,
      },
      normal: {
        precision: 0.5425531914893617,
        recall: 0.9444444444444444,
        f1: 0.6891891891891891,
        support: 54,
      },
      'squamous.cell.carcinoma': {
        precision: 0.696969696969697,
        recall: 0.7666666666666667,
        f1: 0.7301587301587301,
        support: 90,
      },
    },
    checkpoint_val_acc: 0.8194444444444444,
    checkpoint_epoch: 18,
    test_set_size: 315,
    architectureDetails: {
      type: 'TransferResNetCNN (ResNet18 Backbone)',
      layers: 'ResNet18 (Conv7x7 + 4 Residual Blocks, layer4 fine-tuned) + Linear Head',
      inputResolution: '128 x 128 (3 Channels)',
      lossFunction: 'CrossEntropyLoss',
      learningRate: '1e-4 backbone / 1e-3 classifier head',
      batchSize: 32,
      trainingEpochs: 40,
      patience: 14,
    },
  },
  mri: {
    modality: 'mri',
    classes: ['glioma', 'meningioma', 'notumor', 'pituitary'],
    test_accuracy: 0.9275,
    macro_f1: 0.9250741785740485,
    roc_auc_ovr: 0.9758718749999999,
    decision_threshold: 0.50,
    confusion_matrix: [
      [304, 46, 39, 11],
      [0, 380, 8, 12],
      [0, 0, 400, 0],
      [0, 0, 0, 400],
    ],
    per_class: {
      glioma: {
        precision: 1.0,
        recall: 0.76,
        f1: 0.8636363636363636,
        support: 400,
      },
      meningioma: {
        precision: 0.892018779342723,
        recall: 0.95,
        f1: 0.9200968523002422,
        support: 400,
      },
      notumor: {
        precision: 0.8948545861297539,
        recall: 1.0,
        f1: 0.9445100354191264,
        support: 400,
      },
      pituitary: {
        precision: 0.9456264775413712,
        recall: 1.0,
        f1: 0.9720534629404617,
        support: 400,
      },
    },
    checkpoint_val_acc: 0.9809523809523809,
    checkpoint_epoch: 30,
    test_set_size: 1600,
    architectureDetails: {
      type: 'MedicalCNN (From Scratch)',
      layers: 'Conv(32)->Conv(64)->Conv(128)->Conv(256) + BatchNorm + MaxPool + Dropout',
      inputResolution: '128 x 128 (3 Channels)',
      lossFunction: 'CrossEntropyLoss',
      learningRate: '1e-3 (Adam optimizer)',
      batchSize: 32,
      trainingEpochs: 30,
      patience: 8,
    },
  },
};

export const SAMPLE_SCANS: SampleScan[] = [
  {
    id: 'sample-xray-pneumonia',
    name: 'Chest X-Ray: Pneumonia Infiltrate',
    modality: 'xray',
    imageUrl: '/samples/gradcam_test_xray.png',
    groundTruth: 'PNEUMONIA',
    clinicalNotes: 'Diffuse right and bibasilar alveolar consolidation with loss of diaphragmatic silhouette.',
  },
  {
    id: 'sample-ct-carcinoma',
    name: 'Chest CT: Lung Adenocarcinoma',
    modality: 'ct',
    imageUrl: '/samples/gradcam_test_ct.png',
    groundTruth: 'adenocarcinoma',
    clinicalNotes: 'Peripheral right lung nodule with irregular spiculated margins and soft-tissue attenuation.',
  },
  {
    id: 'sample-mri-tumor',
    name: 'Brain MRI: Pituitary / Glioma Case',
    modality: 'mri',
    imageUrl: '/samples/gradcam_test_mri.png',
    groundTruth: 'pituitary',
    clinicalNotes: 'T1/T2 axial section demonstrating prominent focal enhancement in intracranial target region.',
  },
];
