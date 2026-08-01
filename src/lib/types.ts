export type EnvironmentProfile = 'indoor' | 'outdoor_low_light';
export type PhaseStatus = 'ready' | 'capturing' | 'complete' | 'failed';
export type ProcessingStatus = 'idle' | 'processing' | 'complete' | 'failed';
export type SensorKind = 'kinect_v2' | 'azure_kinect' | 'femto_mega';
export type SensorConnection = 'usb' | 'network';
export type DepthFieldOfView = 'narrow' | 'wide';

export interface CaptureSettings {
  captureFps: number;
  maxDepthM: number;
  voxelSizeMm: number;
  environment: EnvironmentProfile;
  sensorKind: SensorKind;
  sensorId: string;
  sensorConnection: SensorConnection;
  sensorAddress: string;
  useImu: boolean;
  depthFieldOfView: DepthFieldOfView;
  depthBinned: boolean;
}

export interface AvailableSensor {
  id: string;
  kind: SensorKind;
  name: string;
  connection: SensorConnection;
  address: string;
  serial: string;
  supportsImu: boolean;
}

export interface PhaseSummary {
  id: string;
  name: string;
  createdAt: string;
  durationSeconds: number;
  frameCount: number;
  status: PhaseStatus;
  overlapHint: string;
}

export interface ProjectSummary {
  id: string;
  name: string;
  path: string;
  createdAt: string;
  phases: PhaseSummary[];
  settings: CaptureSettings;
  processingStatus: ProcessingStatus;
  processingError?: string;
  pointCount?: number;
  outputPath?: string;
  meshTriangleCount?: number;
  meshOutputPath?: string;
  cameraFrameCount?: number;
  confidenceScore?: number;
  confidenceLabel?: string;
  confidenceDetail?: string;
  framesUsed?: number;
  processingBackend?: string;
  processingDurationSeconds?: number;
}

export interface RuntimeInfo {
  platform: string;
  sensorWorkerAvailable: boolean;
  sensorConnected: boolean;
  sensorStatus: string;
  reconstructionWorkerAvailable: boolean;
}

export interface CaptureStatus {
  project: ProjectSummary;
  preview: PreviewPoint[];
  capturing: boolean;
  sensorConnected: boolean;
  sensorPaused: boolean;
  sensorStatus: string;
  sensorName: string;
  frameCount: number;
  totalFrameCount: number;
  previewPointCount: number;
  streamFps: number;
  tracking: boolean;
  trackingStatus: string;
  imuActive: boolean;
  imuRateHz: number;
  reconstruction?: ReconstructionProgress;
  error?: string;
}

export interface ReconstructionProgress {
  stage: string;
  detail: string;
  progress: number;
  processedUnits: number;
  totalUnits: number;
  etaSeconds?: number;
  pointCount?: number;
  stageProgress?: number;
  stageEtaSeconds?: number;
  elapsedSeconds?: number;
  computeBackend?: string;
  stageTimingsSeconds?: Record<string, number>;
}

export interface CloudTransform {
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];
}

export interface PreviewPoint {
  position: [number, number, number];
  color: [number, number, number];
}

export interface PackedPreviewFrame {
  frameCount: number;
  pointCount: number;
  positions: Float32Array;
  colors: Uint8Array;
}

export interface PreviewMesh {
  positions: Float32Array;
  uvs: Float32Array;
  indices: Uint32Array;
  texture: Uint8Array;
}

export interface CameraFrame {
  phaseName: string;
  phaseId: string;
  frameIndex: number;
  timestampUs: number;
  /** Row-major camera-to-viewer transform. */
  matrix: [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];
  aspect: number;
  fovYDegrees: number;
  imageYUp: boolean;
  textureFrame: boolean;
}
