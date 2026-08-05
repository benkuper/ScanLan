export type PhaseStatus = 'ready' | 'capturing' | 'complete' | 'failed';
export type ProcessingStatus = 'idle' | 'processing' | 'complete' | 'failed';
export type SensorKind = 'kinect_v2' | 'azure_kinect' | 'femto_mega';
export type SensorConnection = 'usb' | 'network';
export type DepthFieldOfView = 'narrow' | 'wide';
export type MeshViewMode = 'surface' | 'surface-wireframe' | 'wireframe' | 'shaded';
export type LiveReconstructionMode = 'points' | 'mesh';

export interface CaptureSettings {
  captureFps: number;
  maxDepthM: number;
  voxelSizeMm: number;
  sensorKind: SensorKind;
  sensorId: string;
  sensorConnection: SensorConnection;
  sensorAddress: string;
  useImu: boolean;
  depthFieldOfView: DepthFieldOfView;
  depthBinned: boolean;
  rgbJpegQuality: number;
  /** Zero keeps the sensor's native RGB dimensions. */
  maxRgbDimension: number;
  liveReconstruction: LiveReconstructionMode;
}

export type ArtifactStatus = 'ready' | 'building' | 'stale' | 'failed';

export interface ArtifactSummary {
  path: string;
  status: ArtifactStatus;
  sourceFingerprint: string;
  updatedAt: string;
  metric: boolean;
  stale: boolean;
}

export interface ArtifactCatalog {
  pointCloud: ArtifactSummary | null;
  texturedMesh: ArtifactSummary | null;
  gaussianSplat: ArtifactSummary | null;
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
  schemaVersion: number;
  id: string;
  name: string;
  path: string;
  createdAt: string;
  phases: PhaseSummary[];
  artifacts: ArtifactCatalog;
  activeJob: string | null;
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
  sensorCapabilities: SensorKind[];
  sensorWorkerAvailable: boolean;
  sensorStatus: string;
  reconstructionWorkerAvailable: boolean;
  splatWorkerAvailable: boolean;
  splatStatus: string;
}

export type ArtifactTarget = 'pointCloud' | 'texturedMesh' | 'gaussianSplat';
export type ArtifactJobStatus = 'queued' | 'running' | 'cancelling' | 'cancelled' | 'failed' | 'complete';

export interface ArtifactJob {
  id: string;
  projectPath: string;
  targets: ArtifactTarget[];
  stage: string;
  detail: string;
  progress: number;
  iteration: number | null;
  totalIterations: number | null;
  loss: number | null;
  etaSeconds: number | null;
  stageProgress: number | null;
  stageEtaSeconds: number | null;
  elapsedSeconds: number | null;
  computeBackend: string | null;
  status: ArtifactJobStatus;
  createdAt: string;
  startedAt: string | null;
  updatedAt: string;
  sourceFingerprint: string;
  logPath: string;
  error: string | null;
  resumable: boolean;
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
  liveReconstructionActive: boolean;
  liveReconstructionMode: LiveReconstructionMode;
  liveProcessedFrameCount: number;
  liveIntegratedFrameCount: number;
  liveRejectedFrameCount: number;
  liveTriangleCount: number;
  trackingFps: number;
  sourceDropCount: number;
  trackingQueueDropCount: number;
  mappingDropCount: number;
  trackingOverlap: number;
  depthRmseMm?: number;
  liveReconstructionBackend?: string;
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
  uvs?: Float32Array;
  colors?: Uint8Array;
  indices: Uint32Array;
  texture?: Uint8Array;
}
