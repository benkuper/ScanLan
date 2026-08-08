export type PhaseStatus = 'ready' | 'capturing' | 'complete' | 'failed';
export type ProcessingStatus = 'idle' | 'processing' | 'complete' | 'failed';
export type SensorKind = 'kinect_v2' | 'azure_kinect' | 'femto_mega';
export type SensorConnection = 'usb' | 'network';
export type DepthFieldOfView = 'narrow' | 'wide';
export type RgbResolution = 'auto' | '720p' | '1080p' | '1440p' | '1536p' | '2160p' | '3072p';
export type MeshViewMode = 'surface' | 'surface-wireframe' | 'wireframe' | 'shaded';
export type LiveReconstructionMode = 'points' | 'mesh';
export type MeshRepairProfile = 'faithful' | 'architectural' | 'natural';

export interface CaptureSettings {
  captureFps: number;
  /** Zero selects the depth profile's fastest supported sensor rate. */
  sensorFps: number;
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
  rgbResolution: RgbResolution;
  rgbAutoExposure: boolean;
  /** Manual RGB exposure in microseconds. */
  rgbExposureUs: number;
  rgbGain: number;
  rgbAutoWhiteBalance: boolean;
  rgbWhiteBalanceK: number;
  rgbColorAdjustmentsEnabled: boolean;
  rgbBrightness: number;
  rgbContrast: number;
  rgbSaturation: number;
  rgbSharpness: number;
  rgbBacklightCompensation: boolean;
  /** Zero restores the camera's default anti-flicker setting. */
  rgbPowerlineHz: number;
  /** Zero selects the Femto Mega's default IMU profile. */
  imuAccelRateHz: number;
  imuAccelRangeG: number;
  imuGyroRateHz: number;
  imuGyroRangeDps: number;
  liveReconstruction: LiveReconstructionMode;
  repairMesh: boolean;
  meshRepairProfile: MeshRepairProfile;
  fillInferredMeshHoles: boolean;
  produceWatertightMesh: boolean;
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
  connected: boolean;
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

export interface MediaSourceSummary {
  id: string;
  name: string;
  path: string;
  kind: 'photo' | 'video';
  byteSize: number;
  createdAt: string;
}

export interface ProjectSummary {
  schemaVersion: number;
  id: string;
  name: string;
  path: string;
  createdAt: string;
  phases: PhaseSummary[];
  mediaSources: MediaSourceSummary[];
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
  meshRepairProfile?: MeshRepairProfile;
  meshRepairStatus?: string;
  meshRepairReportPath?: string;
  meshRepairFallback?: boolean;
  meshRepairDefectsFixed?: number;
  meshRepairHolesFilled?: number;
  meshRepairOpeningsPreserved?: number;
  meshRepairUnknownPreserved?: number;
  watertightMeshOutputPath?: string;
}

export interface ProjectCatalogEntry {
  id: string;
  name: string;
  path: string;
  createdAt: string;
  modifiedAt: string;
  captureCount: number;
  mediaSourceCount: number;
  frameCount: number;
  artifactCount: number;
  processingStatus: ProcessingStatus;
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
  sourceKind: 'rgbd' | 'media';
  stage: string;
  detail: string;
  progress: number;
  iteration: number | null;
  totalIterations: number | null;
  loss: number | null;
  smoothedLoss: number | null;
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
  previewing: boolean;
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

export interface CloudTransform {
  position: [number, number, number];
  /** XYZ Euler rotation in degrees. */
  rotation: [number, number, number];
  scale: [number, number, number];
}

/** Axis-aligned export bounds, expressed after the model edit pose. */
export interface BoundingBoxClip {
  min: [number, number, number];
  max: [number, number, number];
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
