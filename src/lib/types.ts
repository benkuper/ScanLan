export type PhaseStatus = 'ready' | 'capturing' | 'complete' | 'failed';
export type ProcessingStatus = 'idle' | 'processing' | 'complete' | 'failed';
export type SensorKind = 'kinect_v2' | 'azure_kinect' | 'femto_mega';
export type SensorConnection = 'usb' | 'network';
export type DepthFieldOfView = 'narrow' | 'wide';
export type RgbResolution = 'auto' | '720p' | '1080p' | '1440p' | '1536p' | '2160p' | '3072p';
export type MeshViewMode = 'surface' | 'surface-wireframe' | 'wireframe' | 'shaded';
export type LiveReconstructionMode = 'points' | 'mesh';
export type LiveOverlayMode = 'normal' | 'coverage' | 'tracking' | 'confidence';
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
  /** Hard budget for the active sparse live submap. */
  liveMapMemoryMib: number;
  repairMesh: boolean;
  meshRepairProfile: MeshRepairProfile;
  fillInferredMeshHoles: boolean;
  produceWatertightMesh: boolean;
  /** Run guarded LingBot-Depth completion after metric pose recovery. */
  lingbotDepthRefinement: boolean;
  /** Guarded learned RGB-D completion backend. */
  depthRefinementBackend: 'off' | 'lingbot' | 'mapanything' | 'da3';
  /** Feature-flagged provisional learned-depth preview for imported video. */
  experimentalRgbPreview: boolean;
  neuralSdfRefinement: boolean;
}

export type ArtifactStatus = 'ready' | 'building' | 'stale' | 'failed';

export interface ArtifactSummary {
  path: string;
  materialPath?: string;
  refinedCameraPath?: string;
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
  depthRefinement?: {
    enabled: boolean;
    method: string;
    frameCount?: number;
    acceptedFrameCount?: number;
    generatedPixelCount?: number;
    generatedToMeasuredPercent?: number;
    generatedFusionWeight?: number;
    generatedTrainingConfidence?: number;
    modelRevision?: string;
    modelSha256?: string;
  };
  neuralSdf?: {
    status: 'disabled' | 'skipped' | 'accepted' | 'rejected';
    method: string;
    reason?: string;
    cacheHit?: boolean;
    validation?: {
      heldOutSdfMaeM?: number;
      heldOutSdfP95M?: number;
      medianDisplacementM?: number;
      p95DisplacementM?: number;
    };
  };
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
  geometryWorkerAvailable: boolean;
  geometryStatus: string;
}

export type ArtifactTarget = 'pointCloud' | 'texturedMesh' | 'gaussianSplat';
export type ArtifactJobStatus = 'queued' | 'running' | 'cancelling' | 'cancelled' | 'failed' | 'complete';
export type MediaRestartStage = 'reuse' | 'analysis' | 'decode';

export interface BuildReusePolicy {
  mediaRestart: MediaRestartStage;
  rebuildRgbd: boolean;
}

export interface ArtifactJob {
  id: string;
  projectPath: string;
  targets: ArtifactTarget[];
  sourceKind: 'rgbd' | 'media' | 'hybrid';
  mediaRestart: MediaRestartStage | '';
  rebuildRgbd: boolean;
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
  rgbPreviewActive: boolean;
  rgbPreviewScaleStatus: 'MODEL_METRIC_UNVERIFIED' | 'MODEL_METRIC_VALIDATED' | 'USER_CALIBRATED' | 'RELATIVE_SCALE' | null;
  rgbPreviewConfidence: number | null;
  rgbPreviewDriftRisk: number | null;
  rgbPreviewSubmapCount: number | null;
  rgbPreviewAcceptedFrames: number | null;
  rgbPreviewRejectedFrames: number | null;
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
  liveContractVersion: number;
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
  trackingState: 'ready' | 'preview' | 'tracking' | 'searching' | 'relocalized' | 'frozen' | 'failed' | 'complete';
  trackingConfidence: number;
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
  trackingQueueDepth: number;
  mappingQueueDepth: number;
  trackingOverlap: number;
  poseUncertaintyMm?: number;
  poseUncertaintyDegrees?: number;
  poseLatencyMs?: number;
  mapUpdateLatencyMs?: number;
  mapUpdateHz: number;
  allocatedLiveMapBytes: number;
  activeVoxelCount: number;
  activeSurfelCount: number;
  residentSubmapCount: number;
  hostCachedSubmapCount: number;
  droppedPreviewJobCount: number;
  degradationLevel: number;
  loopClosureCount: number;
  loopCorrectionActive: boolean;
  liveScaleStatus: 'SENSOR_METRIC' | 'MODEL_METRIC_UNVERIFIED' | 'MODEL_METRIC_VALIDATED' | 'USER_CALIBRATED' | 'RELATIVE_SCALE';
  integrationFrozen: boolean;
  depthRmseMm?: number;
  liveReconstructionBackend?: string;
  reconstruction?: ReconstructionProgress;
  error?: string;
}

export interface LiveSubmapDescriptor {
  id: string;
  localOrigin: number[];
  globalFromLocal: number[];
  state: 'active' | 'complete' | 'corrected' | 'frozen';
  firstSequence: number;
  lastSequence: number;
  voxelSizeM: number;
  voxelCount: number;
  pointCount: number;
  observationCount: number;
  confidence: number;
  boundsMin: [number, number, number];
  boundsMax: [number, number, number];
  resident: 'gpu' | 'host';
}

export interface LiveCoverageSummary {
  contractVersion: 2;
  frameSequence: number;
  observedRatio: number;
  weakRatio: number;
  singleViewRatio: number;
  holeBoundaryRatio: number;
  guidance: string[];
}

export interface LiveReconstructionGuidance {
  contractVersion: 2;
  coverage: LiveCoverageSummary | null;
  submaps: {
    contractVersion: 2;
    frameSequence: number;
    submaps: LiveSubmapDescriptor[];
    poseGraph?: {
      nodeCount: number;
      loopConstraintCount: number;
      acceptedCorrectionCount: number;
      mapFromTrackingWorld: number[];
    };
    recentLoopEvents?: Array<{
      sequence: number;
      sourceSubmapId: string;
      targetSubmapId: string;
      accepted: boolean;
      requiresProductionRevalidation: boolean;
    }>;
    viewportCorrection?: { durationMs: number; active: boolean };
  } | null;
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
