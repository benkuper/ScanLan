<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { open, save } from '@tauri-apps/plugin-dialog';
  import * as THREE from 'three';
  import PointCloudPreview from './lib/components/PointCloudPreview.svelte';
  import {
    applyCloudTransform,
    artifactJobStatus,
    availableSensors,
    cancelArtifactJob,
    captureStatus,
    createProject,
    currentProject,
    exportGaussianSplat,
    exportPly,
    exportTexturedMesh,
    importMediaSource,
    latestArtifactJob,
    loadCameraFrames,
    loadLivePreviewFrame,
    loadPreview,
    loadPreviewMesh,
    removeCapture,
    removeMediaSource,
    resumeArtifactJob,
    runtimeInfo,
    startArtifactJob,
    startSensorPhase,
    stopSensorPhase,
    updateProjectSettings
  } from './lib/api';
  import type {
    ArtifactJob,
    ArtifactTarget,
    AvailableSensor,
    CameraFrame,
    CaptureSettings,
    CaptureStatus,
    CloudTransform,
    DepthFieldOfView,
    PackedPreviewFrame,
    PreviewMesh,
    PreviewPoint,
    ProjectSummary,
    ReconstructionProgress,
    RuntimeInfo
  } from './lib/types';

  let project: ProjectSummary | null = null;
  let sensor: CaptureStatus | null = null;
  let reconstruction: ReconstructionProgress | null = null;
  let activeJob: ArtifactJob | null = null;
  let runtime: RuntimeInfo | null = null;
  let sensorChoices: AvailableSensor[] = [];
  let selectedSensorOption = '';
  let discoveryInFlight = false;
  let sensorSessionEnabled = false;
  let previewPoints: PreviewPoint[] = [];
  let previewMesh: PreviewMesh | null = null;
  let packedPreviewFrame: PackedPreviewFrame | null = null;
  let cameraFrames: CameraFrame[] = [];
  let busy = false;
  let connecting = false;
  let initializationError = '';
  let statusTimer: number | undefined;
  let previewTimer: number | undefined;
  let settingsSaveTimer: number | undefined;
  let statusInFlight = false;
  let previewInFlight = false;
  let lastPreviewFrame = 0;
  let lastPreviewArrival = 0;
  let previewFps = 0;
  let lastStatusPoll = 0;
  let message = 'Select Scan sensors when you are ready to connect a depth sensor.';
  let viewMode: 'live' | 'preview' = 'live';
  let previewRenderMode: 'points' | 'mesh' = 'points';
  let buildPointCloud = true;
  let buildTexturedMesh = true;
  let buildGaussianSplat = false;
  let splatIterations = 30_000;
  let sourceMode: 'rgbd' | 'media' = 'rgbd';
  let selectedMediaSourceIds: string[] = [];

  let pointSize = 0.034;
  let pointOpacity = 0.92;
  let showColors = true;
  let showCameraFrames = false;
  let floorPickMode = false;
  let editMode = false;
  let anchorPickMode = false;
  let gizmoMode: 'translate' | 'rotate' | 'scale' = 'translate';
  let gizmoAnchor: [number, number, number] | null = null;
  let cloudTransform: CloudTransform = { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] };
  const identityTransform: CloudTransform = { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] };

  const transformStorageKey = (projectId: string) => `scanlan-cloud-transform:${projectId}`;
  const visualizationStorageKey = 'scanlan-visualization-preferences';

  function loadVisualizationPreferences() {
    const stored = localStorage.getItem(visualizationStorageKey);
    if (!stored) return;
    try {
      const value = JSON.parse(stored) as Partial<{
        pointSize: number;
        pointOpacity: number;
        showColors: boolean;
        showCameraFrames: boolean;
        previewRenderMode: 'points' | 'mesh';
        gizmoMode: 'translate' | 'rotate' | 'scale';
      }>;
      if (typeof value.pointSize === 'number') pointSize = value.pointSize;
      if (typeof value.pointOpacity === 'number') pointOpacity = value.pointOpacity;
      if (typeof value.showColors === 'boolean') showColors = value.showColors;
      if (typeof value.showCameraFrames === 'boolean') showCameraFrames = value.showCameraFrames;
      if (value.previewRenderMode === 'points' || value.previewRenderMode === 'mesh') {
        previewRenderMode = value.previewRenderMode;
      }
      if (value.gizmoMode === 'translate' || value.gizmoMode === 'rotate' || value.gizmoMode === 'scale') {
        gizmoMode = value.gizmoMode;
      }
    } catch {
      localStorage.removeItem(visualizationStorageKey);
    }
  }

  function persistVisualizationPreferences() {
    localStorage.setItem(visualizationStorageKey, JSON.stringify({ pointSize, pointOpacity, showColors, showCameraFrames, previewRenderMode, gizmoMode }));
  }

  const scheduleVisualizationSave = () => queueMicrotask(persistVisualizationPreferences);
  const scheduleProjectSettingsSave = () => queueMicrotask(queueProjectSettingsSave);

  function setGizmoMode(mode: 'translate' | 'rotate' | 'scale') {
    gizmoMode = mode;
    anchorPickMode = false;
    persistVisualizationPreferences();
  }

  function isTextEntryTarget(target: EventTarget | null) {
    if (!(target instanceof HTMLElement)) return false;
    return target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName);
  }

  function handlePreviewShortcut(event: KeyboardEvent) {
    if (viewMode !== 'preview' || event.defaultPrevented || event.ctrlKey || event.altKey || event.metaKey) return;

    if (event.key === 'Escape') {
      if (!editMode) return;
      event.preventDefault();
      editMode = false;
      anchorPickMode = false;
      return;
    }

    if (!canEdit || event.repeat || isTextEntryTarget(event.target)) return;
    const mode = {
      w: 'translate',
      e: 'rotate',
      r: 'scale'
    }[event.key.toLowerCase()] as 'translate' | 'rotate' | 'scale' | undefined;
    if (!mode) return;

    event.preventDefault();
    floorPickMode = false;
    editMode = true;
    setGizmoMode(mode);
  }

  function setPreviewRenderMode(mode: 'points' | 'mesh') {
    previewRenderMode = mode;
    persistVisualizationPreferences();
  }

  function pointCloudCenter(source: PreviewPoint[]): [number, number, number] {
    if (source.length === 0) return [0, 0, 0];
    const minimum = [...source[0].position];
    const maximum = [...source[0].position];
    for (let index = 1; index < source.length; index += 1) {
      const position = source[index].position;
      for (let axis = 0; axis < 3; axis += 1) {
        minimum[axis] = Math.min(minimum[axis], position[axis]);
        maximum[axis] = Math.max(maximum[axis], position[axis]);
      }
    }
    return [
      (minimum[0] + maximum[0]) / 2,
      (minimum[1] + maximum[1]) / 2,
      (minimum[2] + maximum[2]) / 2
    ];
  }

  function isPoint(value: unknown): value is [number, number, number] {
    return Array.isArray(value) && value.length === 3 && value.every((entry) => typeof entry === 'number' && Number.isFinite(entry));
  }

  async function saveProjectSettingsNow(path: string, settings: CaptureSettings) {
    try {
      const saved = await updateProjectSettings(path, settings);
      if (project?.id === saved.id) project = saved;
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    }
  }

  function queueProjectSettingsSave() {
    if (!project || capturing || processing) return;
    const path = project.path;
    const settings = { ...project.settings };
    if (settingsSaveTimer !== undefined) window.clearTimeout(settingsSaveTimer);
    settingsSaveTimer = window.setTimeout(() => {
      settingsSaveTimer = undefined;
      void saveProjectSettingsNow(path, settings);
    }, 120);
  }

  function setCaptureFps(fps: number) {
    if (!project) return;
    project = { ...project, settings: { ...project.settings, captureFps: fps } };
    queueProjectSettingsSave();
  }

  function setDepthFieldOfView(depthFieldOfView: DepthFieldOfView) {
    if (!project) return;
    project = { ...project, settings: { ...project.settings, depthFieldOfView } };
    queueProjectSettingsSave();
  }

  function setDepthBinned(depthBinned: boolean) {
    if (!project) return;
    project = { ...project, settings: { ...project.settings, depthBinned } };
    queueProjectSettingsSave();
  }

  const networkFemtoOption = '__network_femto__';

  function configuredSensorOption(settings: CaptureSettings) {
    if (settings.sensorConnection === 'network') return networkFemtoOption;
    return settings.sensorId || `${settings.sensorKind}:default`;
  }

  function sensorOptionLabel(choice: AvailableSensor) {
    const connection = choice.connection === 'network' ? choice.address || 'Network' : 'USB';
    const identity = choice.serial ? ` · ${choice.serial}` : '';
    return `${choice.name} · ${connection}${identity}`;
  }

  function applySensorChoice(choice: AvailableSensor) {
    if (!project) return false;
    const useImu = choice.supportsImu ? project.settings.useImu || project.settings.sensorKind === 'kinect_v2' : false;
    const next = {
      ...project.settings,
      sensorId: choice.id,
      sensorKind: choice.kind,
      sensorConnection: choice.connection,
      sensorAddress: choice.address,
      useImu
    };
    const changed = next.sensorId !== project.settings.sensorId
      || next.sensorKind !== project.settings.sensorKind
      || next.sensorConnection !== project.settings.sensorConnection
      || next.sensorAddress !== project.settings.sensorAddress
      || next.useImu !== project.settings.useImu;
    project = { ...project, settings: next };
    selectedSensorOption = choice.id;
    return changed;
  }

  function resetPackedPreview() {
    packedPreviewFrame = null;
    lastPreviewFrame = 0;
    lastPreviewArrival = 0;
    previewFps = 0;
  }

  async function scanSensors() {
    if (!project || discoveryInFlight || capturing || processing) return;
    discoveryInFlight = true;
    sensorSessionEnabled = false;
    resetPackedPreview();
    connecting = false;
    message = 'Scanning for connected depth sensors…';
    try {
      const choices = await availableSensors();
      sensorChoices = choices;
      const settings = project.settings;
      let preferred = choices.find((choice) => choice.id === settings.sensorId);
      if (!preferred) {
        preferred = choices.find((choice) => choice.kind === settings.sensorKind
          && choice.connection === settings.sensorConnection
          && (choice.connection !== 'network' || choice.address === settings.sensorAddress));
      }
      if (preferred) {
        const path = project.path;
        if (applySensorChoice(preferred)) await saveProjectSettingsNow(path, { ...project.settings });
        sensorSessionEnabled = true;
        connecting = true;
        message = `Opening ${selectedSensorName}…`;
        await refreshSensorStatus();
      } else {
        selectedSensorOption = configuredSensorOption(settings);
        sensor = null;
        previewPoints = [];
        message = choices.length > 0
          ? 'Sensors found. Choose one from the list to connect.'
          : 'No supported sensor was found. Check its power and cable, then scan again.';
      }
    } catch (error) {
      sensor = null;
      previewPoints = [];
      message = error instanceof Error ? error.message : String(error);
    } finally {
      discoveryInFlight = false;
    }
  }

  async function sensorDeviceChanged() {
    if (!project) return;
    if (selectedSensorOption === networkFemtoOption) {
      const address = project.settings.sensorConnection === 'network' ? project.settings.sensorAddress : '';
      project = {
        ...project,
        settings: {
          ...project.settings,
          sensorId: address ? `femto_mega:network:${address}` : 'femto_mega:network',
          sensorKind: 'femto_mega',
          sensorConnection: 'network',
          sensorAddress: address,
          useImu: true
        }
      };
    } else {
      const choice = sensorChoices.find((candidate) => candidate.id === selectedSensorOption);
      if (!choice) return;
      applySensorChoice(choice);
    }
    connecting = true;
    sensorSessionEnabled = true;
    sensor = null;
    previewPoints = [];
    resetPackedPreview();
    await saveProjectSettingsNow(project.path, { ...project.settings });
    if (project.settings.sensorConnection !== 'network' || project.settings.sensorAddress) {
      void refreshSensorStatus();
    } else {
      sensorSessionEnabled = false;
      connecting = false;
      message = 'Enter the camera IP address, then select Scan sensors.';
    }
  }

  function sensorAddressChanged() {
    if (!project) return;
    const address = project.settings.sensorAddress.trim();
    project = {
      ...project,
      settings: {
        ...project.settings,
        sensorId: address ? `femto_mega:network:${address}` : 'femto_mega:network',
        sensorAddress: address
      }
    };
    selectedSensorOption = networkFemtoOption;
    sensorSessionEnabled = false;
    connecting = false;
    sensor = null;
    previewPoints = [];
    resetPackedPreview();
    queueProjectSettingsSave();
    message = address
      ? 'Camera address saved. Select Scan sensors to probe it.'
      : 'Enter the camera IP address, then select Scan sensors.';
  }

  function loadTransform(projectId: string) {
    const stored = localStorage.getItem(transformStorageKey(projectId));
    cloudTransform = { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] };
    gizmoAnchor = null;
    anchorPickMode = false;
    if (!stored) return;
    try {
      const parsed = JSON.parse(stored) as Partial<CloudTransform> & { gizmoAnchor?: unknown };
      cloudTransform = {
        position: parsed.position ?? [0, 0, 0],
        rotation: parsed.rotation ?? [0, 0, 0],
        scale: parsed.scale ?? [1, 1, 1]
      };
      gizmoAnchor = isPoint(parsed.gizmoAnchor) ? [...parsed.gizmoAnchor] : null;
    } catch {
      localStorage.removeItem(transformStorageKey(projectId));
    }
  }

  function persistTransform() {
    if (project) {
      localStorage.setItem(transformStorageKey(project.id), JSON.stringify({ ...cloudTransform, gizmoAnchor }));
    }
  }

  $: capturing = sensor?.capturing ?? Boolean(project?.phases.some((phase) => phase.status === 'capturing'));
  $: jobRunning = Boolean(activeJob && ['queued', 'running', 'cancelling'].includes(activeJob.status));
  $: processing = project?.processingStatus === 'processing' || jobRunning;
  $: selectedSensorName = project?.settings.sensorKind === 'azure_kinect'
    ? 'Azure Kinect DK'
    : project?.settings.sensorKind === 'femto_mega'
      ? 'Orbbec Femto Mega'
      : 'Kinect v2';
  $: canEdit = viewMode === 'preview' && project?.processingStatus === 'complete' && previewPoints.length > 0;
  $: viewerTransform = viewMode === 'preview' ? cloudTransform : identityTransform;
  $: effectiveGizmoAnchor = gizmoAnchor ?? pointCloudCenter(previewPoints);
  $: completedPhases = project?.phases.filter((phase) => phase.status === 'complete').length ?? 0;
  $: hasMediaSources = (project?.mediaSources.length ?? 0) > 0;
  $: canBuildArtifacts = sourceMode === 'rgbd' ? completedPhases > 0 : selectedMediaSourceIds.length > 0;
  $: totalFrames = sensor?.totalFrameCount ?? project?.phases.reduce((sum, phase) => sum + phase.frameCount, 0) ?? 0;
  $: displayedPointCount = processing
    ? reconstruction?.pointCount ?? previewPoints.length
    : viewMode === 'preview'
      ? project?.pointCount
      : capturing
        ? previewPoints.length
        : sensorSessionEnabled && sensor?.sensorConnected
          ? packedPreviewFrame?.pointCount ?? previewPoints.length
          : previewPoints.length;

  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remaining = seconds % 60;
    return `${minutes}:${remaining.toString().padStart(2, '0')}`;
  };

  const formatCount = (value?: number) => {
    if (value === undefined) return '—';
    return new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 2 }).format(value);
  };

  const formatEta = (seconds?: number) => {
    if (seconds === undefined) return 'Estimating…';
    if (seconds < 60) return `about ${seconds}s left`;
    return `about ${Math.ceil(seconds / 60)} min left`;
  };

  const formatEtaValue = (seconds?: number | null) => {
    if (seconds == null) return 'Calculating';
    if (seconds <= 0) return 'Finishing';
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainder = seconds % 60;
    return remainder > 0 && minutes < 10 ? `${minutes}m ${remainder}s` : `${Math.ceil(seconds / 60)}m`;
  };

  type StageDefinition = { key: string; label: string; weight: number };
  const rgbdBaseStages: StageDefinition[] = [
    { key: 'prepare', label: 'Preparing inputs', weight: 0.05 },
    { key: 'track', label: 'Tracking cameras', weight: 0.13 },
    { key: 'trajectory', label: 'Optimizing trajectory', weight: 0.12 },
    { key: 'fuse', label: 'Fusing keyframes', weight: 0.08 },
    { key: 'cloud', label: 'Building point cloud', weight: 0.24 }
  ];
  const mediaStages: StageDefinition[] = [
    { key: 'prepare', label: 'Preparing media', weight: 0.05 },
    { key: 'filter', label: 'Selecting sharp frames', weight: 0.05 },
    { key: 'feature', label: 'Extracting GPU features', weight: 0.08 },
    { key: 'match', label: 'Matching camera views', weight: 0.12 },
    { key: 'map', label: 'Solving camera poses', weight: 0.15 },
    { key: 'splat', label: 'Training Gaussian splat', weight: 0.5 },
    { key: 'publish', label: 'Publishing splat', weight: 0.05 }
  ];

  function stageKey(value: string) {
    const stage = value.toLowerCase().replaceAll('_', ' ');
    if (stage.includes('complete') || stage.includes('export') || stage.includes('publish')) return 'publish';
    if (stage.includes('splat') && (stage.includes('train') || stage.includes('initial'))) return 'splat';
    if (stage.includes('mesh')) return 'mesh';
    if (stage.includes('preparing splat') || stage.includes('posed frame')) return 'dataset';
    if (stage.includes('mapping') || stage.includes('registration')) return 'map';
    if (stage.includes('matching')) return 'match';
    if (stage.includes('feature')) return 'feature';
    if (stage.includes('filter')) return 'filter';
    if (stage.includes('building') || stage.includes('cleaning cloud')) return 'cloud';
    if (stage.includes('fusing') || stage.includes('previewing') || stage.includes('loading cache')) return 'fuse';
    if (stage.includes('stabiliz') || stage.includes('aligning')) return 'trajectory';
    if (stage.includes('tracking') || stage.includes('placing') || stage.includes('keyframe')) return 'track';
    return 'prepare';
  }

  function stagesFor(job: ArtifactJob | null) {
    if (job?.pipeline === 'media_gaussian') return mediaStages;
    const stages = [...rgbdBaseStages];
    const wantsMesh = job?.targets.includes('texturedMesh') ?? buildTexturedMesh;
    const wantsSplat = job?.targets.includes('gaussianSplat') ?? buildGaussianSplat;
    if (wantsMesh || wantsSplat) stages.push({ key: 'dataset', label: 'Preparing posed frames', weight: 0.08 });
    if (wantsMesh) stages.push({ key: 'mesh', label: 'Texturing mesh', weight: 0.2 });
    if (wantsSplat) stages.push({ key: 'splat', label: 'Training Gaussian splat', weight: 0.55 });
    stages.push({ key: 'publish', label: 'Publishing artifacts', weight: 0.05 });
    return stages;
  }

  function stageFeedback(job: ArtifactJob | null, progress: ReconstructionProgress | null) {
    const stages = stagesFor(job);
    const key = stageKey(job?.stage || progress?.stage || 'preparing');
    const found = stages.findIndex((stage) => stage.key === key);
    const index = found < 0 ? 0 : found;
    return { label: stages[index].label, current: index + 1, total: stages.length };
  }

  function weightedOverallProgress(job: ArtifactJob | null, progress: ReconstructionProgress | null) {
    if (!job) return Math.max(0, Math.min(1, progress?.progress ?? 0));
    const stages = stagesFor(job);
    const key = stageKey(job.stage || progress?.stage || 'preparing');
    const index = Math.max(0, stages.findIndex((stage) => stage.key === key));
    const totalWeight = stages.reduce((sum, stage) => sum + stage.weight, 0);
    const completedWeight = stages.slice(0, index).reduce((sum, stage) => sum + stage.weight, 0);
    const phaseProgress = job.stageProgress ?? progress?.stageProgress ?? 0;
    const fraction = key === 'publish' && job.stage.toLowerCase().includes('complete') ? 1 : phaseProgress;
    return Math.max(0, Math.min(1, (completedWeight + stages[index].weight * fraction) / totalWeight));
  }

  function estimateOverallEta(job: ArtifactJob | null, progress: number) {
    if (!job) return null;
    if (progress >= 0.995) return 0;
    const started = Date.parse(job.startedAt ?? job.createdAt);
    const elapsed = Number.isFinite(started) ? Math.max(0, (Date.now() - started) / 1000) : 0;
    if (elapsed < 2 || progress < 0.02) return null;
    return Math.max(1, Math.round(elapsed * (1 - progress) / progress));
  }

  $: buildStage = stageFeedback(activeJob, reconstruction);
  $: overallBuildProgress = weightedOverallProgress(activeJob, reconstruction);
  $: currentStageProgress = Math.max(0, Math.min(1, activeJob?.stageProgress ?? reconstruction?.stageProgress ?? 0));
  $: buildDetail = activeJob?.detail?.trim() || reconstruction?.detail || 'Preparing the next artifact stage';
  $: buildBackend = activeJob?.computeBackend || reconstruction?.computeBackend || (activeJob?.stage.includes('splat') ? 'CUDA AMP / gsplat' : 'GPU preferred');
  $: totalBuildEta = activeJob ? estimateOverallEta(activeJob, overallBuildProgress) : reconstruction?.etaSeconds;
  $: currentStageEta = activeJob?.stageEtaSeconds ?? reconstruction?.stageEtaSeconds;

  const confidenceClass = (score?: number) => score === undefined ? '' : score >= 80 ? 'high' : score >= 60 ? 'medium' : 'low';

  async function refreshResultPreview() {
    if (!project) return;
    try {
      const [nextPoints, nextCameraFrames, nextMesh] = await Promise.all([
        loadPreview(project.path),
        loadCameraFrames(project.path),
        project.meshOutputPath ? loadPreviewMesh(project.path).catch(() => null) : Promise.resolve(null)
      ]);
      previewPoints = nextPoints;
      cameraFrames = nextCameraFrames;
      previewMesh = nextMesh;
      if (!nextMesh && previewRenderMode === 'mesh') setPreviewRenderMode('points');
    } catch {
      previewPoints = [];
      resetPackedPreview();
      cameraFrames = [];
      previewMesh = null;
      if (previewRenderMode === 'mesh') setPreviewRenderMode('points');
    }
  }

  async function refreshSensorStatus() {
    if (!project || statusInFlight || (!sensorSessionEnabled && !capturing && !processing && !activeJob)) return;
    const now = performance.now();
    const minimumInterval = processing ? 250 : 100;
    if (now - lastStatusPoll < minimumInterval) return;
    lastStatusPoll = now;
    statusInFlight = true;
    try {
      const status = await captureStatus();
      const localSettings = project.settings;
      project = { ...status.project, settings: status.capturing || processing ? status.project.settings : localSettings };
      sensor = status;
      reconstruction = status.reconstruction ?? null;
      connecting = !status.sensorConnected && !status.sensorPaused && !status.error;

      if (status.capturing) {
        viewMode = 'live';
        if (status.preview.length > 0) previewPoints = status.preview;
        message = `Capturing frame ${status.frameCount} · ${status.streamFps.toFixed(1)} saved fps · final placement happens during build`;
      } else if (processing && status.preview.length > 0) {
        previewPoints = status.preview;
      } else if (viewMode === 'live' && status.preview.length > 0) {
        previewPoints = status.preview;
      }

      if (status.error) {
        message = `${status.error} Select Scan sensors to retry.`;
        if (!status.capturing && !processing) {
          sensorSessionEnabled = false;
          connecting = false;
        }
      }
      else if (!status.sensorConnected && !processing) message = status.sensorStatus;
      else if (processing && reconstruction && !activeJob) {
        message = `${reconstruction.stage}: ${reconstruction.detail}`;
      } else if (!status.capturing && status.sensorConnected && viewMode === 'live') {
        message = `Live ${status.sensorName} preview · ${status.previewPointCount.toLocaleString()} visible points · ${status.streamFps.toFixed(1)} sensor fps${previewFps > 0 ? ` · ${previewFps.toFixed(1)} preview fps` : ''}`;
      }
      if (activeJob) {
        try {
          const updatedJob = await artifactJobStatus(project.path, activeJob.id);
          activeJob = updatedJob;
          reconstruction = {
            stage: updatedJob.stage,
            detail: updatedJob.iteration != null
              ? `CUDA splat iteration ${updatedJob.iteration.toLocaleString()} of ${(updatedJob.totalIterations ?? 0).toLocaleString()}${updatedJob.loss != null ? ` · loss ${updatedJob.loss.toFixed(4)}` : ''}`
              : updatedJob.detail?.trim() || status.reconstruction?.detail || updatedJob.stage.replaceAll('_', ' '),
            progress: updatedJob.progress,
            processedUnits: updatedJob.iteration ?? Math.round(updatedJob.progress * 1000),
            totalUnits: updatedJob.totalIterations ?? 1000,
            etaSeconds: updatedJob.etaSeconds ?? undefined,
            stageProgress: updatedJob.stageProgress ?? status.reconstruction?.stageProgress,
            stageEtaSeconds: updatedJob.stageEtaSeconds ?? status.reconstruction?.stageEtaSeconds,
            elapsedSeconds: updatedJob.elapsedSeconds ?? status.reconstruction?.elapsedSeconds,
            computeBackend: updatedJob.computeBackend ?? (updatedJob.stage.includes('splat') ? 'CUDA AMP / gsplat' : status.reconstruction?.computeBackend)
          };
          if (['complete', 'failed', 'cancelled'].includes(updatedJob.status)) {
            project = await currentProject();
            activeJob = updatedJob.resumable ? updatedJob : null;
            if (updatedJob.status === 'complete') {
              viewMode = 'preview';
              if (project.artifacts.pointCloud || project.artifacts.texturedMesh) {
                await refreshResultPreview();
              }
              message = `Artifact job complete${project.processingBackend ? ` · ${project.processingBackend}` : ''}${project.artifacts.gaussianSplat ? ' · Gaussian PLY ready' : ''}.`;
            } else {
              message = updatedJob.error ?? `Artifact job ${updatedJob.status}.`;
            }
          }
        } catch (error) {
          message = `Artifact status: ${error instanceof Error ? error.message : String(error)}`;
        }
      }
    } catch (error) {
      sensor = sensor ? { ...sensor, sensorConnected: false, sensorStatus: String(error) } : null;
      message = error instanceof Error ? error.message : String(error);
      connecting = false;
      if (!capturing && !processing) sensorSessionEnabled = false;
    } finally {
      statusInFlight = false;
    }
  }

  function decodeLivePreviewFrame(buffer: ArrayBuffer): PackedPreviewFrame | null {
    if (buffer.byteLength < 24) return null;
    const view = new DataView(buffer);
    if (view.getUint8(0) !== 0x4b || view.getUint8(1) !== 0x32
      || view.getUint8(2) !== 0x50 || view.getUint8(3) !== 0x31) return null;
    const frameCount = view.getUint32(4, true);
    const pointCount = view.getUint32(20, true);
    if (pointCount > 100_000 || buffer.byteLength !== 24 + pointCount * 15) return null;
    const positions = new Float32Array(pointCount * 3);
    const colors = new Uint8Array(pointCount * 3);
    for (let point = 0, source = 24, target = 0; point < pointCount; point += 1, source += 15, target += 3) {
      positions[target] = view.getFloat32(source, true);
      positions[target + 1] = view.getFloat32(source + 4, true);
      positions[target + 2] = view.getFloat32(source + 8, true);
      colors[target] = view.getUint8(source + 12);
      colors[target + 1] = view.getUint8(source + 13);
      colors[target + 2] = view.getUint8(source + 14);
    }
    return { frameCount, pointCount, positions, colors };
  }

  async function refreshLivePreview() {
    if (!project || previewInFlight || !sensorSessionEnabled || capturing || processing || viewMode !== 'live') return;
    previewInFlight = true;
    try {
      const packet = await loadLivePreviewFrame(lastPreviewFrame);
      const frame = decodeLivePreviewFrame(packet);
      if (!frame || frame.frameCount === lastPreviewFrame) return;
      lastPreviewFrame = frame.frameCount;
      packedPreviewFrame = frame;
      const now = performance.now();
      if (lastPreviewArrival > 0) {
        const instantFps = Math.min(60, 1000 / Math.max(1, now - lastPreviewArrival));
        previewFps = previewFps > 0 ? previewFps * 0.8 + instantFps * 0.2 : instantFps;
      }
      lastPreviewArrival = now;
    } catch {
      // capture_status owns connection errors and the user-facing message.
    } finally {
      previewInFlight = false;
    }
  }

  async function showView(mode: 'live' | 'preview') {
    viewMode = mode;
    if (mode === 'live') {
      floorPickMode = false;
      editMode = false;
    }
    if (mode === 'preview') await refreshResultPreview();
    else await refreshSensorStatus();
  }

  async function newProject() {
    busy = true;
    message = 'Creating a clean scan project…';
    try {
      if (settingsSaveTimer !== undefined) {
        window.clearTimeout(settingsSaveTimer);
        settingsSaveTimer = undefined;
      }
      if (project && !capturing && !processing) {
        await saveProjectSettingsNow(project.path, { ...project.settings });
      }
      project = await createProject();
      cloudTransform = { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] };
      floorPickMode = false;
      editMode = false;
      previewPoints = [];
      resetPackedPreview();
      cameraFrames = [];
      previewMesh = null;
      viewMode = 'live';
      message = 'Project created. Begin with a slow reference phase.';
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    } finally {
      busy = false;
    }
  }

  async function captureAction() {
    if (!project) return;
    busy = true;
    try {
      if (capturing) {
        message = 'Finishing the phase and closing its frame index…';
        project = await stopSensorPhase();
        const latestPhase = project.phases.at(-1);
        message = latestPhase?.status === 'complete'
          ? `Phase saved with ${latestPhase.frameCount} RGB-D frames.`
          : 'No usable frames were saved. Check tracking and capture for a few seconds.';
      } else {
        if (!sensor?.sensorConnected) {
          message = sensor?.sensorStatus ?? `${selectedSensorName} is not streaming.`;
          return;
        }
        previewPoints = [];
        resetPackedPreview();
        viewMode = 'live';
        message = `Starting ${selectedSensorName} capture…`;
        project = await startSensorPhase(project.path, project.settings);
      }
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    } finally {
      busy = false;
      void refreshSensorStatus();
    }
  }

  async function removeCaptureAction(phaseId: string, phaseName: string) {
    if (!window.confirm(`Permanently remove "${phaseName}" and invalidate the current reconstruction?`)) return;
    busy = true;
    try {
      project = await removeCapture(phaseId);
      viewMode = 'live';
      floorPickMode = false;
      editMode = false;
      previewPoints = [];
      cameraFrames = [];
      previewMesh = null;
      message = `${phaseName} removed. Rebuild when the remaining captures are ready.`;
      await refreshSensorStatus();
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    } finally {
      busy = false;
    }
  }

  function toggleMediaSource(sourceId: string) {
    selectedMediaSourceIds = selectedMediaSourceIds.includes(sourceId)
      ? selectedMediaSourceIds.filter((value) => value !== sourceId)
      : [...selectedMediaSourceIds, sourceId];
  }

  async function importPhotosAction() {
    if (!project) return;
    const selected = await open({ title: 'Import a photo folder', directory: true, multiple: false });
    if (!selected || Array.isArray(selected)) return;
    busy = true;
    try {
      project = await importMediaSource(project.path, 'photos', [selected]);
      selectedMediaSourceIds = project.mediaSources.map((source) => source.id);
      sourceMode = 'media';
      buildPointCloud = false;
      buildTexturedMesh = false;
      buildGaussianSplat = true;
      message = 'Photo folder copied into the project. Ready for GPU COLMAP registration and splat training.';
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    } finally {
      busy = false;
    }
  }

  async function importVideoAction() {
    if (!project) return;
    const selected = await open({
      title: 'Import a video',
      multiple: false,
      filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'mkv', 'avi', 'm4v', 'webm'] }]
    });
    if (!selected || Array.isArray(selected)) return;
    busy = true;
    try {
      project = await importMediaSource(project.path, 'video', [selected]);
      selectedMediaSourceIds = project.mediaSources.map((source) => source.id);
      sourceMode = 'media';
      buildPointCloud = false;
      buildTexturedMesh = false;
      buildGaussianSplat = true;
      message = 'Video copied into the project. FFmpeg will filter frames before GPU COLMAP registration.';
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    } finally {
      busy = false;
    }
  }

  async function removeMediaSourceAction(sourceId: string, name: string) {
    if (!project || !window.confirm(`Permanently remove media source "${name}"?`)) return;
    busy = true;
    try {
      project = await removeMediaSource(project.path, sourceId);
      selectedMediaSourceIds = selectedMediaSourceIds.filter((value) => value !== sourceId);
      message = `${name} removed.`;
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    } finally {
      busy = false;
    }
  }

  async function processCloud() {
    if (!project) return;
    if (activeJob) {
      try {
        if (activeJob.resumable && ['failed', 'cancelled'].includes(activeJob.status)) {
          activeJob = await resumeArtifactJob(project.path, activeJob.id);
          project = { ...project, processingStatus: 'processing', processingError: undefined, activeJob: activeJob.id };
          message = 'Resuming CUDA splat training from its matching checkpoint…';
        } else {
          activeJob = await cancelArtifactJob(project.path, activeJob.id);
          message = 'Cancelling artifact workers and saving any splat checkpoint…';
        }
      } catch (error) {
        message = error instanceof Error ? error.message : String(error);
      }
      return;
    }
    if (!canBuildArtifacts) return;
    const targets: ArtifactTarget[] = sourceMode === 'media'
      ? ['gaussianSplat']
      : [
          ...(buildPointCloud ? ['pointCloud' as const] : []),
          ...(buildTexturedMesh ? ['texturedMesh' as const] : []),
          ...(buildGaussianSplat ? ['gaussianSplat' as const] : [])
        ];
    if (targets.length === 0) {
      message = 'Choose at least one artifact to build.';
      return;
    }
    if (targets.includes('gaussianSplat') && !runtime?.splatWorkerAvailable) {
      message = runtime?.splatStatus ?? 'Install the optional splat runtime first with npm run prepare:splat.';
      return;
    }
    busy = true;
    reconstruction = null;
    editMode = false;
    project = {
      ...project,
      processingStatus: 'processing',
      processingError: undefined
    };
    message = sourceMode === 'media'
      ? 'Starting GPU media registration and splat training…'
      : 'Starting GPU-preferred RGB-D artifact build…';
    try {
      activeJob = await startArtifactJob(
        project.path,
        sourceMode === 'media' ? 'media_gaussian' : 'rgbd_reconstruction',
        targets,
        sourceMode === 'media' ? selectedMediaSourceIds : [],
        splatIterations
      );
      project = { ...project, activeJob: activeJob.id, processingStatus: 'processing' };
    } catch (error) {
      const failure = error instanceof Error ? error.message : String(error);
      try {
        project = await currentProject();
      } catch {
        project = { ...project, processingStatus: 'failed', processingError: failure };
      }
      message = failure;
    } finally {
      busy = false;
    }
  }

  function setFloorTransform(transform: CloudTransform) {
    cloudTransform = transform;
    floorPickMode = false;
    persistTransform();
  }

  function handleGizmoTransform(transform: CloudTransform) {
    cloudTransform = transform;
  }

  function commitGizmoTransform() {
    persistTransform();
    message = `${gizmoMode === 'translate' ? 'Position' : gizmoMode === 'rotate' ? 'Rotation' : 'Scale'} updated with the edit gizmo.`;
  }

  function toggleAnchorPick() {
    anchorPickMode = !anchorPickMode;
    floorPickMode = false;
    if (anchorPickMode) message = 'Click a point on the mesh to place the gizmo anchor.';
  }

  function setGizmoAnchor(anchor: [number, number, number]) {
    gizmoAnchor = [...anchor];
    anchorPickMode = false;
    editMode = true;
    persistTransform();
    message = 'Gizmo anchor placed on the mesh.';
  }

  function updateGizmoAnchor(axis: number, value: number) {
    if (!Number.isFinite(value)) return;
    const next = [...effectiveGizmoAnchor] as [number, number, number];
    next[axis] = value;
    gizmoAnchor = next;
    persistTransform();
  }

  function centerGizmoAnchor() {
    gizmoAnchor = null;
    anchorPickMode = false;
    persistTransform();
    message = 'Gizmo anchor centered on the mesh bounds.';
  }

  function applyWorldRotation(axis: 'X' | 'Y' | 'Z', degrees: number, label: string) {
    const unit = axis === 'X'
      ? new THREE.Vector3(1, 0, 0)
      : axis === 'Y'
        ? new THREE.Vector3(0, 1, 0)
        : new THREE.Vector3(0, 0, 1);
    const current = new THREE.Quaternion().setFromEuler(new THREE.Euler(
      THREE.MathUtils.degToRad(cloudTransform.rotation[0]),
      THREE.MathUtils.degToRad(cloudTransform.rotation[1]),
      THREE.MathUtils.degToRad(cloudTransform.rotation[2]),
      'XYZ'
    ));
    const rotated = new THREE.Quaternion()
      .setFromAxisAngle(unit, THREE.MathUtils.degToRad(degrees))
      .multiply(current);
    const euler = new THREE.Euler().setFromQuaternion(rotated, 'XYZ');
    cloudTransform = {
      ...cloudTransform,
      rotation: [
        THREE.MathUtils.radToDeg(euler.x),
        THREE.MathUtils.radToDeg(euler.y),
        THREE.MathUtils.radToDeg(euler.z)
      ]
    };
    persistTransform();
    message = label;
  }

  function flipAxis(axis: 'X' | 'Y' | 'Z') {
    const axisIndex = axis === 'X' ? 0 : axis === 'Y' ? 1 : 2;
    const scale = [...cloudTransform.scale] as [number, number, number];
    scale[axisIndex] *= -1;
    cloudTransform = { ...cloudTransform, scale };
    persistTransform();
    message = `${axis} axis direction flipped.`;
  }

  function alignRoomAxes() {
    if (previewPoints.length < 30) {
      message = 'Not enough reconstructed points to estimate the room axes.';
      return;
    }
    const quaternion = new THREE.Quaternion().setFromEuler(new THREE.Euler(
      THREE.MathUtils.degToRad(cloudTransform.rotation[0]),
      THREE.MathUtils.degToRad(cloudTransform.rotation[1]),
      THREE.MathUtils.degToRad(cloudTransform.rotation[2]),
      'XYZ'
    ));
    const stride = Math.max(1, Math.floor(previewPoints.length / 20_000));
    const projected: Array<[number, number]> = [];
    for (let index = 0; index < previewPoints.length; index += stride) {
      const value = new THREE.Vector3()
        .fromArray(previewPoints[index].position)
        .multiply(new THREE.Vector3().fromArray(cloudTransform.scale))
        .applyQuaternion(quaternion);
      projected.push([value.x, value.z]);
    }
    const meanX = projected.reduce((sum, point) => sum + point[0], 0) / projected.length;
    const meanZ = projected.reduce((sum, point) => sum + point[1], 0) / projected.length;
    let covarianceXX = 0;
    let covarianceXZ = 0;
    let covarianceZZ = 0;
    for (const [x, z] of projected) {
      const dx = x - meanX;
      const dz = z - meanZ;
      covarianceXX += dx * dx;
      covarianceXZ += dx * dz;
      covarianceZZ += dz * dz;
    }
    const principalAngle = 0.5 * Math.atan2(2 * covarianceXZ, covarianceXX - covarianceZZ);
    const rightAngle = Math.PI / 2;
    const targetAngle = Math.round(principalAngle / rightAngle) * rightAngle;
    const correctionDegrees = THREE.MathUtils.radToDeg(targetAngle - principalAngle);
    applyWorldRotation('Y', correctionDegrees, `Room axes aligned by ${Math.abs(correctionDegrees).toFixed(1)} degrees.`);
  }

  function resetTransform() {
    cloudTransform = { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] };
    gizmoAnchor = null;
    anchorPickMode = false;
    if (project) localStorage.removeItem(transformStorageKey(project.id));
    message = 'Point-cloud orientation reset.';
  }

  async function applyTransformToExport() {
    if (!project || project.processingStatus !== 'complete') return;
    busy = true;
    try {
      previewPoints = await applyCloudTransform(project.path, cloudTransform);
      cameraFrames = await loadCameraFrames(project.path);
      previewMesh = project.meshOutputPath ? await loadPreviewMesh(project.path) : null;
      cloudTransform = { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] };
      gizmoAnchor = null;
      anchorPickMode = false;
      localStorage.removeItem(transformStorageKey(project.id));
      viewMode = 'preview';
      editMode = false;
      message = 'Orientation applied to the point cloud, textured mesh, and camera poses. Untransformed geometry backups were kept.';
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    } finally {
      busy = false;
    }
  }

  async function exportPlyAction() {
    if (!project || project.processingStatus !== 'complete') return;
    busy = true;
    try {
      const destinationPath = await save({
        title: 'Export Unity-ready PLY',
        defaultPath: 'room-cloud-unity.ply',
        filters: [{ name: 'PLY point cloud', extensions: ['ply'] }]
      });
      if (!destinationPath) return;
      const savedPath = await exportPly(project.path, destinationPath);
      message = `Unity-ready PLY saved to ${savedPath}. The X axis was corrected for Unity.`;
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    } finally {
      busy = false;
    }
  }

  async function exportTexturedMeshAction() {
    if (!project || project.processingStatus !== 'complete' || !project.meshOutputPath) return;
    busy = true;
    try {
      const destinationPath = await save({
        title: 'Export textured OBJ bundle',
        defaultPath: 'room-mesh.obj',
        filters: [{ name: 'Wavefront textured mesh', extensions: ['obj'] }]
      });
      if (!destinationPath) return;
      const savedPath = await exportTexturedMesh(project.path, destinationPath);
      message = `Textured OBJ, MTL, and RGB texture saved beside ${savedPath}.`;
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    } finally {
      busy = false;
    }
  }

  async function exportGaussianSplatAction() {
    if (!project?.artifacts.gaussianSplat || project.artifacts.gaussianSplat.stale) return;
    busy = true;
    try {
      const destinationPath = await save({
        title: 'Export canonical Gaussian splat',
        defaultPath: 'room-splat.ply',
        filters: [{ name: '3D Gaussian splat PLY', extensions: ['ply'] }]
      });
      if (!destinationPath) return;
      const savedPath = await exportGaussianSplat(project.path, destinationPath);
      message = `Gaussian PLY and coordinate sidecars saved to ${savedPath}.`;
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    } finally {
      busy = false;
    }
  }

  onMount(() => {
    // Remove the old global transform once. It used to rotate every project and
    // even the raw sensor feed, which could make a new live view appear inverted.
    localStorage.removeItem('scanlan-cloud-transform');
    loadVisualizationPreferences();
    void (async () => {
      try {
        project = await currentProject();
        runtime = await runtimeInfo().catch(() => null);
        loadTransform(project.id);
        selectedSensorOption = configuredSensorOption(project.settings);
        selectedMediaSourceIds = project.mediaSources.map((source) => source.id);
        if (project.phases.length === 0 && project.mediaSources.length > 0) {
          sourceMode = 'media';
          buildPointCloud = false;
          buildTexturedMesh = false;
          buildGaussianSplat = true;
        }
        if (project.activeJob) {
          activeJob = await artifactJobStatus(project.path, project.activeJob).catch(() => null);
        } else {
          const latestJob = await latestArtifactJob(project.path).catch(() => null);
          activeJob = latestJob?.resumable ? latestJob : null;
        }
        if (project.processingStatus === 'failed') {
          message = project.processingError ?? 'The last reconstruction failed; captured phases are still available.';
        } else {
          message = 'Sensor discovery is manual. Select Scan sensors when you are ready to connect.';
        }
      } catch (error) {
        initializationError = error instanceof Error ? error.message : String(error);
        message = initializationError;
        connecting = false;
      } finally {
        if (statusTimer === undefined) statusTimer = window.setInterval(() => void refreshSensorStatus(), 100);
        if (previewTimer === undefined) previewTimer = window.setInterval(() => void refreshLivePreview(), 25);
      }
    })();
  });

  onDestroy(() => {
    if (statusTimer !== undefined) window.clearInterval(statusTimer);
    if (previewTimer !== undefined) window.clearInterval(previewTimer);
    if (settingsSaveTimer !== undefined) window.clearTimeout(settingsSaveTimer);
  });
</script>

<svelte:head><title>ScanLan</title></svelte:head>
<svelte:window on:keydown={handlePreviewShortcut} />

<main class="app-shell">
  <header class="topbar">
    <div class="brand">
      <div class="brand-mark" aria-hidden="true"><span></span><span></span><span></span></div>
      <div><h1>ScanLan</h1><p>Track live. Capture phases. Reconstruct when ready.</p></div>
    </div>
    <div class="top-actions">
      <div class="runtime-pill" class:connected={!processing && sensor?.sensorConnected} class:paused={processing || sensor?.sensorPaused} title={processing ? 'Sensor preview remains paused until the artifact job is fully published' : sensor?.sensorStatus}>
        <span></span>
        {processing ? 'Sensor paused for artifact build' : sensor?.sensorPaused ? 'Sensor paused for build' : discoveryInFlight ? 'Scanning for sensors…' : connecting ? `Opening ${selectedSensorName}…` : sensor?.sensorConnected ? `${sensor.sensorName} · ${sensor.streamFps.toFixed(1)} sensor fps${!capturing && previewFps > 0 ? ` · ${previewFps.toFixed(1)} preview fps` : ''}` : sensorSessionEnabled ? `${selectedSensorName} disconnected` : 'Sensor scan required'}
      </div>
      <button class="button ghost" on:click={newProject} disabled={busy || capturing || processing}>New project</button>
      <button class:stopping={capturing} class="button primary" on:click={captureAction} disabled={busy || processing || !project || (!capturing && !sensor?.sensorConnected)}>
        <span class="record-dot"></span>{capturing ? 'Stop capture' : 'Capture phase'}
      </button>
    </div>
  </header>

  {#if project}
    <section class="workspace">
      <aside class="sidebar panel">
        <div class="eyebrow">Active project</div>
        <h2>{project.name}</h2>
        <p class="project-path" title={project.path}>{project.path}</p>

        <div class="phase-heading"><span>Capture phases</span><span class="count-badge">{project.phases.length}</span></div>
        <div class="phase-list">
          {#if project.phases.length === 0}
            <div class="empty-phase"><div class="empty-icon">01</div><strong>No captures yet</strong><span>Start with a slow pass around a textured reference area.</span></div>
          {:else}
            {#each project.phases as phase, index}
              <article class="phase-card">
                <div class="phase-index">{String(index + 1).padStart(2, '0')}</div>
                <div class="phase-copy"><strong>{phase.name}</strong><span>{phase.frameCount} frames · {formatDuration(phase.durationSeconds)}</span><small><i></i>{phase.overlapHint}</small></div>
                <div class="phase-actions">
                  <div class:capturing={phase.status === 'capturing'} class:failed={phase.status === 'failed'} class="phase-status" title={phase.status}>
                    {phase.status === 'capturing' ? '●' : phase.status === 'failed' ? '!' : '✓'}
                  </div>
                  {#if phase.status !== 'capturing'}
                    <button class="remove-phase" title={`Remove ${phase.name}`} aria-label={`Remove ${phase.name}`} disabled={busy || capturing || processing} on:click={() => removeCaptureAction(phase.id, phase.name)}>×</button>
                  {/if}
                </div>
              </article>
            {/each}
          {/if}
        </div>
        <button class="add-phase" on:click={captureAction} disabled={busy || capturing || !sensor?.sensorConnected}><span>+</span> Add another phase</button>

        <div class="phase-heading media-heading"><span>Photo &amp; video sources</span><span class="count-badge">{project.mediaSources.length}</span></div>
        <div class="phase-list media-list">
          {#each project.mediaSources as source}
            <article class="phase-card media-card">
              <input type="checkbox" checked={selectedMediaSourceIds.includes(source.id)} on:change={() => toggleMediaSource(source.id)} disabled={processing} aria-label={`Use ${source.name}`} />
              <div class="phase-copy"><strong>{source.name}</strong><span>{source.kind === 'video' ? 'Video' : `${source.imageCount} photos`} · {source.status}</span><small><i></i>{source.quality?.detail ?? 'Ready for registration'}</small></div>
              <button class="remove-phase" title={`Remove ${source.name}`} disabled={busy || processing} on:click={() => removeMediaSourceAction(source.id, source.name)}>×</button>
            </article>
          {/each}
        </div>
        <div class="media-actions">
          <button class="add-phase" on:click={importPhotosAction} disabled={busy || processing}>Import photos…</button>
          <button class="add-phase" on:click={importVideoAction} disabled={busy || processing}>Import video…</button>
        </div>

        <div class="scan-stats">
          <div><span>Saved frames</span><strong>{totalFrames.toLocaleString()}</strong></div>
          <div><span>Current phase</span><strong>{capturing ? sensor?.frameCount ?? 0 : '—'}</strong></div>
          <div><span>Live points</span><strong>{formatCount(capturing ? previewPoints.length : sensorSessionEnabled && sensor?.sensorConnected ? packedPreviewFrame?.pointCount ?? previewPoints.length : previewPoints.length)}</strong></div>
          <div><span>Tracking</span><strong class:warning={capturing && project.settings.sensorKind === 'kinect_v2' && !sensor?.tracking}>{capturing ? sensor?.tracking ? 'Locked' : project.settings.sensorKind === 'kinect_v2' ? 'Searching' : sensor?.imuActive ? 'IMU-aided' : 'Offline RGB-D' : 'Standby'}</strong></div>
          <div><span>Motion aid</span><strong>{sensor?.imuActive ? `IMU ${sensor.imuRateHz.toFixed(0)} Hz` : project.settings.useImu ? 'Waiting for IMU' : 'RGB-D only'}</strong></div>
        </div>

        <div class="capture-tip"><div class="tip-icon">◎</div><div><strong>Stable capture guidance</strong><span>The live view stays camera-relative. When enabled, calibrated gyro samples seed offline RGB-D odometry during Build.</span></div></div>
      </aside>

      <section class="main-stage">
        <div class="stage-header">
          <div><div class="eyebrow">{capturing ? 'Live capture guidance' : viewMode === 'preview' ? processing ? 'Building reconstruction' : 'Model preview' : 'Live sensor data'}</div><h2>{capturing ? `Current ${selectedSensorName} frame` : viewMode === 'preview' ? processing ? '3D model being built' : previewRenderMode === 'mesh' ? 'Textured 3D mesh' : 'Registered point cloud' : `${selectedSensorName} point cloud`}</h2></div>
          <div class="stage-actions">
            <div class="metrics">
              <div title={project.framesUsed !== undefined ? `${project.framesUsed} keyframes selected from ${totalFrames} captured frames` : 'Captured frames'}><span>{project.framesUsed !== undefined ? 'Frames used' : 'Frames'}</span><strong>{project.framesUsed !== undefined ? `${project.framesUsed}/${totalFrames}` : totalFrames.toLocaleString()}</strong></div>
              <div><span>Visible pts</span><strong>{formatCount(displayedPointCount)}</strong></div>
              <div><span>Generated</span><strong>{formatCount(project.pointCount)}</strong></div>
              <div><span>Triangles</span><strong>{formatCount(project.meshTriangleCount)}</strong></div>
              <div title={project.confidenceDetail ?? 'Available after a successful build'}><span>Confidence</span><strong class={`confidence ${confidenceClass(project.confidenceScore)}`}>{project.confidenceScore !== undefined ? `${project.confidenceScore}%` : '—'}</strong></div>
              <div><span>Voxel</span><strong>{project.settings.voxelSizeMm} mm</strong></div>
            </div>
          </div>
        </div>

        <div class="viewer-wrap">
          <PointCloudPreview
            points={previewPoints}
            packedFrame={!capturing && viewMode === 'live' && sensorSessionEnabled && sensor?.sensorConnected ? packedPreviewFrame : null}
            {processing}
            live={capturing || (viewMode === 'live' && Boolean(sensor?.sensorConnected))}
            {pointSize}
            opacity={pointOpacity}
            {showColors}
            renderMode={viewMode === 'preview' && !processing ? previewRenderMode : 'points'}
            mesh={previewMesh}
            {cameraFrames}
            showCameraFrames={showCameraFrames && viewMode === 'preview' && !processing}
            floorPickMode={floorPickMode && viewMode === 'preview' && !processing}
            anchorPickMode={anchorPickMode && canEdit}
            cloudTransform={viewerTransform}
            gizmoAnchor={effectiveGizmoAnchor}
            editMode={editMode && canEdit && !anchorPickMode}
            {gizmoMode}
            onFloorDetected={setFloorTransform}
            onFloorMessage={(value) => message = value}
            onAnchorPicked={setGizmoAnchor}
            onTransformChanged={handleGizmoTransform}
            onTransformCommitted={commitGizmoTransform}
          />
        </div>

        <div class:with-progress={processing} class="status-strip">
          {#if processing}
            <div class="job-feedback">
              <div class="job-heading">
                <span class="status-light busy"></span>
                <div>
                  <span class="job-kicker">{activeJob?.pipeline === 'media_gaussian' ? 'MEDIA SPLAT JOB' : 'RGB-D ARTIFACT JOB'}</span>
                  <strong>{buildStage.label}</strong>
                </div>
                <span class="job-stage-count">Stage {buildStage.current} / {buildStage.total}</span>
              </div>
              <div class="job-detail" title={buildDetail}>{buildDetail}</div>
              <div class="job-progress-grid">
                <span>Overall</span>
                <div class="progress-track overall"><i style={`width: ${Math.max(2, overallBuildProgress * 100)}%`}></i></div>
                <strong>{Math.round(overallBuildProgress * 100)}%</strong>
                <span>Stage</span>
                <div class="progress-track stage"><i style={`width: ${Math.max(2, currentStageProgress * 100)}%`}></i></div>
                <strong>{Math.round(currentStageProgress * 100)}%</strong>
              </div>
              <div class="job-backend">{buildBackend}</div>
            </div>
            <div class="job-timing">
              <span>Estimated left</span>
              <strong>{formatEtaValue(totalBuildEta)}</strong>
              <small>{currentStageEta != null ? `${formatEtaValue(currentStageEta)} this stage` : reconstruction?.elapsedSeconds != null ? `${reconstruction.elapsedSeconds}s elapsed` : 'Measuring throughput'}</small>
            </div>
            <button class="button process cancel-job" on:click={processCloud} disabled={busy || capturing || !activeJob}>{activeJob?.status === 'cancelling' ? 'Cancelling…' : 'Cancel job'}</button>
          {:else}
            <div class="status-copy">
              <div class="status-message"><span class:busy={busy} class="status-light"></span>{message}</div>
            </div>
            <button class="button process" on:click={processCloud} disabled={busy || capturing || (!activeJob?.resumable && !canBuildArtifacts)}>{activeJob?.resumable && ['failed', 'cancelled'].includes(activeJob.status) ? 'Resume splat' : 'Build artifacts'}</button>
          {/if}
        </div>
      </section>

      <aside class="settings panel">
        <div class="editor-mode-header">
          <div><div class="eyebrow">Editor mode</div><h2>{viewMode === 'live' ? 'Live capture' : 'Model preview'}</h2></div>
          <div class="view-switch panel-switch" aria-label="Editor mode">
            <button class:active={viewMode === 'live'} disabled={processing} on:click={() => showView('live')}>Live</button>
            <button class:active={viewMode === 'preview'} disabled={capturing || (project.processingStatus !== 'complete' && !processing)} on:click={() => showView('preview')}>Preview</button>
          </div>
        </div>
        <p>{viewMode === 'live' ? 'Connect and tune the sensor, then configure how frames are captured.' : processing ? 'The model preview updates as captured phases are reconstructed.' : 'Inspect, orient, render, and export the reconstructed model.'}</p>

        {#if viewMode === 'live'}
        <div class="section-divider first-section"><span>Device</span></div>

        <div class="setting-group">
          <div class="label-row"><label for="sensor-device">Available sensor</label><button class="refresh-sensors" disabled={busy || capturing || processing || discoveryInFlight} on:click={() => void scanSensors()}>{discoveryInFlight ? 'Scanning…' : 'Scan sensors'}</button></div>
          <select id="sensor-device" disabled={busy || capturing || processing || discoveryInFlight} bind:value={selectedSensorOption} on:change={sensorDeviceChanged}>
            {#if selectedSensorOption && selectedSensorOption !== networkFemtoOption && !sensorChoices.some((choice) => choice.id === selectedSensorOption)}
              <option value={selectedSensorOption}>{selectedSensorName} · unavailable</option>
            {/if}
            {#each sensorChoices as choice}
              <option value={choice.id}>{sensorOptionLabel(choice)}</option>
            {/each}
            <option value={networkFemtoOption}>Orbbec Femto Mega · Network IP…</option>
          </select>
        </div>
        {#if project.settings.sensorKind === 'femto_mega' && project.settings.sensorConnection === 'network'}
          <div class="setting-group"><label for="sensor-address">Camera IP address</label><input id="sensor-address" class="text-input" disabled={busy || capturing || processing} type="text" inputmode="decimal" placeholder="192.168.1.10 or IP:port" bind:value={project.settings.sensorAddress} on:change={sensorAddressChanged} /></div>
        {/if}
        {#if project.settings.sensorKind !== 'kinect_v2'}
          <div class="setting-group">
            <label for="depth-fov">Depth field of view</label>
            <div class="segmented two-options" id="depth-fov">
              <button disabled={busy || capturing || processing} class:active={project.settings.depthFieldOfView === 'narrow'} on:click={() => setDepthFieldOfView('narrow')}>Narrow</button>
              <button disabled={busy || capturing || processing} class:active={project.settings.depthFieldOfView === 'wide'} on:click={() => setDepthFieldOfView('wide')}>Wide</button>
            </div>
          </div>
          <div class="setting-group">
            <label for="depth-binning">Depth binning</label>
            <div class="segmented two-options" id="depth-binning">
              <button disabled={busy || capturing || processing} class:active={!project.settings.depthBinned} on:click={() => setDepthBinned(false)}>Unbinned</button>
              <button disabled={busy || capturing || processing} class:active={project.settings.depthBinned} on:click={() => setDepthBinned(true)}>2×2 binned</button>
            </div>
            <p class="setting-note">
              {project.settings.depthFieldOfView === 'wide'
                ? project.settings.depthBinned ? '512×512 at 30 fps' : '1024×1024 at 15 fps'
                : project.settings.depthBinned ? '320×288 at 30 fps' : '640×576 at 30 fps'}
            </p>
          </div>
          <label class="toggle-row sensor-toggle"><input disabled={busy || capturing || processing} type="checkbox" bind:checked={project.settings.useImu} on:change={scheduleProjectSettingsSave} /><span>Use IMU to aid tracking</span></label>
        {/if}
        <div class="section-divider"><span>Capture settings</span></div>
        <div class="setting-group"><label for="environment">Environment</label><select id="environment" disabled={busy || capturing || processing} bind:value={project.settings.environment} on:change={scheduleProjectSettingsSave}><option value="indoor">Indoor</option><option value="outdoor_low_light">Outdoor — night / sunset</option></select></div>
        <div class="setting-group range-group"><div class="label-row"><label for="depth">Maximum depth</label><output>{project.settings.maxDepthM.toFixed(1)} m</output></div><input id="depth" disabled={busy || capturing || processing} type="range" min="1.5" max="8" step="0.1" bind:value={project.settings.maxDepthM} on:input={scheduleProjectSettingsSave} /><div class="range-labels"><span>1.5 m</span><span>8.0 m</span></div></div>
        <div class="setting-group range-group"><div class="label-row"><label for="voxel">Point spacing</label><output>{project.settings.voxelSizeMm} mm</output></div><input id="voxel" disabled={busy || capturing || processing} type="range" min="1" max="40" step="1" bind:value={project.settings.voxelSizeMm} on:input={scheduleProjectSettingsSave} /><div class="range-labels"><span>1 mm detail</span><span>Lightweight</span></div></div>
        <div class="setting-group"><label for="fps">Saved-frame rate</label><div class="segmented" id="fps">{#each [5, 10, 15] as fps}<button disabled={busy || capturing || processing} class:active={project.settings.captureFps === fps} on:click={() => setCaptureFps(fps)}>{fps} fps</button>{/each}</div></div>

        <div class="section-divider"><span>Artifact build</span></div>
        {#if completedPhases > 0 && hasMediaSources}
          <div class="setting-group"><span class="control-label">Input source</span><div class="segmented two-options"><button class:active={sourceMode === 'rgbd'} on:click={() => sourceMode = 'rgbd'} disabled={processing}>RGB-D</button><button class:active={sourceMode === 'media'} on:click={() => sourceMode = 'media'} disabled={processing}>Photos / video</button></div></div>
        {/if}
        <div class="artifact-targets">
          <label class="toggle-row"><input type="checkbox" bind:checked={buildPointCloud} disabled={processing || sourceMode === 'media'} /><span>Point cloud</span></label>
          <label class="toggle-row"><input type="checkbox" bind:checked={buildTexturedMesh} disabled={processing || sourceMode === 'media'} /><span>Native-RGB mesh</span></label>
          <label class="toggle-row"><input type="checkbox" bind:checked={buildGaussianSplat} disabled={processing || sourceMode === 'media'} /><span>Gaussian splat (CUDA)</span></label>
        </div>
        {#if buildGaussianSplat || sourceMode === 'media'}
          <div class="setting-group range-group"><div class="label-row"><label for="splat-iterations">Splat iterations</label><output>{splatIterations.toLocaleString()}</output></div><input id="splat-iterations" type="range" min="5000" max="60000" step="5000" bind:value={splatIterations} disabled={processing} /></div>
          <p class:splat-ready={runtime?.splatWorkerAvailable} class="runtime-diagnostic">{runtime?.splatStatus ?? 'Checking optional CUDA runtime…'}{runtime?.splatWorkerAvailable ? ' · CUDA mixed precision enabled' : ''}</p>
          {#if sourceMode === 'media'}<p class="setting-note">FFmpeg: {runtime?.ffmpegAvailable ? 'ready' : 'missing'} · COLMAP GPU: {runtime?.colmapAvailable ? 'ready' : 'missing'}</p>{/if}
        {/if}
        {/if}

        <div class="section-divider"><span>Rendering</span></div>
        {#if viewMode === 'preview'}
          <div class="setting-group">
            <label for="preview-rendering">Render model as</label>
            <div class="segmented two-options" id="preview-rendering">
              <button disabled={processing} class:active={previewRenderMode === 'points'} on:click={() => setPreviewRenderMode('points')}>Points</button>
              <button disabled={processing || !previewMesh} class:active={previewRenderMode === 'mesh'} on:click={() => setPreviewRenderMode('mesh')}>Mesh</button>
            </div>
            {#if !processing && !previewMesh}<p class="setting-note">A textured mesh becomes available after a successful build.</p>{/if}
          </div>
        {/if}
        {#if viewMode === 'live' || previewRenderMode === 'points' || processing}
          <div class="setting-group range-group"><div class="label-row"><label for="point-size">Point size</label><output>{pointSize.toFixed(3)}</output></div><input id="point-size" type="range" min="0.005" max="0.08" step="0.003" bind:value={pointSize} on:input={scheduleVisualizationSave} /></div>
        {/if}
        <div class="setting-group range-group"><div class="label-row"><label for="opacity">Opacity</label><output>{Math.round(pointOpacity * 100)}%</output></div><input id="opacity" type="range" min="0.1" max="1" step="0.05" bind:value={pointOpacity} on:input={scheduleVisualizationSave} /></div>
        <label class="toggle-row"><input type="checkbox" bind:checked={showColors} on:change={scheduleVisualizationSave} /><span>Show captured colors</span></label>
        {#if viewMode === 'preview'}
          <label class="toggle-row camera-toggle"><input type="checkbox" disabled={cameraFrames.length === 0 || processing} bind:checked={showCameraFrames} on:change={scheduleVisualizationSave} /><span>Show capture cameras{cameraFrames.length ? ` (${cameraFrames.length})` : ''}</span></label>
        {/if}

        {#if viewMode === 'preview'}
        <div class="section-divider"><span>Model manipulation</span></div>
        <button class:active={editMode} class="tool-button" disabled={!canEdit} on:click={() => { editMode = !editMode; floorPickMode = false; anchorPickMode = false; }}>{editMode ? 'Exit edit mode' : 'Edit with gizmo'}</button>
        {#if editMode && canEdit}
          <div class="gizmo-modes"><button class:active={gizmoMode === 'translate'} aria-keyshortcuts="W" title="Move gizmo (W)" on:click={() => setGizmoMode('translate')}>Move (W)</button><button class:active={gizmoMode === 'rotate'} aria-keyshortcuts="E" title="Rotate gizmo (E)" on:click={() => setGizmoMode('rotate')}>Rotate (E)</button><button class:active={gizmoMode === 'scale'} aria-keyshortcuts="R" title="Scale gizmo (R)" on:click={() => setGizmoMode('scale')}>Scale (R)</button></div>
        {/if}
        {#if canEdit}
          <div class="anchor-editor">
            <div class="label-row"><span>Gizmo anchor</span><button on:click={centerGizmoAnchor}>Center on mesh</button></div>
            <div class="anchor-grid">
              {#each ['X', 'Y', 'Z'] as axis, index}
                <label><span>{axis}</span><input type="number" step="0.05" value={effectiveGizmoAnchor[index]} on:input={(event) => updateGizmoAnchor(index, Number(event.currentTarget.value))} /></label>
              {/each}
            </div>
            <button class:active={anchorPickMode} class="tool-button anchor-pick" on:click={toggleAnchorPick}>{anchorPickMode ? 'Cancel anchor pick' : 'Set anchor on mesh'}</button>
          </div>
        {/if}
        <button class:active={floorPickMode} class="tool-button orientation-action" disabled={!canEdit} on:click={() => { floorPickMode = !floorPickMode; editMode = false; anchorPickMode = false; }}>⌖ {floorPickMode ? 'Cancel floor pick' : 'Pick floor point'}</button>
        <button class="tool-button orientation-action" disabled={!canEdit} on:click={alignRoomAxes}>Align room axes</button>
        <div class="axis-actions"><button disabled={!canEdit} on:click={() => flipAxis('X')}>Flip X</button><button disabled={!canEdit} on:click={() => flipAxis('Y')}>Flip Y</button><button disabled={!canEdit} on:click={() => flipAxis('Z')}>Flip Z</button></div>
        <button class="tool-button subtle" on:click={resetTransform} disabled={!canEdit}>Reset orientation</button>
        <button class="tool-button export-transform" on:click={applyTransformToExport} disabled={busy || project.processingStatus !== 'complete'}>Apply pose to model exports</button>

        <div class="section-divider"><span>Export</span></div>
        <div class:stale={project.artifacts.gaussianSplat?.stale} class="export-card">
          <span>Canonical 3DGS PLY</span>
          <strong>{project.artifacts.gaussianSplat ? project.artifacts.gaussianSplat.metric ? 'Metric Gaussian splat' : 'Arbitrary-scale Gaussian splat' : 'Not built'}</strong>
          <small>Exports the Gaussian PLY plus manifest and GameObject-level coordinate transform. Viewer transforms are not baked into covariance.</small>
          <button class="tool-button export-ply" on:click={exportGaussianSplatAction} disabled={busy || !project.artifacts.gaussianSplat || project.artifacts.gaussianSplat.stale}>Export Gaussian splat...</button>
        </div>
        <div class="export-card">
          <span>RGB-reprojected OBJ</span>
          <strong>{formatCount(project.meshTriangleCount)} textured triangles</strong>
          <small>Exports OBJ + MTL + PNG. Surface UVs point back into selected captured RGB frames.</small>
          <button class="tool-button export-ply" on:click={exportTexturedMeshAction} disabled={busy || project.processingStatus !== 'complete' || !project.meshOutputPath}>Export textured mesh...</button>
        </div>
        <div class="export-card">
          <span>Unity-ready PLY</span>
          <strong>{formatCount(project.pointCount)} colored points</strong>
          <small>The saved copy corrects the X axis for Unity. Your project PLY stays unchanged.</small>
          <button class="tool-button export-ply" on:click={exportPlyAction} disabled={busy || project.processingStatus !== 'complete'}>Export PLY...</button>
        </div>
        {/if}
      </aside>
    </section>
  {:else if initializationError}
    <div class="loading">{initializationError}</div>
  {:else}
    <div class="loading">Preparing scanner workspace…</div>
  {/if}
</main>
