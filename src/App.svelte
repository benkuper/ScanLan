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
    discardArtifactJob,
    exportGaussianSplat,
    exportPly,
    exportTexturedMesh,
    importMediaSource,
    latestArtifactJob,
    loadCameraFrames,
    loadGaussianSplat,
    loadLivePreviewFrame,
    loadLiveReconstructionMesh,
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
    MeshViewMode,
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
  let previewSplat: Uint8Array | null = null;
  let splatPreviewError = '';
  let splatPreviewLoading = false;
  let previewAssetLoading: 'points' | 'mesh' | 'splat' | null = null;
  let splatPreviewInFlight = false;
  let lastSplatPreviewPoll = 0;
  let lastSplatPreviewSignature = '';
  let liveSplatUpdatedAt: number | null = null;
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
  let lastLiveMeshFrame = 0;
  let lastLiveMeshPoll = 0;
  let lastStatusPoll = 0;
  let message = 'Select Scan sensors when you are ready to connect a depth sensor.';
  type WorkspaceMode = 'device' | 'capture' | 'media' | 'process' | 'render' | 'export';
  const workflowModes: { id: WorkspaceMode; step: string; label: string; description: string }[] = [
    { id: 'device', step: '01', label: 'Input device', description: 'Connect & calibrate' },
    { id: 'capture', step: '02', label: 'Capture', description: 'Record RGB-D phases' },
    { id: 'media', step: '03', label: 'Media', description: 'Review all inputs' },
    { id: 'process', step: '04', label: 'Process', description: 'Build 3D artifacts' },
    { id: 'render', step: '05', label: 'Edit', description: 'Inspect & adjust' },
    { id: 'export', step: '06', label: 'Export', description: 'Choose deliverables' }
  ];
  let workspaceMode: WorkspaceMode = 'device';
  let viewMode: 'live' | 'preview' = 'live';
  let previewRenderMode: 'points' | 'mesh' | 'splat' = 'points';
  let meshViewMode: MeshViewMode = 'surface';
  let lightDirection: [number, number, number] = [0.45, 0.8, 0.35];
  let lightEditMode = false;
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
  let loadedResultPreviewSignature = '';
  let resultPreviewRequest: { signature: string; promise: Promise<void> } | null = null;

  const transformStorageKey = (projectId: string) => `scanlan-cloud-transform:${projectId}`;
  const sourceModeStorageKey = (projectId: string) => `scanlan-source-mode:${projectId}`;
  const workspaceModeStorageKey = (projectId: string) => `scanlan-workspace-mode:${projectId}`;
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
        previewRenderMode: 'points' | 'mesh' | 'splat';
        meshViewMode: MeshViewMode;
        lightDirection: [number, number, number];
        gizmoMode: 'translate' | 'rotate' | 'scale';
      }>;
      if (typeof value.pointSize === 'number') pointSize = value.pointSize;
      if (typeof value.pointOpacity === 'number') pointOpacity = value.pointOpacity;
      if (typeof value.showColors === 'boolean') showColors = value.showColors;
      if (typeof value.showCameraFrames === 'boolean') showCameraFrames = value.showCameraFrames;
      if (value.previewRenderMode === 'points' || value.previewRenderMode === 'mesh' || value.previewRenderMode === 'splat') {
        previewRenderMode = value.previewRenderMode;
      }
      if (value.meshViewMode === 'surface' || value.meshViewMode === 'surface-wireframe' || value.meshViewMode === 'wireframe' || value.meshViewMode === 'shaded') {
        meshViewMode = value.meshViewMode;
      }
      if (Array.isArray(value.lightDirection) && value.lightDirection.length === 3 && value.lightDirection.every((component) => typeof component === 'number' && Number.isFinite(component))) {
        const [x, y, z] = value.lightDirection;
        if (Math.hypot(x, y, z) > 0.0001) lightDirection = [x, y, z];
      }
      if (value.gizmoMode === 'translate' || value.gizmoMode === 'rotate' || value.gizmoMode === 'scale') {
        gizmoMode = value.gizmoMode;
      }
    } catch {
      localStorage.removeItem(visualizationStorageKey);
    }
  }

  function persistVisualizationPreferences() {
    localStorage.setItem(visualizationStorageKey, JSON.stringify({ pointSize, pointOpacity, showColors, showCameraFrames, previewRenderMode, meshViewMode, lightDirection, gizmoMode }));
  }

  const scheduleVisualizationSave = () => queueMicrotask(persistVisualizationPreferences);
  const scheduleProjectSettingsSave = () => queueMicrotask(queueProjectSettingsSave);

  function setGizmoMode(mode: 'translate' | 'rotate' | 'scale') {
    gizmoMode = mode;
    lightEditMode = false;
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
      if (!editMode && !lightEditMode) return;
      event.preventDefault();
      editMode = false;
      lightEditMode = false;
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

  function setPreviewRenderMode(mode: 'points' | 'mesh' | 'splat') {
    previewRenderMode = mode;
    if (mode !== 'mesh') lightEditMode = false;
    if (mode === 'splat') {
      floorPickMode = false;
      anchorPickMode = false;
      editMode = false;
    }
    persistVisualizationPreferences();
  }

  function setMeshViewMode(mode: MeshViewMode) {
    meshViewMode = mode;
    if (mode !== 'shaded') lightEditMode = false;
    persistVisualizationPreferences();
  }

  function updateLightDirection(axis: number, value: number) {
    if (!Number.isFinite(value)) return;
    const next = [...lightDirection] as [number, number, number];
    next[axis] = value;
    if (Math.hypot(...next) < 0.0001) return;
    lightDirection = next;
    scheduleVisualizationSave();
  }

  function handleLightDirectionChanged(direction: [number, number, number]) {
    lightDirection = direction.map((component) => Number(component.toFixed(4))) as [number, number, number];
    scheduleVisualizationSave();
  }

  function resetLightDirection() {
    lightDirection = [0.45, 0.8, 0.35];
    persistVisualizationPreferences();
  }

  function toggleLightEdit() {
    lightEditMode = !lightEditMode;
    editMode = false;
    floorPickMode = false;
    anchorPickMode = false;
    message = lightEditMode ? 'Drag the light gizmo to change the shaded mesh lighting.' : 'Light direction updated.';
  }

  async function selectSplatPreview() {
    const artifact = project?.artifacts.gaussianSplat;
    if (previewSplat) {
      setPreviewRenderMode('splat');
      return;
    }
    if (processing || !project || !artifact || artifact.stale) {
      message = processing && activeBuildIncludesSplat
        ? splatPreviewError || (gaussianTrainingStage
          ? 'Gaussian training is active. Waiting for the trainer to publish its first live snapshot.'
          : 'The Gaussian preview becomes available when the build reaches its training stage.')
        : splatPreviewLoading
        ? 'The compact Gaussian preview is still decoding and uploading to the GPU.'
        : splatPreviewError || (artifact?.stale
        ? 'The Gaussian splat is stale. Train it again before previewing.'
        : artifact
          ? 'The Gaussian splat could not be loaded. Rebuild it and try again.'
          : 'No Gaussian splat artifact exists yet. Choose Gaussian splat below, then select Train Gaussian splat.');
      return;
    }
    setPreviewRenderMode('splat');
    previewAssetLoading = 'splat';
    splatPreviewLoading = true;
    splatPreviewError = '';
    message = 'Loading the optimized Gaussian preview…';
    try {
      const nextSplat = await loadGaussianSplat(project.path);
      previewSplat = nextSplat;
      lastSplatPreviewSignature = splatPreviewSignature(nextSplat);
    } catch (error) {
      splatPreviewError = error instanceof Error ? error.message : String(error);
      message = splatPreviewError;
    } finally {
      splatPreviewLoading = false;
      if (previewAssetLoading === 'splat') previewAssetLoading = null;
    }
  }

  async function selectMeshPreview() {
    if (previewMesh) {
      setPreviewRenderMode('mesh');
      return;
    }
    if (!project?.meshOutputPath || previewAssetLoading === 'mesh') return;
    setPreviewRenderMode('mesh');
    previewAssetLoading = 'mesh';
    message = 'Loading the optimized mesh preview…';
    try {
      previewMesh = await loadPreviewMesh(project.path);
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    } finally {
      if (previewAssetLoading === 'mesh') previewAssetLoading = null;
    }
  }

  function setSourceMode(mode: 'rgbd' | 'media') {
    sourceMode = mode;
    if (project) localStorage.setItem(sourceModeStorageKey(project.id), mode);
    if (mode === 'media') {
      buildPointCloud = false;
      buildTexturedMesh = false;
      buildGaussianSplat = true;
      message = 'Photos / video selected. Gaussian splat training is enabled.';
    }
    if (viewMode === 'preview' && !processing) void refreshResultPreview();
  }

  function storeWorkspaceMode(mode: WorkspaceMode) {
    workspaceMode = mode;
    if (project) localStorage.setItem(workspaceModeStorageKey(project.id), mode);
  }

  async function selectWorkspaceMode(mode: WorkspaceMode) {
    if (capturing && mode !== 'capture') {
      message = 'Stop the active capture phase before changing workspace.';
      return;
    }
    if (processing && mode !== 'process' && mode !== 'render') {
      message = 'While a build is running, use Process for job progress or Edit for the live 3D result.';
      return;
    }

    storeWorkspaceMode(mode);
    if (mode === 'device' || mode === 'capture') {
      if (mode === 'capture') setSourceMode('rgbd');
      await showView('live');
    } else if (mode === 'media') {
      setSourceMode('media');
    } else if (mode === 'process') {
      if (completedPhases === 0 && hasMediaSources) setSourceMode('media');
      else if (!hasMediaSources && completedPhases > 0) setSourceMode('rgbd');
    } else if (mode === 'render' && processing) {
      viewMode = 'preview';
      if (previewSplat) previewRenderMode = 'splat';
      message = previewSplat
        ? 'Showing the latest Gaussian snapshot published by the active trainer.'
        : 'Edit is following the active build. The Gaussian view will appear as soon as the trainer publishes its first snapshot.';
    } else if (project?.processingStatus === 'complete') {
      await showView('preview');
    } else {
      message = mode === 'export'
        ? 'Build at least one artifact before exporting.'
        : 'Build an artifact first, then return here to inspect it.';
    }
  }

  function workflowBadge(mode: WorkspaceMode) {
    if (!project) return '';
    if (mode === 'device') return sensor?.sensorConnected ? 'Connected' : sensorSessionEnabled ? 'Needs attention' : 'Not connected';
    if (mode === 'capture') return `${completedPhases} phase${completedPhases === 1 ? '' : 's'}`;
    if (mode === 'media') {
      const mediaItemCount = completedPhases + project.mediaSources.length;
      return `${mediaItemCount} item${mediaItemCount === 1 ? '' : 's'}`;
    }
    if (mode === 'process') return processing ? `${Math.round(overallBuildProgress * 100)}%` : canBuildArtifacts ? 'Ready' : 'Needs input';
    if (mode === 'render') {
      if (processing) return previewSplat ? `${Math.floor(previewSplat.byteLength / 32).toLocaleString()} live` : previewPoints.length > 0 ? 'Live points' : 'Waiting';
      return artifactCount > 0 ? `${artifactCount} view${artifactCount === 1 ? '' : 's'}` : 'No model';
    }
    return exportCount > 0 ? `${exportCount} ready` : 'No output';
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

  function setLiveReconstruction(liveReconstruction: 'off' | 'points' | 'mesh') {
    if (!project) return;
    project = { ...project, settings: { ...project.settings, liveReconstruction } };
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
    lastLiveMeshFrame = 0;
    lastLiveMeshPoll = 0;
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
  $: jobAwaitingDecision = Boolean(activeJob?.resumable && ['failed', 'cancelled'].includes(activeJob.status));
  $: processing = project?.processingStatus === 'processing' || jobRunning;
  $: statusMessage = jobAwaitingDecision
    ? activeJob?.stage === 'interrupted'
      ? 'Build interrupted. Resume from its checkpoint or cancel it.'
      : 'Build stopped. Resume from its checkpoint or cancel it.'
    : message;
  $: activeBuildIncludesSplat = Boolean(processing && activeJob?.targets.includes('gaussianSplat'));
  $: gaussianTrainingStage = Boolean(activeBuildIncludesSplat && activeJob && stageKey(activeJob.stage) === 'splat');
  $: liveSplatCount = previewSplat ? Math.floor(previewSplat.byteLength / 32) : 0;
  $: liveSplatState = previewSplat
    ? 'ready'
    : splatPreviewError
      ? 'error'
      : splatPreviewInFlight
        ? 'loading'
        : gaussianTrainingStage
          ? 'waiting'
          : 'pending';
  $: selectedSensorName = project?.settings.sensorKind === 'azure_kinect'
    ? 'Azure Kinect DK'
    : project?.settings.sensorKind === 'femto_mega'
      ? 'Orbbec Femto Mega'
      : 'Kinect v2';
  $: canEdit = viewMode === 'preview' && previewRenderMode !== 'splat' && project?.processingStatus === 'complete' && previewPoints.length > 0;
  $: viewerTransform = viewMode === 'preview' ? cloudTransform : identityTransform;
  $: effectiveGizmoAnchor = gizmoAnchor ?? pointCloudCenter(previewPoints);
  $: completedPhases = project?.phases.filter((phase) => phase.status === 'complete').length ?? 0;
  $: hasMediaSources = (project?.mediaSources.length ?? 0) > 0;
  $: canBuildArtifacts = sourceMode === 'rgbd' ? completedPhases > 0 : selectedMediaSourceIds.length > 0;
  $: artifactCount = project
    ? [project.artifacts.pointCloud, project.artifacts.texturedMesh, project.artifacts.gaussianSplat].filter((artifact) => artifact && !artifact.stale).length
    : 0;
  $: exportCount = project
    ? Number(project.processingStatus === 'complete' && Boolean(project.outputPath))
      + Number(Boolean(project.meshOutputPath))
      + Number(Boolean(project.artifacts.gaussianSplat && !project.artifacts.gaussianSplat.stale))
    : 0;
  $: workspaceTitle = workspaceMode === 'device'
    ? `${selectedSensorName} input`
    : workspaceMode === 'capture'
      ? capturing ? `Recording ${project?.phases.at(-1)?.name ?? 'capture phase'}` : 'Live capture view'
      : workspaceMode === 'media'
        ? 'Recorded & imported media'
        : workspaceMode === 'process'
          ? processing ? 'Building 3D artifacts' : 'Artifact build plan'
          : workspaceMode === 'render'
            ? previewRenderMode === 'splat' ? 'Gaussian splat viewer' : previewRenderMode === 'mesh' ? 'Textured mesh viewer' : 'Point cloud viewer'
            : 'Export preview';
  $: workspaceKicker = workspaceMode === 'device'
    ? 'INPUT DEVICE MONITOR'
    : workspaceMode === 'capture'
      ? capturing ? 'CAPTURE IN PROGRESS' : 'CAPTURE WORKSPACE'
      : workspaceMode === 'media'
        ? 'PROJECT MEDIA LIBRARY'
        : workspaceMode === 'process'
          ? processing ? 'PROCESSING NOW' : 'PROCESSING SETUP'
          : workspaceMode === 'render'
            ? processing ? 'LIVE BUILD EDITOR' : 'EDIT WORKSPACE'
            : 'EXPORT WORKSPACE';
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

  const formatSnapshotTime = (value: number | null) => value == null
    ? 'Not published yet'
    : new Intl.DateTimeFormat([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(value);

  function splatPreviewSignature(bytes: Uint8Array) {
    let hash = 2166136261;
    const stride = Math.max(1, Math.floor(bytes.byteLength / 97));
    for (let index = 0; index < bytes.byteLength; index += stride) {
      hash = Math.imul(hash ^ bytes[index], 16777619);
    }
    hash = Math.imul(hash ^ (bytes.at(-1) ?? 0), 16777619);
    return `${bytes.byteLength}:${hash >>> 0}`;
  }

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
    if (stage.includes('mesh') || stage.includes('textur')) return 'mesh';
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
    if (wantsSplat) stages.push({ key: 'dataset', label: 'Preparing posed frames', weight: 0.08 });
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
    const requestedProjectPath = project.path;
    const splatArtifact = project.artifacts.gaussianSplat;
    const signature = JSON.stringify([
      requestedProjectPath,
      project.outputPath,
      project.meshOutputPath,
      project.artifacts.pointCloud?.updatedAt,
      project.artifacts.texturedMesh?.updatedAt,
      splatArtifact?.updatedAt
    ]);
    if (signature === loadedResultPreviewSignature) return;
    if (resultPreviewRequest?.signature === signature) return resultPreviewRequest.promise;

    const request = (async () => {
      const preferMediaSplat = sourceMode === 'media' && Boolean(splatArtifact && !splatArtifact.stale);
      if (preferMediaSplat && previewRenderMode !== 'splat') {
        setPreviewRenderMode('splat');
      } else if (previewRenderMode === 'mesh' && !project.meshOutputPath) {
        setPreviewRenderMode(splatArtifact && !splatArtifact.stale ? 'splat' : 'points');
      } else if (previewRenderMode === 'splat' && (!splatArtifact || splatArtifact.stale)) {
        setPreviewRenderMode(project.meshOutputPath ? 'mesh' : 'points');
      }
      const shouldLoadMesh = !preferMediaSplat && Boolean(project.meshOutputPath) && previewRenderMode === 'mesh';
      const shouldLoadSplat = Boolean(splatArtifact && !splatArtifact.stale)
        && (preferMediaSplat || previewRenderMode === 'splat');
      const requestedAsset = shouldLoadMesh ? 'mesh' : shouldLoadSplat ? 'splat' : 'points';
      previewAssetLoading = requestedAsset;
      splatPreviewLoading = shouldLoadSplat;
      let nextSplatError = '';
      const [nextPoints, nextCameraFrames, nextMesh, nextSplat] = await Promise.all([
        preferMediaSplat ? Promise.resolve([]) : loadPreview(project.path).catch(() => []),
        preferMediaSplat ? Promise.resolve([]) : loadCameraFrames(project.path).catch(() => []),
        shouldLoadMesh ? loadPreviewMesh(project.path).catch(() => null) : Promise.resolve(previewMesh),
        shouldLoadSplat
          ? loadGaussianSplat(project.path).catch((error: unknown) => {
              nextSplatError = error instanceof Error ? error.message : String(error);
              return null;
            })
          : Promise.resolve(null)
      ]);
      if (project?.path !== requestedProjectPath) {
        splatPreviewLoading = false;
        if (previewAssetLoading === requestedAsset) previewAssetLoading = null;
        return;
      }
      previewPoints = nextPoints;
      if (nextPoints.length === 0) resetPackedPreview();
      cameraFrames = nextCameraFrames;
      previewMesh = nextMesh;
      previewSplat = nextSplat;
      lastSplatPreviewSignature = nextSplat ? splatPreviewSignature(nextSplat) : '';
      splatPreviewError = nextSplatError;
      splatPreviewLoading = false;
      if (previewAssetLoading === requestedAsset) previewAssetLoading = null;
      if (preferMediaSplat && nextSplat) {
        setPreviewRenderMode('splat');
      } else if (shouldLoadMesh && !nextMesh) {
        setPreviewRenderMode(nextSplat ? 'splat' : 'points');
      } else if (shouldLoadSplat && previewRenderMode === 'splat' && !nextSplat) {
        setPreviewRenderMode(nextMesh ? 'mesh' : 'points');
      } else if (nextPoints.length === 0 && nextSplat) {
        setPreviewRenderMode('splat');
      }
      const loadedEverything = (!shouldLoadMesh || Boolean(nextMesh)) && (!shouldLoadSplat || Boolean(nextSplat));
      if (loadedEverything) loadedResultPreviewSignature = signature;
    })();
    resultPreviewRequest = { signature, promise: request };
    try {
      await request;
    } finally {
      if (resultPreviewRequest?.promise === request) resultPreviewRequest = null;
    }
  }

  async function refreshProcessingSplatPreview(job: ArtifactJob) {
    if (!project || job.status !== 'running' || stageKey(job.stage) !== 'splat' || splatPreviewInFlight) return;
    const now = performance.now();
    if (now - lastSplatPreviewPoll < 2000) return;
    lastSplatPreviewPoll = now;
    splatPreviewInFlight = true;
    try {
      const nextSplat = await loadGaussianSplat(project.path);
      if (nextSplat.byteLength >= 32) {
        const signature = splatPreviewSignature(nextSplat);
        if (signature !== lastSplatPreviewSignature) {
          previewSplat = nextSplat;
          lastSplatPreviewSignature = signature;
          liveSplatUpdatedAt = Date.now();
        }
        previewRenderMode = 'splat';
        splatPreviewError = '';
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      splatPreviewError = detail.includes('has not published its first live splat preview') || detail.includes('still being published')
        ? ''
        : detail;
    } finally {
      splatPreviewInFlight = false;
    }
  }

  async function refreshSensorStatus() {
    if (!project || statusInFlight || (!sensorSessionEnabled && !capturing && !processing)) return;
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
        if (workspaceMode !== 'capture') storeWorkspaceMode('capture');
        viewMode = 'live';
        if (status.preview.length > 0) previewPoints = status.preview;
        if (status.liveReconstructionActive) {
          message = status.tracking
            ? `Live ${status.liveReconstructionMode === 'mesh' ? 'mesh' : 'point'} reconstruction · ${status.liveIntegratedFrameCount} keyframes fused${status.liveRejectedFrameCount ? ` · ${status.liveRejectedFrameCount} rejected` : ''}`
            : status.liveIntegratedFrameCount > 0
              ? status.trackingStatus
              : 'Initializing live reconstruction from the first valid RGB-D frame…';
        } else {
          message = `Capturing frame ${status.frameCount} · ${status.streamFps.toFixed(1)} saved fps · final placement happens during build`;
        }
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
      if (activeJob && ['queued', 'running', 'cancelling'].includes(activeJob.status)) {
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
          await refreshProcessingSplatPreview(updatedJob);
          if (['complete', 'failed', 'cancelled'].includes(updatedJob.status)) {
            project = await currentProject();
            activeJob = updatedJob.resumable ? updatedJob : null;
            if (updatedJob.status === 'complete') {
              storeWorkspaceMode('render');
              viewMode = 'preview';
              if (project.artifacts.pointCloud || project.artifacts.texturedMesh || project.artifacts.gaussianSplat) {
                await refreshResultPreview();
              }
              if (updatedJob.pipeline === 'media_gaussian' && previewSplat) setPreviewRenderMode('splat');
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
    if (!project || previewInFlight || !sensorSessionEnabled || processing || viewMode !== 'live') return;
    if (capturing && !sensor?.liveReconstructionActive) return;
    previewInFlight = true;
    try {
      const packet = await loadLivePreviewFrame(lastPreviewFrame);
      const frame = decodeLivePreviewFrame(packet);
      const now = performance.now();
      if (frame && frame.frameCount !== lastPreviewFrame) {
        lastPreviewFrame = frame.frameCount;
        packedPreviewFrame = frame;
        if (lastPreviewArrival > 0) {
          const instantFps = Math.min(60, 1000 / Math.max(1, now - lastPreviewArrival));
          previewFps = previewFps > 0 ? previewFps * 0.8 + instantFps * 0.2 : instantFps;
        }
        lastPreviewArrival = now;
      }
      if (capturing && sensor?.liveReconstructionMode === 'mesh' && now - lastLiveMeshPoll >= 450) {
        lastLiveMeshPoll = now;
        const update = await loadLiveReconstructionMesh(lastLiveMeshFrame);
        if (update && update.frameCount !== lastLiveMeshFrame) {
          lastLiveMeshFrame = update.frameCount;
          previewMesh = update.mesh;
        }
      }
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
      storeWorkspaceMode('device');
      cloudTransform = { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] };
      floorPickMode = false;
      editMode = false;
      previewPoints = [];
      resetPackedPreview();
      cameraFrames = [];
      previewMesh = null;
      previewSplat = null;
      splatPreviewError = '';
      splatPreviewLoading = false;
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
        previewMesh = null;
        resetPackedPreview();
        storeWorkspaceMode('capture');
        setSourceMode('rgbd');
        viewMode = 'live';
        message = project.settings.liveReconstruction === 'off'
          ? `Starting ${selectedSensorName} capture…`
          : `Warming ${project.settings.liveReconstruction === 'mesh' ? 'live mesh' : 'live point'} reconstruction before recording…`;
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
      previewSplat = null;
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
      previewSplat = null;
      splatPreviewError = '';
      selectedMediaSourceIds = project.mediaSources.map((source) => source.id);
      setSourceMode('media');
      storeWorkspaceMode('media');
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
      previewSplat = null;
      splatPreviewError = '';
      selectedMediaSourceIds = project.mediaSources.map((source) => source.id);
      setSourceMode('media');
      storeWorkspaceMode('media');
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
      previewSplat = null;
      splatPreviewError = '';
      if (previewRenderMode === 'splat') setPreviewRenderMode(previewMesh ? 'mesh' : 'points');
      message = `${name} removed.`;
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    } finally {
      busy = false;
    }
  }

  async function processCloud() {
    if (!project || busy) return;
    if (activeJob && ['queued', 'running', 'cancelling'].includes(activeJob.status)) {
      try {
        activeJob = await cancelArtifactJob(project.path, activeJob.id);
        message = 'Cancelling artifact workers and saving any splat checkpoint…';
      } catch (error) {
        message = error instanceof Error ? error.message : String(error);
      }
      return;
    }
    if (jobAwaitingDecision) {
      await resumeInterruptedJob();
      return;
    }
    if (activeJob && ['failed', 'cancelled', 'complete'].includes(activeJob.status)) activeJob = null;
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
    if (sourceMode === 'media' && (!runtime?.ffmpegAvailable || !runtime?.colmapAvailable)) {
      message = 'Photo/video builds require the bundled FFmpeg and CUDA COLMAP tools. Run npm run prepare:media and restart debug mode.';
      return;
    }
    storeWorkspaceMode('process');
    if (settingsSaveTimer !== undefined) {
      window.clearTimeout(settingsSaveTimer);
      settingsSaveTimer = undefined;
    }
    try {
      project = await updateProjectSettings(project.path, { ...project.settings });
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
      return;
    }
    busy = true;
    reconstruction = null;
    editMode = false;
    if (targets.includes('gaussianSplat')) {
      previewSplat = null;
      lastSplatPreviewPoll = 0;
      lastSplatPreviewSignature = '';
      liveSplatUpdatedAt = null;
      splatPreviewError = '';
    }
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

  async function resumeInterruptedJob() {
    if (!project || !activeJob || !jobAwaitingDecision || busy) return;
    const jobId = activeJob.id;
    busy = true;
    try {
      storeWorkspaceMode('process');
      lastSplatPreviewPoll = 0;
      splatPreviewError = '';
      activeJob = await resumeArtifactJob(project.path, jobId);
      project = { ...project, processingStatus: 'processing', processingError: undefined, activeJob: activeJob.id };
      message = 'Resuming CUDA splat training from its matching checkpoint…';
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    } finally {
      busy = false;
    }
  }

  async function discardInterruptedJob() {
    if (!project || !activeJob || !jobAwaitingDecision || busy) return;
    if (!window.confirm('Cancel this interrupted build and delete its saved checkpoint? Finished artifacts will stay safe.')) return;
    const jobId = activeJob.id;
    busy = true;
    try {
      await discardArtifactJob(project.path, jobId);
      activeJob = null;
      reconstruction = null;
      previewSplat = null;
      lastSplatPreviewSignature = '';
      liveSplatUpdatedAt = null;
      splatPreviewError = '';
      project = await currentProject();
      message = 'Interrupted build cancelled. You can configure and start a new build.';
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
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
    lightEditMode = false;
    floorPickMode = false;
    if (anchorPickMode) message = 'Click a point on the mesh to place the gizmo anchor.';
  }

  function setGizmoAnchor(anchor: [number, number, number]) {
    gizmoAnchor = [...anchor];
    lightEditMode = false;
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
        const savedSourceMode = localStorage.getItem(sourceModeStorageKey(project.id));
        if (
          (savedSourceMode === 'media' && project.mediaSources.length > 0) ||
          (!savedSourceMode && project.mediaSources.length > 0 && (
            !project.artifacts.gaussianSplat ||
            !project.artifacts.gaussianSplat.metric ||
            project.phases.length === 0
          ))
        ) setSourceMode('media');
        else if (savedSourceMode === 'rgbd' && project.phases.length > 0) setSourceMode('rgbd');
        if (project.activeJob) {
          activeJob = await artifactJobStatus(project.path, project.activeJob).catch(() => null);
        } else {
          const latestJob = await latestArtifactJob(project.path).catch(() => null);
          activeJob = latestJob?.resumable ? latestJob : null;
        }
        const recoveredJob = Boolean(activeJob?.resumable && ['failed', 'cancelled'].includes(activeJob.status));
        if (recoveredJob) setSourceMode(activeJob?.pipeline === 'media_gaussian' ? 'media' : 'rgbd');
        const savedWorkspaceMode = localStorage.getItem(workspaceModeStorageKey(project.id));
        const validSavedMode = workflowModes.some((mode) => mode.id === savedWorkspaceMode)
          ? savedWorkspaceMode as WorkspaceMode
          : null;
        if (recoveredJob || project.processingStatus === 'processing' || (activeJob && ['queued', 'running', 'cancelling'].includes(activeJob.status))) {
          workspaceMode = 'process';
          viewMode = 'preview';
        } else if (validSavedMode) {
          workspaceMode = validSavedMode;
        } else if (project.processingStatus === 'complete') {
          workspaceMode = 'render';
        } else if (project.mediaSources.length > 0 && project.phases.length === 0) {
          workspaceMode = 'media';
        }
        if ((workspaceMode === 'render' || workspaceMode === 'export') && project.processingStatus === 'complete') {
          viewMode = 'preview';
          await refreshResultPreview();
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

<main class:processing-workspace={processing} class="app-shell">
  <header class="topbar">
    <div class="brand">
      <div class="brand-mark" aria-hidden="true"><span></span><span></span><span></span></div>
      <div><h1>ScanLan</h1><p>Spatial capture workspace</p></div>
    </div>
    {#if project}
      <div class="project-crumb"><span>ACTIVE PROJECT</span><strong>{project.name}</strong><small title={project.path}>{project.path}</small></div>
    {/if}
    <div class="top-actions">
      <div class="runtime-pill" class:connected={!processing && sensor?.sensorConnected} class:paused={processing || sensor?.sensorPaused} title={processing ? 'Sensor preview remains paused until the artifact job is fully published' : sensor?.sensorStatus}>
        <span></span>
        <div><small>SENSOR</small><strong>{processing ? 'Paused for build' : discoveryInFlight ? 'Scanning…' : connecting ? 'Connecting…' : sensor?.sensorConnected ? 'Connected' : 'Not connected'}</strong></div>
      </div>
      <button class="button ghost" on:click={newProject} disabled={busy || capturing || processing}>New project</button>
    </div>
  </header>

  {#if project}
    <nav class="workflow-nav" aria-label="Scan workflow">
      {#each workflowModes as mode}
        <button
          class:active={workspaceMode === mode.id}
          class:complete={(mode.id === 'capture' && completedPhases > 0) || (mode.id === 'media' && hasMediaSources) || ((mode.id === 'render' || mode.id === 'export') && artifactCount > 0)}
          aria-current={workspaceMode === mode.id ? 'step' : undefined}
          on:click={() => void selectWorkspaceMode(mode.id)}
        >
          <span class="workflow-step">{mode.step}</span>
          <span class="workflow-copy"><strong>{mode.label}</strong><small>{mode.description}</small></span>
          <span class="workflow-badge">{workflowBadge(mode.id)}</span>
        </button>
      {/each}
    </nav>

    <section class="workspace">
      <aside class="context-panel panel">
        <div class="panel-heading">
          <div class="eyebrow">{workspaceMode === 'device' ? 'SETUP' : workspaceMode === 'capture' ? 'SESSION' : workspaceMode === 'media' ? 'INPUT LIBRARY' : workspaceMode === 'process' ? 'BUILD INPUT' : workspaceMode === 'render' ? 'AVAILABLE VIEWS' : 'DELIVERABLES'}</div>
          <h2>{workspaceMode === 'device' ? 'Device readiness' : workspaceMode === 'capture' ? 'Capture phases' : workspaceMode === 'media' ? 'Media overview' : workspaceMode === 'process' ? 'Choose source' : workspaceMode === 'render' ? processing ? 'Current build result' : 'Current model' : 'Ready to export'}</h2>
        </div>

        {#if workspaceMode === 'device'}
          <div class:connected={sensor?.sensorConnected} class="connection-card">
            <span class="connection-icon">{sensor?.sensorConnected ? '✓' : '1'}</span>
            <div><small>{sensor?.sensorConnected ? 'INPUT ONLINE' : 'ACTION REQUIRED'}</small><strong>{sensor?.sensorConnected ? sensor.sensorName : 'Connect a depth sensor'}</strong><p>{sensor?.sensorConnected ? `${sensor.streamFps.toFixed(1)} fps stream available` : 'Scan USB devices or enter the network camera address.'}</p></div>
          </div>
          <div class="checklist">
            <div class:ready={runtime?.sensorWorkerAvailable}><span>01</span><div><strong>Capture runtime</strong><small>{runtime?.sensorWorkerAvailable ? 'Available' : 'Unavailable'}</small></div></div>
            <div class:ready={sensorChoices.length > 0 || sensor?.sensorConnected}><span>02</span><div><strong>Device discovered</strong><small>{sensorChoices.length > 0 || sensor?.sensorConnected ? 'Sensor found' : 'Scan required'}</small></div></div>
            <div class:ready={sensor?.sensorConnected}><span>03</span><div><strong>Live stream</strong><small>{sensor?.sensorConnected ? 'Ready to capture' : 'Waiting for input'}</small></div></div>
          </div>
          <div class="context-note"><strong>What belongs here</strong><p>Device selection, network address, depth mode, and tracking hardware. Capture quality controls live in the next workspace.</p></div>

        {:else if workspaceMode === 'capture'}
          <div class="phase-list">
            {#if project.phases.length === 0}
              <div class="empty-state compact"><span>01</span><strong>No RGB-D phases yet</strong><p>Connect the sensor, then record a slow pass with overlapping views.</p></div>
            {:else}
              {#each project.phases as phase, index}
                <article class:active={phase.status === 'capturing'} class="phase-card">
                  <div class="phase-index">{String(index + 1).padStart(2, '0')}</div>
                  <div class="phase-copy"><strong>{phase.name}</strong><span>{phase.frameCount} frames · {formatDuration(phase.durationSeconds)}</span><small><i></i>{phase.overlapHint}</small></div>
                  <div class="phase-actions">
                    <div class:capturing={phase.status === 'capturing'} class:failed={phase.status === 'failed'} class="phase-status">{phase.status === 'capturing' ? '●' : phase.status === 'failed' ? '!' : '✓'}</div>
                    {#if phase.status !== 'capturing'}<button class="remove-phase" title={`Remove ${phase.name}`} disabled={busy || processing} on:click={() => removeCaptureAction(phase.id, phase.name)}>×</button>{/if}
                  </div>
                </article>
              {/each}
            {/if}
          </div>
          <button class:stopping={capturing} class="button primary context-primary" on:click={captureAction} disabled={busy || processing || (!capturing && !sensor?.sensorConnected)}><span class="record-dot"></span>{capturing ? 'Stop & save phase' : project.phases.length ? 'Record another phase' : 'Start first phase'}</button>
          <div class="scan-stats">
            <div><span>Saved frames</span><strong>{totalFrames.toLocaleString()}</strong></div>
            <div><span>Current phase</span><strong>{capturing ? sensor?.frameCount ?? 0 : '—'}</strong></div>
            <div><span>{capturing && sensor?.liveReconstructionMode === 'mesh' ? 'Live triangles' : 'Live points'}</span><strong>{formatCount(capturing && sensor?.liveReconstructionMode === 'mesh' ? sensor.liveTriangleCount : packedPreviewFrame?.pointCount ?? previewPoints.length)}</strong></div>
            <div><span>Tracking</span><strong class:warning={capturing && sensor?.liveReconstructionActive && !sensor?.tracking}>{capturing ? sensor?.tracking ? 'Locked' : 'Searching' : 'Standby'}</strong></div>
          </div>
          {#if capturing && sensor?.liveReconstructionActive}<div class="context-note"><strong>Live fusion selection</strong><p>{sensor.liveIntegratedFrameCount} keyframes fused · {sensor.liveRejectedFrameCount} tracking-gap frames excluded from offline reconstruction.</p></div>{/if}
          <div class="context-note accent"><strong>Capture guidance</strong><p>Move slowly, keep textured surfaces in view, and overlap each phase with the previous one.</p></div>

        {:else if workspaceMode === 'media'}
          <div class="summary-stack">
            <div><span>RGB-D recordings</span><strong>{completedPhases}</strong><small>{totalFrames.toLocaleString()} saved depth + color frames</small></div>
            <div><span>Imported sources</span><strong>{project.mediaSources.length}</strong><small>{project.mediaSources.reduce((sum, source) => sum + source.imageCount, 0).toLocaleString()} photos / extracted frames</small></div>
            <div><span>Selected imports</span><strong>{selectedMediaSourceIds.length}</strong><small>Included in the media pipeline</small></div>
          </div>
          <div class="context-note"><strong>Two source types</strong><p>RGB-D recordings use calibrated depth reconstruction. Imported photos and video use camera registration and Gaussian training.</p></div>
          <button class="button secondary full" on:click={() => void selectWorkspaceMode('process')} disabled={completedPhases === 0 && !hasMediaSources}>Configure processing →</button>

        {:else if workspaceMode === 'process'}
          <div class="source-choice">
            <button class:active={sourceMode === 'rgbd'} on:click={() => setSourceMode('rgbd')} disabled={processing || completedPhases === 0}>
              <span class="source-mark">D</span><div><strong>RGB-D captures</strong><small>{completedPhases} completed phases · {totalFrames.toLocaleString()} frames</small></div><i>{sourceMode === 'rgbd' ? 'Selected' : completedPhases ? 'Use' : 'Empty'}</i>
            </button>
            <button class:active={sourceMode === 'media'} on:click={() => setSourceMode('media')} disabled={processing || !hasMediaSources}>
              <span class="source-mark media">M</span><div><strong>Photos / video</strong><small>{selectedMediaSourceIds.length} of {project.mediaSources.length} sources selected</small></div><i>{sourceMode === 'media' ? 'Selected' : hasMediaSources ? 'Use' : 'Empty'}</i>
            </button>
          </div>
          {#if sourceMode === 'media' && hasMediaSources}
            <div class="mini-source-list">
              <span>INCLUDED SOURCES</span>
              {#each project.mediaSources as source}
                <label><input type="checkbox" checked={selectedMediaSourceIds.includes(source.id)} on:change={() => toggleMediaSource(source.id)} disabled={processing} /><div><strong>{source.name}</strong><small>{source.kind === 'video' ? 'Video' : `${source.imageCount} photos`}</small></div></label>
              {/each}
            </div>
          {/if}
          <div class="context-note"><strong>Source determines the pipeline</strong><p>{sourceMode === 'media' ? 'Media runs frame filtering, GPU camera registration, then Gaussian training.' : 'RGB-D uses calibrated depth, camera tracking, fusion, and optional texturing.'}</p></div>

        {:else if workspaceMode === 'render'}
          <div class="artifact-list">
            <div class:live={processing && previewPoints.length > 0} class:ready={previewPoints.length > 0 || Boolean(project.artifacts.pointCloud && !project.artifacts.pointCloud.stale)}><span class="artifact-icon">P</span><div><strong>Point cloud</strong><small>{processing && previewPoints.length > 0 ? `${formatCount(previewPoints.length)} live reconstruction points` : project.artifacts.pointCloud ? project.artifacts.pointCloud.stale ? 'Rebuild required' : `${formatCount(project.pointCount)} points` : 'Not built'}</small></div></div>
            <div class:ready={Boolean(project.artifacts.texturedMesh && !project.artifacts.texturedMesh.stale)}><span class="artifact-icon">M</span><div><strong>Textured mesh</strong><small>{project.artifacts.texturedMesh ? project.artifacts.texturedMesh.stale ? 'Rebuild required' : `${formatCount(project.meshTriangleCount)} triangles` : 'Not built'}</small></div></div>
            <div class:live={processing && Boolean(previewSplat)} class:ready={Boolean(previewSplat) || Boolean(project.artifacts.gaussianSplat && !project.artifacts.gaussianSplat.stale)}><span class="artifact-icon">G</span><div><strong>Gaussian splat</strong><small>{processing ? previewSplat ? `${formatCount(liveSplatCount)} live Gaussians · ${formatSnapshotTime(liveSplatUpdatedAt)}` : gaussianTrainingStage ? 'Waiting for the first training snapshot' : activeBuildIncludesSplat ? 'Available when Gaussian training starts' : 'Not part of this build' : project.artifacts.gaussianSplat ? project.artifacts.gaussianSplat.stale ? 'Rebuild required' : project.artifacts.gaussianSplat.metric ? 'Metric scale' : 'Arbitrary scale' : 'Not built'}</small></div></div>
          </div>
          {#if processing}
            <div class="context-note accent"><strong>Live build view</strong><p>Orbit, zoom, and switch between available geometry while the job continues. Model-orientation tools unlock after publishing finishes.</p></div>
          {:else}
            <div class="context-note accent"><strong>Viewer vs. saved model</strong><p>Edit controls are non-destructive. Use “Apply pose to exports” only when the saved coordinate system should change.</p></div>
          {/if}
          <button class="button secondary full" on:click={() => void selectWorkspaceMode('export')} disabled={processing || artifactCount === 0}>Continue to export →</button>

        {:else}
          <div class="delivery-summary"><strong>{exportCount}</strong><span>formats ready</span><p>Each export is created as a new copy. Project artifacts remain in place.</p></div>
          <div class="artifact-list compact-list">
            <div class:ready={Boolean(project.artifacts.gaussianSplat && !project.artifacts.gaussianSplat.stale)}><span class="artifact-icon">G</span><div><strong>Gaussian PLY</strong><small>{project.artifacts.gaussianSplat && !project.artifacts.gaussianSplat.stale ? 'Ready' : 'Unavailable'}</small></div></div>
            <div class:ready={Boolean(project.meshOutputPath)}><span class="artifact-icon">O</span><div><strong>Textured OBJ</strong><small>{project.meshOutputPath ? 'Ready' : 'Unavailable'}</small></div></div>
            <div class:ready={project.processingStatus === 'complete' && Boolean(project.outputPath)}><span class="artifact-icon">P</span><div><strong>Unity PLY</strong><small>{project.processingStatus === 'complete' && project.outputPath ? 'Ready' : 'Unavailable'}</small></div></div>
          </div>
          <div class="context-note"><strong>Coordinate note</strong><p>The Unity PLY corrects the X axis in the exported copy. Other source artifacts stay unchanged.</p></div>
        {/if}

        <div class="project-footer"><span>PROJECT</span><strong>{project.name}</strong><small title={project.path}>{project.path}</small></div>
      </aside>

      <section class="main-stage">
        <div class="stage-header">
          <div><div class="eyebrow">{workspaceKicker}</div><h2>{workspaceTitle}</h2></div>
          <div class="metrics">
            {#if workspaceMode === 'device'}
              <div><span>Stream</span><strong>{sensor?.sensorConnected ? `${sensor.streamFps.toFixed(1)} fps` : 'Offline'}</strong></div>
              <div><span>Preview</span><strong>{previewFps > 0 ? `${previewFps.toFixed(1)} fps` : '—'}</strong></div>
              <div><span>Visible</span><strong>{formatCount(packedPreviewFrame?.pointCount ?? previewPoints.length)}</strong></div>
            {:else if workspaceMode === 'capture'}
              <div><span>Phase frames</span><strong>{capturing ? (sensor?.frameCount ?? 0).toLocaleString() : '—'}</strong></div>
              <div><span>Total saved</span><strong>{totalFrames.toLocaleString()}</strong></div>
              <div><span>Tracking</span><strong>{capturing ? sensor?.tracking ? 'Locked' : 'Searching' : 'Standby'}</strong></div>
            {:else if workspaceMode === 'media'}
              <div><span>Recordings</span><strong>{completedPhases}</strong></div>
              <div><span>Imports</span><strong>{project.mediaSources.length}</strong></div>
              <div><span>Images</span><strong>{formatCount(project.mediaSources.reduce((sum, source) => sum + source.imageCount, 0))}</strong></div>
            {:else if workspaceMode === 'process'}
              {#if processing}
                <div><span>Stage</span><strong>{buildStage.label}</strong></div>
                <div><span>Live preview</span><strong>{previewSplat ? `${formatCount(previewSplat.byteLength / 32)} splats` : `${formatCount(previewPoints.length)} points`}</strong></div>
                <div><span>Overall</span><strong>{Math.round(overallBuildProgress * 100)}%</strong></div>
              {:else}
                <div><span>Input</span><strong>{sourceMode === 'media' ? 'Media' : 'RGB-D'}</strong></div>
                <div><span>Outputs</span><strong>{sourceMode === 'media' ? 1 : Number(buildPointCloud) + Number(buildTexturedMesh) + Number(buildGaussianSplat)}</strong></div>
                <div><span>Compute</span><strong>{sourceMode === 'media' || buildGaussianSplat ? 'CUDA' : 'GPU preferred'}</strong></div>
              {/if}
            {:else if workspaceMode === 'render' && processing}
              <div><span>Displayed</span><strong>{previewRenderMode === 'splat' && previewSplat ? 'Live Gaussian' : previewRenderMode === 'mesh' && previewMesh ? 'Mesh' : 'Live points'}</strong></div>
              <div><span>Gaussian</span><strong>{liveSplatState === 'ready' ? `${formatCount(liveSplatCount)} splats` : liveSplatState === 'loading' ? 'Checking…' : liveSplatState === 'error' ? 'Update error' : gaussianTrainingStage ? 'Waiting…' : activeBuildIncludesSplat ? 'Not started' : 'Not requested'}</strong></div>
              <div><span>Snapshot</span><strong>{formatSnapshotTime(liveSplatUpdatedAt)}</strong></div>
            {:else}
              <div><span>Points</span><strong>{formatCount(project.pointCount)}</strong></div>
              <div><span>Triangles</span><strong>{formatCount(project.meshTriangleCount)}</strong></div>
              <div title={project.confidenceDetail ?? 'Available after a successful build'}><span>Confidence</span><strong class={`confidence ${confidenceClass(project.confidenceScore)}`}>{project.confidenceScore !== undefined ? `${project.confidenceScore}%` : '—'}</strong></div>
            {/if}
          </div>
        </div>

        {#if workspaceMode === 'media'}
          <div class="media-workspace panel-inset">
            <div class="media-hero">
              <div><span>PROJECT MEDIA</span><h3>Recorded captures and imported media</h3><p>Review every source in one place before choosing a reconstruction pipeline.</p></div>
              <div class="media-import-actions"><button class="button secondary" on:click={importPhotosAction} disabled={busy || processing}>+ Photo folder</button><button class="button secondary" on:click={importVideoAction} disabled={busy || processing}>+ Video file</button></div>
            </div>
            {#if project.phases.length === 0 && project.mediaSources.length === 0}
              <div class="empty-state media-empty"><span>M</span><strong>No imported media</strong><p>Add a folder of overlapping photos or a steady video orbit to train a Gaussian splat.</p></div>
            {:else}
              {#if project.phases.length > 0}
                <div class="media-section-heading"><div><span>RGB-D RECORDINGS</span><small>All completed phases are used together</small></div><strong>{project.phases.length}</strong></div>
                <div class="media-grid">
                  {#each project.phases as phase, index}
                    <article class="media-source-card recorded-source">
                      <div class="source-select"><span>{phase.status === 'complete' ? 'Ready for RGB-D reconstruction' : phase.status}</span></div>
                      <div class="media-type-icon">D</div>
                      <div class="media-source-copy"><small>RGB-D PHASE {String(index + 1).padStart(2, '0')}</small><strong title={phase.name}>{phase.name}</strong><p>{phase.frameCount.toLocaleString()} frames · {formatDuration(phase.durationSeconds)}</p></div>
                      <div class="media-quality"><span>{phase.overlapHint}</span><small>Depth, color, calibration, and motion samples</small></div>
                      <button class="remove-source" on:click={() => removeCaptureAction(phase.id, phase.name)} disabled={busy || processing || phase.status === 'capturing'}>Remove</button>
                    </article>
                  {/each}
                </div>
              {/if}
              {#if project.mediaSources.length > 0}
                <div class="media-section-heading"><div><span>IMPORTED PHOTOS &amp; VIDEO</span><small>Select sources for Gaussian training</small></div><strong>{project.mediaSources.length}</strong></div>
                <div class="media-grid">
                  {#each project.mediaSources as source}
                    <article class:selected={selectedMediaSourceIds.includes(source.id)} class="media-source-card">
                      <label class="source-select"><input type="checkbox" checked={selectedMediaSourceIds.includes(source.id)} on:change={() => toggleMediaSource(source.id)} disabled={processing} /><span>{selectedMediaSourceIds.includes(source.id) ? 'Included in next build' : 'Excluded from build'}</span></label>
                      <div class="media-type-icon">{source.kind === 'video' ? '▶' : '▦'}</div>
                      <div class="media-source-copy"><small>{source.kind === 'video' ? 'VIDEO SOURCE' : 'PHOTO SET'}</small><strong title={source.name}>{source.name}</strong><p>{source.kind === 'video' ? 'Frames extracted during processing' : `${source.imageCount.toLocaleString()} images`} · {source.status}</p></div>
                      <div class="media-quality"><span>{source.quality?.registeredImages ?? 0}/{source.quality?.totalImages ?? source.imageCount} registered</span><small>{source.quality?.detail ?? 'Ready for registration'}</small></div>
                      <button class="remove-source" on:click={() => removeMediaSourceAction(source.id, source.name)} disabled={busy || processing}>Remove</button>
                    </article>
                  {/each}
                </div>
              {/if}
            {/if}
          </div>
        {:else if workspaceMode === 'process' && !processing}
          <div class="process-workspace panel-inset">
            <div class="process-plan-heading"><span>BUILD PLAN</span><h3>{sourceMode === 'media' ? 'Media to Gaussian splat' : 'RGB-D reconstruction'}</h3><p>{sourceMode === 'media' ? `${selectedMediaSourceIds.length} selected media sources will be registered and trained.` : `${completedPhases} capture phases and ${totalFrames.toLocaleString()} saved frames will be reconstructed.`}</p></div>
            <div class="pipeline-flow">
              {#each sourceMode === 'media' ? mediaStages : stagesFor(null) as stage, index}
                <div><span>{String(index + 1).padStart(2, '0')}</span><strong>{stage.label}</strong></div>
              {/each}
            </div>
            <div class="build-output-summary">
              <span>PLANNED OUTPUTS</span>
              <div>
                {#if sourceMode === 'media'}<strong>Gaussian splat</strong>{/if}
                {#if sourceMode === 'rgbd' && buildPointCloud}<strong>Point cloud</strong>{/if}
                {#if sourceMode === 'rgbd' && buildTexturedMesh}<strong>Textured mesh</strong>{/if}
                {#if sourceMode === 'rgbd' && buildGaussianSplat}<strong>Gaussian splat</strong>{/if}
                {#if sourceMode === 'rgbd' && !buildPointCloud && !buildTexturedMesh && !buildGaussianSplat}<small>No outputs selected. Choose one in Processing controls.</small>{/if}
              </div>
            </div>
          </div>
        {:else}
          <div class="viewer-wrap">
            <PointCloudPreview
              points={previewPoints}
              packedFrame={viewMode === 'live' && sensorSessionEnabled && sensor?.sensorConnected ? packedPreviewFrame : null}
              {processing}
              live={capturing || ((workspaceMode === 'device' || workspaceMode === 'capture') && Boolean(sensor?.sensorConnected))}
              {pointSize}
              opacity={pointOpacity}
              {showColors}
              {meshViewMode}
              {lightDirection}
              lightEditMode={lightEditMode && workspaceMode === 'render' && previewRenderMode === 'mesh' && meshViewMode === 'shaded'}
              renderMode={processing ? workspaceMode === 'render' ? previewRenderMode === 'splat' && previewSplat ? 'splat' : previewRenderMode === 'mesh' && previewMesh ? 'mesh' : 'points' : previewSplat ? 'splat' : 'points' : viewMode === 'preview' ? previewRenderMode : capturing && sensor?.liveReconstructionMode === 'mesh' && previewMesh ? 'mesh' : 'points'}
              mesh={previewMesh}
              splatBytes={previewSplat}
              assetLoading={previewAssetLoading === previewRenderMode ? previewAssetLoading : null}
              {cameraFrames}
              showCameraFrames={showCameraFrames && workspaceMode === 'render' && !processing}
              floorPickMode={floorPickMode && workspaceMode === 'render' && !processing}
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
              onLightDirectionChanged={handleLightDirectionChanged}
            />
            {#if capturing && sensor?.liveReconstructionActive && sensor.liveProcessedFrameCount > 0 && !sensor.tracking}
              <div class="tracking-lost-overlay"><span>TRACKING LOST</span><strong>Return to the last reconstructed area</strong><p>The map is frozen. Frames captured during this gap are being rejected until continuity is recovered.</p></div>
            {/if}
            {#if workspaceMode === 'render' && processing && previewPoints.length === 0 && !previewMesh && !previewSplat}
              <div class="viewer-empty live-wait"><span>WAITING FOR LIVE GEOMETRY</span><strong>The build is still preparing its first 3D preview</strong><button class="button secondary" on:click={() => void selectWorkspaceMode('process')}>View job progress</button></div>
            {:else if (workspaceMode === 'render' || workspaceMode === 'export') && artifactCount === 0 && previewPoints.length === 0 && !previewMesh && !previewSplat}
              <div class="viewer-empty"><span>NO MODEL YET</span><strong>Build an artifact to unlock this workspace</strong><button class="button secondary" on:click={() => void selectWorkspaceMode('process')}>Open Processing</button></div>
            {/if}
          </div>
        {/if}

        <div class:with-progress={processing} class="status-strip">
          {#if processing}
            <div class="job-feedback">
              <div class="job-heading"><span class="status-light busy"></span><div><span class="job-kicker">{activeJob?.pipeline === 'media_gaussian' ? 'MEDIA SPLAT JOB' : 'RGB-D ARTIFACT JOB'}</span><strong>{buildStage.label}</strong></div><span class="job-stage-count">Stage {buildStage.current} / {buildStage.total}</span></div>
              <div class="job-detail" title={buildDetail}>{buildDetail}</div>
              <div class="job-progress-grid"><span>Overall</span><div class="progress-track overall"><i style={`width: ${Math.max(2, overallBuildProgress * 100)}%`}></i></div><strong>{Math.round(overallBuildProgress * 100)}%</strong><span>Stage</span><div class="progress-track stage"><i style={`width: ${Math.max(2, currentStageProgress * 100)}%`}></i></div><strong>{Math.round(currentStageProgress * 100)}%</strong></div>
              <div class="job-backend">{buildBackend}</div>
            </div>
            <div class="job-timing"><span>Estimated left</span><strong>{formatEtaValue(totalBuildEta)}</strong><small>{currentStageEta != null ? `${formatEtaValue(currentStageEta)} this stage` : reconstruction?.elapsedSeconds != null ? `${reconstruction.elapsedSeconds}s elapsed` : 'Measuring throughput'}</small></div>
            <div class="job-actions">
              <button class="button secondary" on:click={() => void selectWorkspaceMode(workspaceMode === 'render' ? 'process' : 'render')}>{workspaceMode === 'render' ? 'Job details' : 'Open live Edit'}</button>
              <button class="button process cancel-job" on:click={processCloud} disabled={busy || !activeJob}>{activeJob?.status === 'cancelling' ? 'Cancelling…' : 'Cancel job'}</button>
            </div>
          {:else}
            <div class="status-copy"><div class="status-message"><span class:busy={busy} class="status-light"></span><div><small>STATUS</small><strong>{statusMessage}</strong></div></div></div>
            {#if jobAwaitingDecision}
              <div class="recovery-actions">
                <button class="button secondary" on:click={discardInterruptedJob} disabled={busy}>Cancel build</button>
                <button class="button process strong" on:click={resumeInterruptedJob} disabled={busy}>Resume build</button>
              </div>
            {:else if workspaceMode === 'device'}
              <button class="button secondary" disabled={busy || discoveryInFlight} on:click={() => void scanSensors()}>{discoveryInFlight ? 'Scanning…' : sensor?.sensorConnected ? 'Rescan devices' : 'Scan for devices'}</button>
            {:else if workspaceMode === 'capture'}
              <button class:stopping={capturing} class="button primary" on:click={captureAction} disabled={busy || (!capturing && !sensor?.sensorConnected)}><span class="record-dot"></span>{capturing ? 'Stop capture' : 'Start capture'}</button>
            {:else if workspaceMode === 'media'}
              <button class="button secondary" on:click={() => void selectWorkspaceMode('process')} disabled={completedPhases === 0 && !hasMediaSources}>Configure processing →</button>
            {:else if workspaceMode === 'process'}
              <button class="button process strong" on:click={processCloud} disabled={busy || (!activeJob?.resumable && !canBuildArtifacts)}>{activeJob?.resumable && ['failed', 'cancelled'].includes(activeJob.status) && activeJob.pipeline === (sourceMode === 'media' ? 'media_gaussian' : 'rgbd_reconstruction') ? 'Resume splat' : sourceMode === 'media' ? 'Train Gaussian splat' : 'Build artifacts'}</button>
            {:else if workspaceMode === 'render'}
              <button class="button secondary" on:click={() => void selectWorkspaceMode('export')} disabled={artifactCount === 0}>Open Export →</button>
            {/if}
          {/if}
        </div>
      </section>

      <aside class="settings panel">
        <div class="panel-heading control-heading"><div><div class="eyebrow">CONTROLS</div><h2>{workspaceMode === 'device' ? 'Input device' : workspaceMode === 'capture' ? 'Capture settings' : workspaceMode === 'media' ? 'Import settings' : workspaceMode === 'process' ? 'Processing' : workspaceMode === 'render' ? processing ? 'Live preview & edit' : 'Edit model' : 'Export formats'}</h2></div><span class="mode-chip">{workflowModes.find((mode) => mode.id === workspaceMode)?.step}</span></div>
        <p class="panel-intro">{workspaceMode === 'device' ? 'Only hardware and sensor-stream controls are shown.' : workspaceMode === 'capture' ? 'These settings apply to the next RGB-D phase.' : workspaceMode === 'media' ? 'Add or remove recorded media for this project.' : workspaceMode === 'process' ? 'Choose what the selected input should become.' : workspaceMode === 'render' ? processing ? 'Inspect live build geometry now. Saved-model edits unlock when publishing finishes.' : 'Choose a view and adjust the finished model without changing the source data.' : 'Save new copies in the format your next tool needs.'}</p>

        {#if workspaceMode === 'device'}
          <div class="section-divider first-section"><span>Connection</span></div>
          <div class="setting-group"><div class="label-row"><label for="sensor-device">Input sensor</label><button class="refresh-sensors" disabled={busy || processing || discoveryInFlight} on:click={() => void scanSensors()}>{discoveryInFlight ? 'Scanning…' : 'Scan again'}</button></div><select id="sensor-device" disabled={busy || processing || discoveryInFlight} bind:value={selectedSensorOption} on:change={sensorDeviceChanged}>{#if selectedSensorOption && selectedSensorOption !== networkFemtoOption && !sensorChoices.some((choice) => choice.id === selectedSensorOption)}<option value={selectedSensorOption}>{selectedSensorName} · unavailable</option>{/if}{#each sensorChoices as choice}<option value={choice.id}>{sensorOptionLabel(choice)}</option>{/each}<option value={networkFemtoOption}>Orbbec Femto Mega · Network IP…</option></select></div>
          {#if project.settings.sensorKind === 'femto_mega' && project.settings.sensorConnection === 'network'}<div class="setting-group"><label for="sensor-address">Camera IP address</label><input id="sensor-address" class="text-input" disabled={busy || processing} type="text" inputmode="decimal" placeholder="192.168.1.10 or IP:port" bind:value={project.settings.sensorAddress} on:change={sensorAddressChanged} /></div>{/if}
          {#if project.settings.sensorKind !== 'kinect_v2'}
            <div class="section-divider"><span>Depth stream</span></div>
            <div class="setting-group"><label for="depth-fov">Field of view</label><div class="segmented two-options" id="depth-fov"><button disabled={busy || processing} class:active={project.settings.depthFieldOfView === 'narrow'} on:click={() => setDepthFieldOfView('narrow')}>Narrow</button><button disabled={busy || processing} class:active={project.settings.depthFieldOfView === 'wide'} on:click={() => setDepthFieldOfView('wide')}>Wide</button></div></div>
            <div class="setting-group"><label for="depth-binning">Resolution mode</label><div class="segmented two-options" id="depth-binning"><button disabled={busy || processing} class:active={!project.settings.depthBinned} on:click={() => setDepthBinned(false)}>Full detail</button><button disabled={busy || processing} class:active={project.settings.depthBinned} on:click={() => setDepthBinned(true)}>2×2 binned</button></div><p class="setting-note">{project.settings.depthFieldOfView === 'wide' ? project.settings.depthBinned ? '512×512 at 30 fps' : '1024×1024 at 15 fps' : project.settings.depthBinned ? '320×288 at 30 fps' : '640×576 at 30 fps'}</p></div>
            <label class="toggle-row"><input disabled={busy || processing} type="checkbox" bind:checked={project.settings.useImu} on:change={scheduleProjectSettingsSave} /><span><strong>IMU tracking aid</strong><small>Seed offline camera motion</small></span></label>
          {/if}

        {:else if workspaceMode === 'capture'}
          <div class="section-divider first-section"><span>Scene</span></div>
          <div class="setting-group"><label for="live-reconstruction">Capture view</label><div class="segmented" id="live-reconstruction"><button disabled={busy || capturing || processing} class:active={project.settings.liveReconstruction === 'off'} on:click={() => setLiveReconstruction('off')}>Sensor frames</button><button disabled={busy || capturing || processing} class:active={project.settings.liveReconstruction === 'points'} on:click={() => setLiveReconstruction('points')}>Live points</button><button disabled={busy || capturing || processing} class:active={project.settings.liveReconstruction === 'mesh'} on:click={() => setLiveReconstruction('mesh')}>Live mesh</button></div><p class="setting-note">Live fusion uses a 10 mm or coarser working volume. The final Build still reruns full-quality global optimization.</p></div>
          <div class="setting-group"><label for="environment">Lighting environment</label><select id="environment" disabled={busy || capturing || processing} bind:value={project.settings.environment} on:change={scheduleProjectSettingsSave}><option value="indoor">Indoor / controlled light</option><option value="outdoor_low_light">Outdoor / night / sunset</option></select></div>
          <div class="setting-group range-group"><div class="label-row"><label for="depth">Maximum capture depth</label><output>{project.settings.maxDepthM.toFixed(1)} m</output></div><input id="depth" disabled={busy || capturing || processing} type="range" min="1.5" max="8" step="0.1" bind:value={project.settings.maxDepthM} on:input={scheduleProjectSettingsSave} /><div class="range-labels"><span>Near · 1.5 m</span><span>Far · 8.0 m</span></div></div>
          <div class="section-divider"><span>Saved data</span></div>
          <div class="setting-group"><label for="fps">Saved-frame rate</label><div class="segmented" id="fps">{#each [5, 10, 15] as fps}<button disabled={busy || capturing || processing} class:active={project.settings.captureFps === fps} on:click={() => setCaptureFps(fps)}>{fps} fps</button>{/each}</div><p class="setting-note">Sensor preview remains live; this controls only frames written to disk.</p></div>
          <div class="capture-ready-card"><span>{sensor?.sensorConnected ? 'READY TO RECORD' : 'DEVICE REQUIRED'}</span><strong>{sensor?.sensorConnected ? selectedSensorName : 'Return to Input device'}</strong><small>{sensor?.sensorConnected ? 'Settings lock while a phase is recording.' : 'Connect a sensor before starting a phase.'}</small><button class:stopping={capturing} class="button primary full" on:click={captureAction} disabled={busy || processing || (!capturing && !sensor?.sensorConnected)}><span class="record-dot"></span>{capturing ? 'Stop & save phase' : 'Start capture phase'}</button></div>

        {:else if workspaceMode === 'media'}
          <div class="section-divider first-section"><span>Add input</span></div>
          <button class="import-option" on:click={importPhotosAction} disabled={busy || processing}><span>▦</span><div><strong>Import photo folder</strong><small>Overlapping JPG or PNG images</small></div><i>Choose…</i></button>
          <button class="import-option" on:click={importVideoAction} disabled={busy || processing}><span>▶</span><div><strong>Import video file</strong><small>MP4, MOV, MKV, AVI, M4V, WebM</small></div><i>Choose…</i></button>
          <div class="section-divider"><span>Tool readiness</span></div>
          <div class="runtime-list"><div class:ready={runtime?.ffmpegAvailable}><span></span><div><strong>FFmpeg</strong><small>{runtime?.ffmpegAvailable ? 'Ready for video frames' : 'Missing media runtime'}</small></div></div><div class:ready={runtime?.colmapAvailable}><span></span><div><strong>COLMAP GPU</strong><small>{runtime?.colmapAvailable ? 'Ready for registration' : 'Missing registration runtime'}</small></div></div><div class:ready={runtime?.splatWorkerAvailable}><span></span><div><strong>Gaussian trainer</strong><small>{runtime?.splatWorkerAvailable ? 'CUDA runtime ready' : 'Optional runtime missing'}</small></div></div></div>

        {:else if workspaceMode === 'process'}
          {#if sourceMode === 'rgbd'}
            <div class="section-divider first-section"><span>Reconstruction resolution</span></div>
            <div class="setting-group range-group"><div class="label-row"><label for="voxel">Output point spacing</label><output>{project.settings.voxelSizeMm} mm</output></div><input id="voxel" disabled={busy || processing} type="range" min="1" max="40" step="1" bind:value={project.settings.voxelSizeMm} on:input={scheduleProjectSettingsSave} /><div class="range-labels"><span>Fine · more points</span><span>Coarse · lighter</span></div><p class="setting-note important-note">Reconstruction only — this does not change the frames you recorded.</p></div>
          {/if}
          <div class="section-divider" class:first-section={sourceMode === 'media'}><span>Output artifacts</span></div>
          {#if sourceMode === 'media'}
            <div class="locked-target"><span>G</span><div><strong>Gaussian splat</strong><small>Required output for photos / video</small></div><i>LOCKED</i></div>
          {:else}
            <div class="artifact-targets">
              <label class="target-option"><input type="checkbox" bind:checked={buildPointCloud} disabled={processing} /><span>P</span><div><strong>Point cloud</strong><small>Colored PLY · fast</small></div></label>
              <label class="target-option"><input type="checkbox" bind:checked={buildTexturedMesh} disabled={processing} /><span>M</span><div><strong>Textured mesh</strong><small>OBJ + MTL + PNG</small></div></label>
              <label class="target-option"><input type="checkbox" bind:checked={buildGaussianSplat} disabled={processing} /><span>G</span><div><strong>Gaussian splat</strong><small>Canonical PLY · CUDA</small></div></label>
            </div>
          {/if}
          {#if buildGaussianSplat || sourceMode === 'media'}
            <div class="section-divider"><span>Gaussian quality</span></div>
            <div class="setting-group range-group"><div class="label-row"><label for="splat-iterations">Training iterations</label><output>{splatIterations.toLocaleString()}</output></div><input id="splat-iterations" type="range" min="5000" max="60000" step="5000" bind:value={splatIterations} disabled={processing} /><div class="range-labels"><span>Faster</span><span>Higher quality</span></div></div>
            <p class:splat-ready={runtime?.splatWorkerAvailable} class="runtime-diagnostic">{runtime?.splatStatus ?? 'Checking optional CUDA runtime…'}{runtime?.splatWorkerAvailable ? ' · mixed precision ready' : ''}</p>
          {/if}
          {#if sourceMode === 'media'}<p class="setting-note readiness-line"><span class:ready={runtime?.ffmpegAvailable}>FFmpeg</span><span class:ready={runtime?.colmapAvailable}>COLMAP GPU</span><span class:ready={runtime?.splatWorkerAvailable}>CUDA trainer</span></p>{/if}
          {#if jobAwaitingDecision}
            <div class="build-recovery-actions">
              <button class="button secondary full" on:click={discardInterruptedJob} disabled={busy}>Cancel interrupted build</button>
              <button class="button process strong full" on:click={resumeInterruptedJob} disabled={busy}>Resume build</button>
            </div>
          {:else}
            <button class="button process strong full build-button" on:click={processCloud} disabled={busy || capturing || (!processing && !canBuildArtifacts)}>{processing ? 'Cancel build' : sourceMode === 'media' ? 'Train Gaussian splat' : 'Build selected artifacts'}</button>
          {/if}

        {:else if workspaceMode === 'render'}
          {#if processing && activeBuildIncludesSplat}
            <div class="section-divider first-section"><span>Live training output</span></div>
            <div class:ready={liveSplatState === 'ready'} class:error={liveSplatState === 'error'} class="live-splat-card">
              <div class="live-splat-heading"><span class:busy={liveSplatState === 'loading' || liveSplatState === 'waiting'}></span><div><small>GAUSSIAN SNAPSHOT</small><strong>{liveSplatState === 'ready' ? 'Live Gaussian available' : liveSplatState === 'loading' ? 'Checking for a snapshot' : liveSplatState === 'error' ? 'Preview update failed' : gaussianTrainingStage ? 'Training is running' : 'Waiting for Gaussian stage'}</strong></div></div>
              <p>{liveSplatState === 'ready' ? 'The viewer follows the latest published training snapshot and updates when a new one arrives.' : liveSplatState === 'error' ? splatPreviewError : gaussianTrainingStage ? 'The first snapshot appears after the trainer completes its initial optimization step.' : 'Registration and dataset preparation must finish before live Gaussians can be shown.'}</p>
              <div class="live-splat-meta"><span>{liveSplatState === 'ready' ? `${formatCount(liveSplatCount)} Gaussians` : buildStage.label}</span><span>{formatSnapshotTime(liveSplatUpdatedAt)}</span></div>
            </div>
          {/if}
          <div class="section-divider" class:first-section={!processing || !activeBuildIncludesSplat}><span>Display</span></div>
          <div class="setting-group"><label for="preview-rendering">Show available geometry as</label><div class="segmented renderer-options" id="preview-rendering"><button disabled={previewPoints.length === 0} class:active={previewRenderMode === 'points'} on:click={() => setPreviewRenderMode('points')}>Points</button><button disabled={!previewMesh && !project.meshOutputPath} class:active={previewRenderMode === 'mesh'} on:click={() => void selectMeshPreview()}>Mesh</button><button aria-disabled={!previewSplat && !project.artifacts.gaussianSplat} title={!previewSplat && !project.artifacts.gaussianSplat ? processing && activeBuildIncludesSplat ? 'Waiting for a live Gaussian snapshot' : 'Train a Gaussian splat to enable this view' : 'Show the Gaussian splat'} class:active={previewRenderMode === 'splat'} on:click={() => void selectSplatPreview()}>Gaussian</button></div>{#if !previewMesh && !previewSplat && !project.meshOutputPath && !project.artifacts.gaussianSplat}<p class="setting-note">{processing && activeBuildIncludesSplat ? 'Points remain available until the first Gaussian snapshot is published.' : 'Only point display is available for the current model.'}</p>{/if}</div>
          {#if previewRenderMode === 'mesh'}
            <div class="setting-group"><label for="mesh-view-mode">Mesh view</label><div class="segmented mesh-view-options" id="mesh-view-mode"><button class:active={meshViewMode === 'surface'} on:click={() => setMeshViewMode('surface')}>Mesh</button><button class:active={meshViewMode === 'surface-wireframe'} on:click={() => setMeshViewMode('surface-wireframe')}>Mesh + wire</button><button class:active={meshViewMode === 'wireframe'} on:click={() => setMeshViewMode('wireframe')}>Wireframe</button><button class:active={meshViewMode === 'shaded'} on:click={() => setMeshViewMode('shaded')}>Shaded</button></div></div>
            {#if meshViewMode === 'shaded'}
              <div class="light-editor"><div class="label-row"><span>Light direction</span><button on:click={resetLightDirection}>Reset</button></div><p class="setting-note">Vector from the model toward the light.</p><div class="light-direction-grid">{#each ['X', 'Y', 'Z'] as axis, index}<label><span>{axis}</span><input aria-label={`Light direction ${axis}`} type="number" step="0.05" value={lightDirection[index]} on:input={(event) => updateLightDirection(index, Number(event.currentTarget.value))} /></label>{/each}</div><button class:active={lightEditMode} class="tool-button light-gizmo-button" disabled={!previewMesh} on:click={toggleLightEdit}>{lightEditMode ? 'Finish light gizmo' : 'Edit direction with gizmo'}</button></div>
            {/if}
          {/if}
          {#if previewRenderMode === 'points'}<div class="setting-group range-group"><div class="label-row"><label for="point-size">Point size</label><output>{pointSize.toFixed(3)}</output></div><input id="point-size" type="range" min="0.005" max="0.08" step="0.003" bind:value={pointSize} on:input={scheduleVisualizationSave} /></div>{/if}
          <div class="setting-group range-group"><div class="label-row"><label for="opacity">Opacity</label><output>{Math.round(pointOpacity * 100)}%</output></div><input id="opacity" type="range" min="0.1" max="1" step="0.05" bind:value={pointOpacity} on:input={scheduleVisualizationSave} /></div>
          <label class="toggle-row"><input type="checkbox" bind:checked={showColors} on:change={scheduleVisualizationSave} /><span><strong>Captured colors</strong><small>Show RGB data on the model</small></span></label>
          <label class="toggle-row"><input type="checkbox" disabled={processing || cameraFrames.length === 0} bind:checked={showCameraFrames} on:change={scheduleVisualizationSave} /><span><strong>Capture cameras</strong><small>{processing ? 'Available after the build finishes' : cameraFrames.length ? `${cameraFrames.length} registered views` : 'No camera poses available'}</small></span></label>
          <div class="section-divider"><span>Model orientation</span></div>
          {#if processing}<p class="setting-note edit-lock-note">View controls stay active during training. Orientation and saved-model edits unlock after artifacts are published.</p>{/if}
          <button class:active={editMode} class="tool-button" disabled={!canEdit} on:click={() => { editMode = !editMode; lightEditMode = false; floorPickMode = false; anchorPickMode = false; }}>{editMode ? 'Exit transform gizmo' : 'Transform with gizmo'}</button>
          {#if editMode && canEdit}<div class="gizmo-modes"><button class:active={gizmoMode === 'translate'} on:click={() => setGizmoMode('translate')}>Move · W</button><button class:active={gizmoMode === 'rotate'} on:click={() => setGizmoMode('rotate')}>Rotate · E</button><button class:active={gizmoMode === 'scale'} on:click={() => setGizmoMode('scale')}>Scale · R</button></div>{/if}
          {#if canEdit}<div class="anchor-editor"><div class="label-row"><span>Gizmo anchor</span><button on:click={centerGizmoAnchor}>Center</button></div><div class="anchor-grid">{#each ['X', 'Y', 'Z'] as axis, index}<label><span>{axis}</span><input type="number" step="0.05" value={effectiveGizmoAnchor[index]} on:input={(event) => updateGizmoAnchor(index, Number(event.currentTarget.value))} /></label>{/each}</div><button class:active={anchorPickMode} class="tool-button anchor-pick" on:click={toggleAnchorPick}>{anchorPickMode ? 'Cancel anchor pick' : 'Pick anchor on model'}</button></div>{/if}
          <button class:active={floorPickMode} class="tool-button orientation-action" disabled={!canEdit} on:click={() => { floorPickMode = !floorPickMode; editMode = false; lightEditMode = false; anchorPickMode = false; }}>{floorPickMode ? 'Cancel floor pick' : 'Pick floor point'}</button>
          <button class="tool-button orientation-action" disabled={!canEdit} on:click={alignRoomAxes}>Align room axes</button>
          <div class="axis-actions"><button disabled={!canEdit} on:click={() => flipAxis('X')}>Flip X</button><button disabled={!canEdit} on:click={() => flipAxis('Y')}>Flip Y</button><button disabled={!canEdit} on:click={() => flipAxis('Z')}>Flip Z</button></div>
          <button class="tool-button subtle" on:click={resetTransform} disabled={!canEdit}>Reset viewer pose</button>

        {:else}
          <div class="export-warning"><span>EXPORTS ARE COPIES</span><p>Choose a format below. The project’s working artifacts will not be replaced.</p></div>
          <div class:stale={project.artifacts.gaussianSplat?.stale} class="export-card"><span>GAUSSIAN SPLAT</span><strong>Canonical 3DGS PLY</strong><small>{project.artifacts.gaussianSplat ? project.artifacts.gaussianSplat.metric ? 'Metric scale · coordinate sidecars included' : 'Arbitrary scale · coordinate sidecars included' : 'Build a Gaussian splat to enable'}</small><button class="tool-button export-ply" on:click={exportGaussianSplatAction} disabled={busy || !project.artifacts.gaussianSplat || project.artifacts.gaussianSplat.stale}>Export Gaussian PLY…</button></div>
          <div class="export-card"><span>TEXTURED SURFACE</span><strong>OBJ + MTL + PNG</strong><small>{project.meshOutputPath ? `${formatCount(project.meshTriangleCount)} textured triangles` : 'Build a textured mesh to enable'}</small><button class="tool-button export-ply" on:click={exportTexturedMeshAction} disabled={busy || project.processingStatus !== 'complete' || !project.meshOutputPath}>Export OBJ bundle…</button></div>
          <div class="export-card"><span>COLORED POINTS</span><strong>Unity-ready PLY</strong><small>{project.outputPath ? `${formatCount(project.pointCount)} points · X-axis corrected copy` : 'Build a point cloud to enable'}</small><button class="tool-button export-ply" on:click={exportPlyAction} disabled={busy || project.processingStatus !== 'complete' || !project.outputPath}>Export Unity PLY…</button></div>
          <div class="section-divider"><span>Coordinate system</span></div>
          <button class="tool-button export-transform" on:click={applyTransformToExport} disabled={busy || project.processingStatus !== 'complete'}>Apply current pose to model exports</button>
          <p class="setting-note">This bakes the viewer pose into the point cloud, mesh, and camera coordinates. Backups are retained.</p>
        {/if}
      </aside>
    </section>
  {:else if initializationError}
    <div class="loading">{initializationError}</div>
  {:else}
    <div class="loading">Preparing scanner workspace…</div>
  {/if}
</main>
