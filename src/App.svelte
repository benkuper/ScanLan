<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { save } from '@tauri-apps/plugin-dialog';
  import * as THREE from 'three';
  import PointCloudPreview from './lib/components/PointCloudPreview.svelte';
  import {
    applyCloudTransform,
    availableSensors,
    captureStatus,
    createProject,
    currentProject,
    exportPly,
    exportTexturedMesh,
    loadCameraFrames,
    loadLivePreviewFrame,
    loadPreview,
    loadPreviewMesh,
    removeCapture,
    reconstructProject,
    startSensorPhase,
    stopSensorPhase,
    updateProjectSettings
  } from './lib/api';
  import type {
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
    ReconstructionProgress
  } from './lib/types';

  let project: ProjectSummary | null = null;
  let sensor: CaptureStatus | null = null;
  let reconstruction: ReconstructionProgress | null = null;
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
  $: processing = project?.processingStatus === 'processing';
  $: selectedSensorName = project?.settings.sensorKind === 'azure_kinect'
    ? 'Azure Kinect DK'
    : project?.settings.sensorKind === 'femto_mega'
      ? 'Orbbec Femto Mega'
      : 'Kinect v2';
  $: canEdit = viewMode === 'preview' && project?.processingStatus === 'complete' && previewPoints.length > 0;
  $: viewerTransform = viewMode === 'preview' ? cloudTransform : identityTransform;
  $: effectiveGizmoAnchor = gizmoAnchor ?? pointCloudCenter(previewPoints);
  $: completedPhases = project?.phases.filter((phase) => phase.status === 'complete').length ?? 0;
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
      else if (processing && reconstruction) {
        message = `${reconstruction.stage}: ${reconstruction.detail}`;
      } else if (!status.capturing && status.sensorConnected && viewMode === 'live') {
        message = `Live ${status.sensorName} preview · ${status.previewPointCount.toLocaleString()} visible points · ${status.streamFps.toFixed(1)} sensor fps${previewFps > 0 ? ` · ${previewFps.toFixed(1)} preview fps` : ''}`;
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

  async function processCloud() {
    if (!project || completedPhases === 0) return;
    busy = true;
    reconstruction = null;
    viewMode = 'preview';
    editMode = false;
    previewPoints = [];
    cameraFrames = [];
    previewMesh = null;
    project = {
      ...project,
      processingStatus: 'processing',
      processingError: undefined,
      pointCount: undefined,
      outputPath: undefined,
      meshTriangleCount: undefined,
      meshOutputPath: undefined,
      cameraFrameCount: undefined,
      confidenceScore: undefined,
      confidenceLabel: undefined,
      confidenceDetail: undefined,
      framesUsed: undefined,
      processingBackend: undefined,
      processingDurationSeconds: undefined
    };
    message = 'Preparing captured frames…';
    try {
      project = await reconstructProject(project.path, project.settings);
      viewMode = 'preview';
      await refreshResultPreview();
      message = `3D model ready with ${project.meshTriangleCount?.toLocaleString() ?? 0} textured triangles and ${project.pointCount?.toLocaleString() ?? 0} points · ${project.confidenceScore ?? 0}% confidence${project.processingBackend ? ` · ${project.processingBackend}` : ''}${project.processingDurationSeconds !== undefined ? ` in ${formatEta(Math.round(project.processingDurationSeconds))}` : ''}.`;
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

  onMount(() => {
    // Remove the old global transform once. It used to rotate every project and
    // even the raw sensor feed, which could make a new live view appear inverted.
    localStorage.removeItem('scanlan-cloud-transform');
    loadVisualizationPreferences();
    void (async () => {
      try {
        project = await currentProject();
        loadTransform(project.id);
        selectedSensorOption = configuredSensorOption(project.settings);
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
      <div class="runtime-pill" class:connected={sensor?.sensorConnected} class:paused={sensor?.sensorPaused} title={sensor?.sensorStatus}>
        <span></span>
        {sensor?.sensorPaused ? 'Sensor paused for build' : discoveryInFlight ? 'Scanning for sensors…' : connecting ? `Opening ${selectedSensorName}…` : sensor?.sensorConnected ? `${sensor.sensorName} · ${sensor.streamFps.toFixed(1)} sensor fps${!capturing && previewFps > 0 ? ` · ${previewFps.toFixed(1)} preview fps` : ''}` : sensorSessionEnabled ? `${selectedSensorName} disconnected` : 'Sensor scan required'}
      </div>
      <button class="button ghost" on:click={newProject} disabled={busy || capturing}>New project</button>
      <button class:stopping={capturing} class="button primary" on:click={captureAction} disabled={busy || !project || (!capturing && !sensor?.sensorConnected)}>
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

        <div class="status-strip" class:with-progress={processing}>
          <div class="status-copy">
            <div class="status-message"><span class:busy={busy || processing} class="status-light"></span>{message}</div>
            {#if processing}
              <div class="progress-row">
                <div class="progress-track"><i style={`width: ${Math.max(2, (reconstruction?.progress ?? 0) * 100)}%`}></i></div>
                <strong>{Math.round((reconstruction?.progress ?? 0) * 100)}%</strong>
                <span>{reconstruction?.stageEtaSeconds !== undefined ? formatEta(reconstruction.stageEtaSeconds) : reconstruction?.elapsedSeconds !== undefined ? `${reconstruction.elapsedSeconds}s elapsed` : formatEta(reconstruction?.etaSeconds)}{reconstruction?.computeBackend ? ` · ${reconstruction.computeBackend}` : ''}</span>
              </div>
            {/if}
          </div>
          <button class="button process" on:click={processCloud} disabled={busy || capturing || completedPhases === 0}>{processing ? 'Processing…' : 'Build 3D model'}</button>
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
