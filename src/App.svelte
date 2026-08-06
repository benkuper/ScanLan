<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { open, save } from '@tauri-apps/plugin-dialog';
  import PointCloudPreview from './lib/components/PointCloudPreview.svelte';
  import {
    artifactJobStatus,
    availableSensors,
    cancelArtifactJob,
    captureStatus,
    createProject,
    currentProject,
    deleteProject,
    discardArtifactJob,
    exportGaussianSplat,
    exportPly,
    exportTexturedMesh,
    importMediaSources,
    latestArtifactJob,
    listProjects,
    loadCaptureDraft,
    loadGaussianSplat,
    loadLivePreviewFrame,
    loadLiveReconstructionMesh,
    localizeSupplementalPhotos,
    loadPreview,
    loadPreviewMesh,
    openProject,
    removeCapture,
    removeSupplementalPhoto,
    resumeArtifactJob,
    runtimeInfo,
    saveProject,
    startArtifactJob,
    startSensorPreview,
    startSensorPhase,
    stopSensorPreview,
    stopSensorPhase,
    supplementalPhotoProgress,
    supplementalPhotos,
    updateProjectSettings
  } from './lib/api';
  import type { SupplementalPhotoAttempt, SupplementalPhotoProgress } from './lib/api';
  import type {
    ArtifactJob,
    ArtifactTarget,
    AvailableSensor,
    CaptureSettings,
    CaptureStatus,
    CloudTransform,
    DepthFieldOfView,
    LiveReconstructionMode,
    MeshViewMode,
    PackedPreviewFrame,
    PreviewMesh,
    PreviewPoint,
    ProjectCatalogEntry,
    ProjectSummary,
    RuntimeInfo,
    SensorKind
  } from './lib/types';

  type Workspace = 'capture' | 'reconstruct' | 'inspect';
  type RenderMode = 'points' | 'mesh' | 'splat';
  type TransformSaveMode = 'auto' | 'manual';

  let project: ProjectSummary | null = null;
  let projectCatalog: ProjectCatalogEntry[] = [];
  let projectManagerOpen = false;
  let projectCatalogLoading = false;
  let projectManagerError = '';
  let projectNameDraft = '';
  let newProjectName = '';
  let newProjectFormOpen = false;
  let sensor: CaptureStatus | null = null;
  let runtime: RuntimeInfo | null = null;
  let sensors: AvailableSensor[] = [];
  let activeJob: ArtifactJob | null = null;
  let workspace: Workspace = 'capture';
  let renderMode: RenderMode = 'points';
  let meshViewMode: MeshViewMode = 'surface';
  let floorPickMode = false;
  let editMode = false;
  let gizmoMode: 'translate' | 'rotate' | 'scale' = 'translate';
  let rotationSnapDegrees = 0;
  let transformSaveMode: TransformSaveMode = 'auto';
  let transformDirty = false;
  let cloudTransform: CloudTransform = identityTransform();
  let savedCloudTransform: CloudTransform = identityTransform();
  let gizmoAnchor: [number, number, number] = [0, 0, 0];
  let loadedTransformProjectId = '';

  let previewPoints: PreviewPoint[] = [];
  let packedPreviewFrame: PackedPreviewFrame | null = null;
  let captureDraftFrame: PackedPreviewFrame | null = null;
  let previewMesh: PreviewMesh | null = null;
  let liveMesh: PreviewMesh | null = null;
  let previewSplat: Uint8Array | null = null;
  let assetLoading: RenderMode | null = null;

  let buildPointCloud = true;
  let buildTexturedMesh = true;
  let buildGaussianSplat = false;
  let splatIterations = 30_000;

  let busy = false;
  let discovering = false;
  let sensorScanInFlight = false;
  let statusInFlight = false;
  let geometryInFlight = false;
  let resultInFlight = false;
  let message = 'Initializing the RGB-D engine…';
  let fatalError = '';
  let statusTimer: number | undefined;
  let geometryTimer: number | undefined;
  let settingsTimer: number | undefined;
  let settingsRevision = 0;
  let projectGeneration = 0;
  let settingsDirty = false;
  let selectingSensor = false;
  let lastPreviewFrame = 0;
  let lastMeshFrame = 0;
  let lastBuildPreviewAt = 0;
  let lastBuildSplatSignature = '';
  let lastSensorScanAt = 0;
  let completedJobId = '';
  let texturePhotos: SupplementalPhotoAttempt[] = [];
  let texturePhotoProgress: SupplementalPhotoProgress | null = null;
  let photoLocalizationActive = false;
  let photoProgressTimer: number | undefined;
  let photoProgressInFlight = false;
  let photoProgressProcessed = -1;

  let capturing = false;
  let previewing = false;
  let liveSensor = false;
  let processing = false;
  let completedCaptures = 0;
  let totalFrames = 0;
  let mediaSourceCount = 0;
  let mediaOnlyProject = false;
  let readyArtifacts = 0;
  let viewerRenderMode: RenderMode = 'points';
  let viewerMesh: PreviewMesh | null = null;
  let viewerPackedFrame: PackedPreviewFrame | null = null;
  let currentSensorKey = '';
  let selectedSensor: AvailableSensor | null = null;
  let selectedSensorConnected = false;
  let canEditModel = false;
  let hasEditPose = false;
  let viewerCloudTransform: CloudTransform = identityTransform();
  let localizedTexturePhotoCount = 0;
  let rejectedTexturePhotoCount = 0;
  let pendingTexturePhotoCount = 0;

  $: capturing = Boolean(sensor?.capturing);
  $: previewing = Boolean(sensor?.previewing);
  $: liveSensor = capturing || previewing;
  $: processing = Boolean(activeJob && ['queued', 'running', 'cancelling'].includes(activeJob.status));
  $: completedCaptures = project?.phases.filter((capture) => capture.status === 'complete').length ?? 0;
  $: totalFrames = project?.phases.reduce((sum, capture) => sum + capture.frameCount, 0) ?? 0;
  $: mediaSourceCount = project?.mediaSources.length ?? 0;
  $: mediaOnlyProject = mediaSourceCount > 0 && completedCaptures === 0;
  $: if (mediaOnlyProject) {
    buildPointCloud = false;
    buildTexturedMesh = false;
  }
  $: readyArtifacts = project
    ? Object.values(project.artifacts).filter((artifact) => artifact && !artifact.stale && artifact.status === 'ready').length
    : 0;
  $: viewerRenderMode = previewing
    ? 'points'
    : capturing
      ? project?.settings.liveReconstruction === 'mesh' ? 'mesh' : 'points'
      : renderMode;
  $: viewerMesh = capturing ? liveMesh : previewing ? null : previewMesh;
  $: viewerPackedFrame = liveSensor
    ? packedPreviewFrame
    : workspace === 'reconstruct' && !processing && readyArtifacts === 0
      ? captureDraftFrame
      : null;
  $: currentSensorKey = project ? configuredSensorKey(project.settings) : '';
  $: selectedSensor = sensors.find((candidate) => sensorKey(candidate) === currentSensorKey) ?? null;
  $: selectedSensorConnected = liveSensor
    ? Boolean(sensor?.sensorConnected)
    : Boolean(selectedSensor?.connected);
  $: canEditModel = workspace === 'inspect'
    && !processing
    && renderMode !== 'splat'
    && (previewPoints.length > 0 || Boolean(previewMesh?.positions.length));
  $: hasEditPose = !isIdentityTransform(cloudTransform);
  $: gizmoAnchor = modelCenter(renderMode, previewPoints, previewMesh);
  $: viewerCloudTransform = workspace === 'inspect' && renderMode !== 'splat'
    ? cloudTransform
    : identityTransform();
  $: localizedTexturePhotoCount = texturePhotos.filter((photo) => photo.status === 'localized').length;
  $: rejectedTexturePhotoCount = texturePhotos.filter((photo) => photo.status === 'rejected').length;
  $: pendingTexturePhotoCount = texturePhotos.filter((photo) => ['queued', 'localizing'].includes(photo.status)).length;
  $: if (project && loadedTransformProjectId !== project.id) loadTransform(project.id);

  function identityTransform(): CloudTransform {
    return { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] };
  }

  function isIdentityTransform(transform: CloudTransform): boolean {
    const values = [...transform.position, ...transform.rotation];
    return values.every((value) => Math.abs(value) < 1e-5)
      && transform.scale.every((value) => Math.abs(value - 1) < 1e-5);
  }

  function errorText(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }

  function formatCount(value: number | undefined): string {
    if (!value) return '0';
    return new Intl.NumberFormat('en', { notation: value >= 100_000 ? 'compact' : 'standard', maximumFractionDigits: 1 }).format(value);
  }

  function formatDuration(seconds: number | undefined): string {
    if (!seconds) return '0s';
    const minutes = Math.floor(seconds / 60);
    const remainder = Math.round(seconds % 60);
    return minutes ? `${minutes}m ${remainder}s` : `${remainder}s`;
  }

  function formatProjectDate(value: string): string {
    const date = new Date(value);
    return Number.isNaN(date.getTime())
      ? 'Unknown date'
      : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date);
  }

  function readyArtifactCount(candidate: ProjectSummary): number {
    return Object.values(candidate.artifacts)
      .filter((artifact) => artifact && artifact.status === 'ready' && !artifact.stale)
      .length;
  }

  function splatSignature(bytes: Uint8Array): string {
    if (!bytes.byteLength) return '';
    let hash = 2166136261;
    const sampleCount = Math.min(32, bytes.byteLength);
    for (let sample = 0; sample < sampleCount; sample += 1) {
      const index = Math.floor(sample * (bytes.byteLength - 1) / Math.max(sampleCount - 1, 1));
      hash ^= bytes[index];
      hash = Math.imul(hash, 16777619);
    }
    return `${bytes.byteLength}:${(hash >>> 0).toString(16)}`;
  }

  function artifactReady(target: ArtifactTarget): boolean {
    const artifact = project?.artifacts[target];
    return Boolean(artifact && artifact.status === 'ready' && !artifact.stale);
  }

  function inputValue(event: Event): string {
    return (event.currentTarget as HTMLInputElement | HTMLSelectElement).value;
  }

  function inputChecked(event: Event): boolean {
    return (event.currentTarget as HTMLInputElement).checked;
  }

  const transformStorageKey = (projectId: string) => `scanlan-cloud-transform:${projectId}`;
  const transformSaveModeStorageKey = 'scanlan-transform-save-mode';
  const rotationSnapStorageKey = 'scanlan-rotation-snap-degrees';

  function cloneTransform(transform: CloudTransform): CloudTransform {
    return {
      position: [...transform.position],
      rotation: [...transform.rotation],
      scale: [...transform.scale]
    };
  }

  function transformsEqual(left: CloudTransform, right: CloudTransform): boolean {
    const leftValues = [...left.position, ...left.rotation, ...left.scale];
    const rightValues = [...right.position, ...right.rotation, ...right.scale];
    return leftValues.every((value, index) => Math.abs(value - rightValues[index]) < 1e-5);
  }

  function loadEditorPreferences(): void {
    const storedSaveMode = localStorage.getItem(transformSaveModeStorageKey);
    if (storedSaveMode === 'auto' || storedSaveMode === 'manual') transformSaveMode = storedSaveMode;
    const storedSnap = Number(localStorage.getItem(rotationSnapStorageKey));
    if ([0, 1, 5, 15].includes(storedSnap)) rotationSnapDegrees = storedSnap;
  }

  function validVector(value: unknown, fallback: [number, number, number]): [number, number, number] {
    if (!Array.isArray(value) || value.length !== 3 || value.some((item) => !Number.isFinite(item))) {
      return [...fallback];
    }
    return [Number(value[0]), Number(value[1]), Number(value[2])];
  }

  function loadTransform(projectId: string): void {
    loadedTransformProjectId = projectId;
    cloudTransform = identityTransform();
    savedCloudTransform = identityTransform();
    transformDirty = false;
    editMode = false;
    floorPickMode = false;
    const stored = localStorage.getItem(transformStorageKey(projectId));
    if (!stored) return;
    try {
      const parsed = JSON.parse(stored) as Partial<CloudTransform>;
      const scale = validVector(parsed.scale, [1, 1, 1]);
      cloudTransform = {
        position: validVector(parsed.position, [0, 0, 0]),
        rotation: validVector(parsed.rotation, [0, 0, 0]),
        scale: scale.map((value) => Math.abs(value) < 1e-4 ? (value < 0 ? -1e-4 : 1e-4) : value) as [number, number, number]
      };
      savedCloudTransform = cloneTransform(cloudTransform);
    } catch {
      localStorage.removeItem(transformStorageKey(projectId));
    }
  }

  function persistTransform(): void {
    if (!project) return;
    if (isIdentityTransform(cloudTransform)) localStorage.removeItem(transformStorageKey(project.id));
    else localStorage.setItem(transformStorageKey(project.id), JSON.stringify(cloudTransform));
    savedCloudTransform = cloneTransform(cloudTransform);
    transformDirty = false;
  }

  function setTransformSaveMode(mode: TransformSaveMode): void {
    transformSaveMode = mode;
    localStorage.setItem(transformSaveModeStorageKey, mode);
    if (mode === 'auto' && transformDirty) {
      persistTransform();
      message = 'Automatic pose saving enabled; the current pose was saved.';
    }
  }

  function setRotationSnap(degrees: number): void {
    rotationSnapDegrees = degrees;
    localStorage.setItem(rotationSnapStorageKey, String(degrees));
  }

  function finishTransform(label: string): void {
    const changed = !transformsEqual(cloudTransform, savedCloudTransform);
    transformDirty = changed;
    if (transformSaveMode === 'auto') {
      if (changed) persistTransform();
      message = changed
        ? `${label} updated and saved; exports will use this pose.`
        : `${label} is unchanged.`;
    } else {
      message = changed
        ? `${label} updated in the draft. Save the pose when you are ready.`
        : `${label} matches the saved pose.`;
    }
  }

  function saveTransform(): void {
    persistTransform();
    message = 'Model pose saved; exports will use this pose.';
  }

  function modelCenter(mode: RenderMode, points: PreviewPoint[], mesh: PreviewMesh | null): [number, number, number] {
    let count = 0;
    let stride = 1;
    let pointAt: (index: number) => [number, number, number];
    if (mode === 'mesh' && mesh?.positions.length) {
      count = Math.floor(mesh.positions.length / 3);
      stride = Math.max(1, Math.floor(count / 100_000));
      pointAt = (index) => [mesh.positions[index * 3], mesh.positions[index * 3 + 1], mesh.positions[index * 3 + 2]];
    } else if (points.length) {
      count = points.length;
      stride = Math.max(1, Math.floor(count / 100_000));
      pointAt = (index) => points[index].position;
    } else {
      return [0, 0, 0];
    }
    const minimum = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
    const maximum = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
    for (let index = 0; index < count; index += stride) {
      const point = pointAt(index);
      for (let axis = 0; axis < 3; axis += 1) {
        minimum[axis] = Math.min(minimum[axis], point[axis]);
        maximum[axis] = Math.max(maximum[axis], point[axis]);
      }
    }
    return [0, 1, 2].map((axis) => (minimum[axis] + maximum[axis]) * 0.5) as [number, number, number];
  }

  function setGizmoMode(mode: 'translate' | 'rotate' | 'scale'): void {
    gizmoMode = mode;
    editMode = true;
    floorPickMode = false;
  }

  function handleGizmoTransform(transform: CloudTransform): void {
    cloudTransform = {
      ...transform,
      scale: transform.scale.map((value) => Math.abs(value) < 1e-4 ? (value < 0 ? -1e-4 : 1e-4) : value) as [number, number, number]
    };
  }

  function commitGizmoTransform(): void {
    finishTransform(gizmoMode === 'translate' ? 'Position' : gizmoMode === 'rotate' ? 'Rotation' : 'Scale');
  }

  function setFloorTransform(transform: CloudTransform): void {
    cloudTransform = transform;
    floorPickMode = false;
    finishTransform('Floor alignment');
  }

  function resetEditPose(): void {
    cloudTransform = identityTransform();
    editMode = false;
    floorPickMode = false;
    finishTransform('Model pose');
  }

  function isTextEntryTarget(target: EventTarget | null): boolean {
    return target instanceof HTMLElement
      && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName));
  }

  function handleEditShortcut(event: KeyboardEvent): void {
    if (event.key === 'Escape' && projectManagerOpen) {
      closeProjectManager();
      return;
    }
    if (event.key === 'Escape' && (editMode || floorPickMode)) {
      editMode = false;
      floorPickMode = false;
      return;
    }
    if (!canEditModel || event.repeat || isTextEntryTarget(event.target)) return;
    const mode = event.key.toLowerCase() === 'w'
      ? 'translate'
      : event.key.toLowerCase() === 'e'
        ? 'rotate'
        : event.key.toLowerCase() === 'r'
          ? 'scale'
          : null;
    if (!mode) return;
    event.preventDefault();
    setGizmoMode(mode);
  }

  function updateSetting<K extends keyof CaptureSettings>(key: K, value: CaptureSettings[K]): void {
    if (!project || capturing || processing) return;
    project = { ...project, settings: { ...project.settings, [key]: value } };
    settingsRevision += 1;
    projectGeneration += 1;
    settingsDirty = true;
    const revision = settingsRevision;
    if (settingsTimer) window.clearTimeout(settingsTimer);
    settingsTimer = window.setTimeout(() => {
      settingsTimer = undefined;
      void persistSettings(revision);
    }, 250);
  }

  async function persistSettings(revision = settingsRevision, refreshRuntime = false): Promise<boolean> {
    if (!project || capturing || processing) return false;
    const snapshot = project;
    // Invalidate status requests that started before this write. A delayed
    // captureStatus response must never replace newer local camera settings.
    projectGeneration += 1;
    try {
      if (previewing) {
        await stopSensorPreview();
        sensor = null;
        packedPreviewFrame = null;
        liveMesh = null;
        lastPreviewFrame = 0;
        lastMeshFrame = 0;
      }
      const saved = await updateProjectSettings(snapshot.path, snapshot.settings);
      if (revision === settingsRevision) {
        project = saved;
        settingsDirty = false;
        projectGeneration += 1;
        if (refreshRuntime) runtime = await runtimeInfo().catch(() => runtime);
        await ensureSensorPreview();
      }
      return true;
    } catch (error) {
      message = errorText(error);
      return false;
    }
  }

  function sensorKey(candidate: AvailableSensor): string {
    return [candidate.kind, candidate.id, candidate.connection, candidate.address].join('|');
  }

  function configuredSensorKey(settings: CaptureSettings): string {
    return [settings.sensorKind, settings.sensorId, settings.sensorConnection, settings.sensorAddress].join('|');
  }

  async function ensureSensorPreview(): Promise<void> {
    if (!project || sensor?.capturing || sensor?.previewing || processing || project.settings.sensorKind === 'kinect_v2') return;
    const configured = sensors.find((candidate) => sensorKey(candidate) === configuredSensorKey(project!.settings));
    if (!configured?.connected) return;
    const revision = settingsRevision;
    const snapshot = project;
    projectGeneration += 1;
    message = `Starting live preview from ${configured.name}â€¦`;
    try {
      const started = await startSensorPreview(snapshot.path, snapshot.settings);
      if (revision !== settingsRevision) return;
      project = started;
      settingsDirty = false;
      projectGeneration += 1;
      sensor = await captureStatus();
      await pollLiveGeometry();
      message = `${configured.name} connected Â· live preview ready.`;
    } catch (error) {
      message = `Camera preview: ${errorText(error)}`;
    }
  }

  function settingsForSensor(candidate: AvailableSensor): CaptureSettings {
    if (!project) throw new Error('No active project');
    const changingFamily = project.settings.sensorKind !== candidate.kind;
    return {
      ...project.settings,
      sensorKind: candidate.kind,
      sensorId: candidate.id,
      sensorConnection: candidate.connection,
      sensorAddress: candidate.address,
      useImu: candidate.supportsImu && (changingFamily ? true : project.settings.useImu)
    };
  }

  function settingsForSensorKind(kind: SensorKind): CaptureSettings {
    if (!project) throw new Error('No active project');
    if (project.settings.sensorKind === kind) return project.settings;
    return {
      ...project.settings,
      sensorKind: kind,
      sensorId: '',
      sensorConnection: 'usb',
      sensorAddress: '',
      useImu: kind !== 'kinect_v2'
    };
  }

  async function commitSensorSettings(settings: CaptureSettings): Promise<boolean> {
    if (!project || capturing || processing || selectingSensor) return false;
    selectingSensor = true;
    settingsRevision += 1;
    projectGeneration += 1;
    settingsDirty = true;
    const revision = settingsRevision;
    if (settingsTimer) window.clearTimeout(settingsTimer);
    settingsTimer = undefined;
    project = { ...project, settings };
    try {
      return await persistSettings(revision, true);
    } finally {
      selectingSensor = false;
    }
  }

  async function chooseSensor(event: Event): Promise<void> {
    if (!project) return;
    const candidate = sensors.find((item) => sensorKey(item) === inputValue(event));
    if (!candidate) return;
    if (await commitSensorSettings(settingsForSensor(candidate))) {
      message = candidate.connected
        ? `${candidate.name} connected and selected.`
        : `${candidate.name} selected. It will be checked when capture starts.`;
    }
  }

  async function chooseSensorKind(event: Event): Promise<void> {
    const kind = inputValue(event) as SensorKind;
    if (await commitSensorSettings(settingsForSensorKind(kind))) {
      message = runtime?.sensorStatus ?? 'Capture source updated.';
    }
  }

  async function selectSupportedFallback(): Promise<string | null> {
    if (!project || !runtime || runtime.sensorWorkerAvailable) return null;
    const candidate = sensors.find((item) => runtime?.sensorCapabilities.includes(item.kind));
    if (candidate) {
      return await commitSensorSettings(settingsForSensor(candidate)) ? candidate.name : null;
    }
    const kind = runtime.sensorCapabilities[0];
    if (!kind) return null;
    return await commitSensorSettings(settingsForSensorKind(kind))
      ? kind.replaceAll('_', ' ')
      : null;
  }

  async function discoverSensors(): Promise<void> {
    if (discovering || capturing || processing) return;
    discovering = true;
    message = 'Refreshing RGB-D capture sources…';
    try {
      [sensors, runtime] = await Promise.all([availableSensors(), runtimeInfo()]);
      lastSensorScanAt = performance.now();
      const fallback = await selectSupportedFallback();
      const selectedSettings = project?.settings;
      const selected = selectedSettings
        ? sensors.find((candidate) => sensorKey(candidate) === configuredSensorKey(selectedSettings))
        : null;
      message = fallback
        ? `${fallback} selected because the previous camera backend is not installed.`
        : selected?.connected
          ? `${selected.name} connected.`
          : sensors.length
            ? `${sensors.length} RGB-D source${sensors.length === 1 ? '' : 's'} available.`
            : runtime.sensorStatus;
    } catch (error) {
      message = errorText(error);
    } finally {
      discovering = false;
    }
  }

  async function refreshSensorConnections(): Promise<void> {
    if (sensorScanInFlight || discovering || busy || liveSensor || processing || selectingSensor) return;
    const now = performance.now();
    if (now - lastSensorScanAt < 8000) return;
    sensorScanInFlight = true;
    try {
      sensors = await availableSensors();
    } catch {
      // Keep the last known list. Manual refresh reports discovery errors.
    } finally {
      lastSensorScanAt = performance.now();
      sensorScanInFlight = false;
    }
  }

  function parsePointPacket(buffer: ArrayBuffer): PackedPreviewFrame | null {
    if (buffer.byteLength < 24) return null;
    const bytes = new Uint8Array(buffer);
    if (new TextDecoder().decode(bytes.subarray(0, 4)) !== 'K2P1') return null;
    const view = new DataView(buffer);
    const frameCount = view.getUint32(4, true);
    const pointCount = view.getUint32(20, true);
    if (pointCount > 150_000 || buffer.byteLength !== 24 + pointCount * 15) return null;
    const positions = new Float32Array(pointCount * 3);
    const colors = new Uint8Array(pointCount * 3);
    for (let point = 0, source = 24, target = 0; point < pointCount; point += 1, source += 15, target += 3) {
      positions[target] = view.getFloat32(source, true);
      positions[target + 1] = view.getFloat32(source + 4, true);
      positions[target + 2] = view.getFloat32(source + 8, true);
      colors[target] = bytes[source + 12];
      colors[target + 1] = bytes[source + 13];
      colors[target + 2] = bytes[source + 14];
    }
    return { frameCount, pointCount, positions, colors };
  }

  async function pollLiveGeometry(): Promise<void> {
    if (!project || !liveSensor || geometryInFlight) return;
    geometryInFlight = true;
    try {
      const packet = parsePointPacket(await loadLivePreviewFrame(lastPreviewFrame));
      if (packet && packet.frameCount > lastPreviewFrame) {
        packedPreviewFrame = packet;
        if (capturing) captureDraftFrame = packet;
        lastPreviewFrame = packet.frameCount;
      }
      if (capturing && project.settings.liveReconstruction === 'mesh') {
        const result = await loadLiveReconstructionMesh(lastMeshFrame);
        if (result && result.frameCount > lastMeshFrame) {
          liveMesh = result.mesh;
          lastMeshFrame = result.frameCount;
        }
      }
    } catch (error) {
      message = `Live geometry: ${errorText(error)}`;
    } finally {
      geometryInFlight = false;
    }
  }

  async function loadBuildPreview(): Promise<void> {
    if (!project || !processing || resultInFlight || performance.now() - lastBuildPreviewAt < 3000) return;
    lastBuildPreviewAt = performance.now();
    resultInFlight = true;
    try {
      if (activeJob?.stage.includes('splat')) {
        const next = await loadGaussianSplat(project.path).catch(() => null);
        if (next?.byteLength) {
          const signature = splatSignature(next);
          if (signature !== lastBuildSplatSignature) {
            lastBuildSplatSignature = signature;
            previewSplat = next;
          }
          renderMode = 'splat';
        }
      } else {
        const next = await loadPreview(project.path).catch(() => []);
        if (next.length) {
          previewPoints = next;
          renderMode = 'points';
        }
      }
    } finally {
      resultInFlight = false;
    }
  }

  async function loadResult(mode: RenderMode, force = false): Promise<void> {
    if (!project || resultInFlight) return;
    renderMode = mode;
    if (mode === 'splat') {
      editMode = false;
      floorPickMode = false;
    }
    if (!force) {
      if (mode === 'points' && previewPoints.length) return;
      if (mode === 'mesh' && previewMesh) return;
      if (mode === 'splat' && previewSplat) return;
    }
    if (!artifactReady(mode === 'points' ? 'pointCloud' : mode === 'mesh' ? 'texturedMesh' : 'gaussianSplat')) return;
    resultInFlight = true;
    assetLoading = mode;
    try {
      if (mode === 'points') previewPoints = await loadPreview(project.path);
      else if (mode === 'mesh') previewMesh = await loadPreviewMesh(project.path);
      else previewSplat = await loadGaussianSplat(project.path);
    } catch (error) {
      message = errorText(error);
    } finally {
      assetLoading = null;
      resultInFlight = false;
    }
  }

  async function refreshCompletedJob(job: ArtifactJob): Promise<void> {
    if (!project || completedJobId === job.id) return;
    completedJobId = job.id;
    project = await currentProject();
    message = `Reconstruction complete${job.computeBackend ? ` · ${job.computeBackend}` : ''}.`;
    workspace = 'inspect';
    if (job.targets.includes('gaussianSplat')) await loadResult('splat', true);
    else if (job.targets.includes('texturedMesh')) await loadResult('mesh', true);
    else await loadResult('points', true);
  }

  async function pollStatus(forceCapture = false): Promise<void> {
    if (statusInFlight || !project || (busy && !forceCapture)) return;
    statusInFlight = true;
    const generation = projectGeneration;
    const mayApplyProject = !settingsDirty;
    try {
      const wasCapturing = capturing;
      if (forceCapture || wasCapturing || previewing) {
        const next = await captureStatus();
        if (generation !== projectGeneration || !mayApplyProject || settingsDirty || busy) return;
        sensor = next;
        project = next.project;
        if (next.error) message = next.error;
        if (!next.capturing && !next.previewing && wasCapturing && !next.error) {
          message = next.frameCount
            ? `Capture saved · ${next.frameCount.toLocaleString()} archived frames.`
            : 'Capture stopped without usable archived frames.';
        }
      }

      if (activeJob && ['queued', 'running', 'cancelling'].includes(activeJob.status)) {
        activeJob = await artifactJobStatus(project.path, activeJob.id);
        if (activeJob.status === 'complete') await refreshCompletedJob(activeJob);
        else if (activeJob.status === 'failed') message = activeJob.error ?? 'Reconstruction failed.';
        else await loadBuildPreview();
      }
      if (!wasCapturing && !previewing && !processing) await refreshSensorConnections();
    } catch (error) {
      message = errorText(error);
    } finally {
      statusInFlight = false;
    }
  }

  async function captureAction(): Promise<void> {
    if (!project || busy || processing || photoLocalizationActive) return;
    busy = true;
    projectGeneration += 1;
    try {
      if (capturing) {
        message = 'Finishing the archive and draining reconstruction queues…';
        project = await stopSensorPhase();
        settingsDirty = false;
        projectGeneration += 1;
        const completed = project.phases[project.phases.length - 1];
        const savedDraft = parsePointPacket(await loadCaptureDraft(project.path));
        if (savedDraft) captureDraftFrame = savedDraft;
        sensor = await captureStatus();
        packedPreviewFrame = null;
        liveMesh = null;
        lastPreviewFrame = 0;
        lastMeshFrame = 0;
        await ensureSensorPreview();
        if (sensor?.previewing) {
          message = completed?.frameCount
            ? `Capture saved · ${completed.frameCount.toLocaleString()} archived frames · live camera preview resumed.`
            : 'Capture stopped without usable archived frames · live camera preview resumed.';
        }
      } else {
        if (settingsTimer) window.clearTimeout(settingsTimer);
        settingsTimer = undefined;
        message = previewing ? 'Recording started.' : 'Opening the camera and starting recording…';
        project = await startSensorPhase(project.path, project.settings);
        settingsDirty = false;
        projectGeneration += 1;
        captureDraftFrame = null;
        packedPreviewFrame = null;
        liveMesh = null;
        lastPreviewFrame = 0;
        lastMeshFrame = 0;
        sensor = await captureStatus();
        await pollLiveGeometry();
        message = 'Recording started · building the quality-gated live reconstruction.';
      }
    } catch (error) {
      message = errorText(error);
    } finally {
      busy = false;
    }
  }

  async function showCaptureWorkspace(): Promise<void> {
    workspace = 'capture';
    editMode = false;
    floorPickMode = false;
    await ensureSensorPreview();
  }

  async function showReconstructWorkspace(): Promise<void> {
    if (!project || capturing) return;
    workspace = 'reconstruct';
    editMode = false;
    floorPickMode = false;
    if (previewing) {
      await stopSensorPreview();
      sensor = null;
      packedPreviewFrame = null;
      liveMesh = null;
      lastPreviewFrame = 0;
      lastMeshFrame = 0;
    }
    if (!processing && readyArtifacts === 0) {
      const savedDraft = parsePointPacket(await loadCaptureDraft(project.path));
      if (savedDraft) {
        captureDraftFrame = savedDraft;
        message = 'Showing the preserved live reconstruction draft.';
      }
    }
  }

  async function showInspectWorkspace(): Promise<void> {
    if (!project || capturing || readyArtifacts === 0) return;
    workspace = 'inspect';
    if (previewing) {
      await stopSensorPreview();
      sensor = null;
      packedPreviewFrame = null;
      liveMesh = null;
      lastPreviewFrame = 0;
      lastMeshFrame = 0;
    }
    if (artifactReady('texturedMesh')) await loadResult('mesh');
    else if (artifactReady('pointCloud')) await loadResult('points');
    else if (artifactReady('gaussianSplat')) await loadResult('splat');
  }

  async function removeCaptureAction(id: string, name: string): Promise<void> {
    if (busy || processing || capturing || !window.confirm(`Delete ${name} and invalidate all reconstructed outputs?`)) return;
    busy = true;
    try {
      project = await removeCapture(id);
      previewPoints = [];
      previewMesh = null;
      previewSplat = null;
      captureDraftFrame = parsePointPacket(await loadCaptureDraft(project.path));
      activeJob = null;
      message = `${name} deleted. Existing reconstruction outputs were invalidated.`;
    } catch (error) {
      message = errorText(error);
    } finally {
      busy = false;
    }
  }

  async function startBuild(resume = false): Promise<void> {
    if (!project || busy || capturing || photoLocalizationActive) return;
    if (resume && activeJob) {
      busy = true;
      try {
        if (previewing) {
          await stopSensorPreview();
          sensor = null;
        }
        activeJob = await resumeArtifactJob(project.path, activeJob.id);
        workspace = 'reconstruct';
        message = 'Resuming the saved 2D Gaussian checkpoint…';
      } catch (error) {
        message = errorText(error);
      } finally {
        busy = false;
      }
      return;
    }
    const targets: ArtifactTarget[] = [];
    if (buildPointCloud) targets.push('pointCloud');
    if (buildTexturedMesh) targets.push('texturedMesh');
    if (buildGaussianSplat) targets.push('gaussianSplat');
    if (!targets.length) {
      message = 'Choose at least one reconstruction output.';
      return;
    }
    busy = true;
    try {
      if (previewing) {
        await stopSensorPreview();
        sensor = null;
        packedPreviewFrame = null;
        liveMesh = null;
      }
      completedJobId = '';
      if (targets.includes('gaussianSplat')) {
        previewSplat = null;
        lastBuildSplatSignature = '';
      }
      activeJob = await startArtifactJob(project.path, targets, splatIterations);
      workspace = 'reconstruct';
      message = mediaOnlyProject
        ? 'Started photo/video camera solving and photoreal Gaussian reconstruction.'
        : 'Started quality-gated RGB-D reconstruction.';
    } catch (error) {
      message = errorText(error);
    } finally {
      busy = false;
    }
  }

  async function addMediaSource(): Promise<void> {
    if (!project || busy || capturing || processing) return;
    const selected = await open({
      title: 'Choose overlapping photos or video',
      multiple: true,
      directory: false,
      filters: [
        { name: 'Photos and video', extensions: ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp', 'bmp', 'mp4', 'mov', 'm4v', 'avi', 'mkv', 'webm', 'mts', 'm2ts'] }
      ]
    });
    const paths = Array.isArray(selected) ? selected : selected ? [selected] : [];
    if (!paths.length) return;
    busy = true;
    try {
      if (previewing) {
        await stopSensorPreview();
        sensor = null;
        packedPreviewFrame = null;
      }
      project = await importMediaSources(project.path, paths);
      buildPointCloud = false;
      buildTexturedMesh = false;
      buildGaussianSplat = true;
      workspace = 'reconstruct';
      message = `Imported ${paths.length} source${paths.length === 1 ? '' : 's'}. Ready to solve cameras and train a photoreal 3D Gaussian splat.`;
    } catch (error) {
      message = errorText(error);
    } finally {
      busy = false;
    }
  }

  async function refreshTexturePhotos(projectPath: string): Promise<void> {
    const manifest = await supplementalPhotos(projectPath);
    if (project?.path === projectPath) texturePhotos = manifest.attempts;
  }

  function stopPhotoProgressPolling(): void {
    if (photoProgressTimer) window.clearInterval(photoProgressTimer);
    photoProgressTimer = undefined;
  }

  async function pollTexturePhotoProgress(projectPath: string): Promise<void> {
    if (photoProgressInFlight || project?.path !== projectPath) return;
    photoProgressInFlight = true;
    try {
      const progress = await supplementalPhotoProgress(projectPath);
      if (project?.path !== projectPath || !progress) return;
      texturePhotoProgress = progress;
      photoLocalizationActive = progress.status === 'running';
      if (progress.processedPhotos !== photoProgressProcessed) {
        photoProgressProcessed = progress.processedPhotos;
        await refreshTexturePhotos(projectPath);
      }
      if (progress.status === 'running') {
        message = progress.detail;
      } else {
        stopPhotoProgressPolling();
        if (progress.status === 'failed') message = progress.detail;
        project = await currentProject();
      }
    } catch (error) {
      stopPhotoProgressPolling();
      photoLocalizationActive = false;
      message = errorText(error);
    } finally {
      photoProgressInFlight = false;
    }
  }

  function startPhotoProgressPolling(projectPath: string): void {
    stopPhotoProgressPolling();
    photoProgressTimer = window.setInterval(
      () => void pollTexturePhotoProgress(projectPath),
      500
    );
  }

  async function addTexturePhotos(): Promise<void> {
    if (!project || busy || capturing || processing || photoLocalizationActive) return;
    const selected = await open({
      title: 'Choose high-resolution scene photos',
      multiple: true,
      directory: false,
      filters: [{ name: 'Scene photos', extensions: ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp'] }]
    });
    const paths = Array.isArray(selected) ? selected : selected ? [selected] : [];
    if (!paths.length) return;
    busy = true;
    photoLocalizationActive = true;
    photoProgressProcessed = -1;
    texturePhotoProgress = {
      schemaVersion: 1,
      status: 'running',
      stage: 'starting',
      detail: `Starting localization for ${paths.length} high-resolution photo${paths.length === 1 ? '' : 's'}`,
      progress: 0,
      processedPhotos: 0,
      totalPhotos: paths.length,
      localizedPhotos: 0,
      failedPhotos: 0
    };
    startPhotoProgressPolling(project.path);
    message = `Localizing ${paths.length} high-resolution photo${paths.length === 1 ? '' : 's'} against the RGB-D scan…`;
    try {
      const result = await localizeSupplementalPhotos(project.path, paths);
      project = await currentProject();
      await refreshTexturePhotos(project.path);
      texturePhotoProgress = await supplementalPhotoProgress(project.path);
      buildTexturedMesh = true;
      if (result.localizedPhotoCount) {
        const quality = result.localized
          .map((photo) => `${photo.name}: ${photo.inlierCount} inliers, ${photo.reprojectionRmsePixels.toFixed(2)} px`)
          .join(' · ');
        const rejected = result.failedPhotoCount
          ? ` ${result.failedPhotoCount} ambiguous photo${result.failedPhotoCount === 1 ? ' was' : 's were'} rejected and will not be baked.`
          : '';
        message = `${result.localizedPhotoCount} photo${result.localizedPhotoCount === 1 ? '' : 's'} localized.${rejected} Rebuild the textured mesh to use them. ${quality}`;
      } else {
        message = result.failures.map((failure) => `${failure.path}: ${failure.error}`).join(' · ');
      }
    } catch (error) {
      message = errorText(error);
      texturePhotoProgress = await supplementalPhotoProgress(project.path).catch(() => null);
    } finally {
      busy = false;
      photoLocalizationActive = false;
      stopPhotoProgressPolling();
    }
  }

  async function removeTexturePhoto(photo: SupplementalPhotoAttempt): Promise<void> {
    if (!project || busy || processing || photoLocalizationActive) return;
    const consequence = photo.status === 'localized'
      ? 'It will no longer be used for texture baking. The original photo will not be deleted.'
      : 'This removes it from the rejected-photo history. The original photo will not be deleted.';
    if (!window.confirm(`Remove ${photo.name}? ${consequence}`)) return;
    busy = true;
    try {
      const manifest = await removeSupplementalPhoto(project.path, photo.id);
      texturePhotos = manifest.attempts;
      project = await currentProject();
      if (photo.status === 'localized') buildTexturedMesh = true;
      message = photo.status === 'localized'
        ? `${photo.name} removed. Rebuild the textured mesh to bake without it.`
        : `${photo.name} removed from the rejected-photo history.`;
    } catch (error) {
      message = errorText(error);
    } finally {
      busy = false;
    }
  }

  async function cancelBuild(): Promise<void> {
    if (!project || !activeJob || !processing) return;
    try {
      activeJob = await cancelArtifactJob(project.path, activeJob.id);
      message = 'Cancelling workers and saving the Gaussian checkpoint when available…';
    } catch (error) {
      message = errorText(error);
    }
  }

  async function discardBuild(): Promise<void> {
    if (!project || !activeJob || processing || !window.confirm('Discard this interrupted job and its saved checkpoint? Finished artifacts remain untouched.')) return;
    try {
      activeJob = await discardArtifactJob(project.path, activeJob.id);
      activeJob = null;
      project = await currentProject();
      message = 'Interrupted job discarded.';
    } catch (error) {
      message = errorText(error);
    }
  }

  async function refreshProjectCatalog(): Promise<void> {
    projectCatalogLoading = true;
    projectManagerError = '';
    try {
      projectCatalog = await listProjects();
    } catch (error) {
      projectManagerError = errorText(error);
    } finally {
      projectCatalogLoading = false;
    }
  }

  function showProjectManager(showNewProject = false): void {
    projectManagerOpen = true;
    projectManagerError = '';
    projectNameDraft = project?.name ?? '';
    newProjectName = '';
    newProjectFormOpen = showNewProject;
    void refreshProjectCatalog();
  }

  function closeProjectManager(): void {
    if (busy) return;
    projectManagerOpen = false;
    projectManagerError = '';
    newProjectFormOpen = false;
  }

  async function activateProject(next: ProjectSummary, statusMessage: string): Promise<void> {
    if (settingsTimer) window.clearTimeout(settingsTimer);
    settingsTimer = undefined;
    settingsRevision += 1;
    projectGeneration += 1;
    project = next;
    settingsDirty = false;
    projectNameDraft = next.name;
    sensor = null;
    activeJob = null;
    previewPoints = [];
    previewMesh = null;
    previewSplat = null;
    packedPreviewFrame = null;
    captureDraftFrame = null;
    liveMesh = null;
    texturePhotos = [];
    texturePhotoProgress = null;
    photoLocalizationActive = false;
    photoProgressProcessed = -1;
    completedJobId = '';
    lastPreviewFrame = 0;
    lastMeshFrame = 0;
    lastBuildSplatSignature = '';
    stopPhotoProgressPolling();

    if (next.activeJob || next.processingStatus === 'failed') {
      workspace = 'reconstruct';
      activeJob = await latestArtifactJob(next.path).catch(() => null);
      if (readyArtifactCount(next) === 0) {
        captureDraftFrame = parsePointPacket(await loadCaptureDraft(next.path).catch(() => new ArrayBuffer(0)));
      }
    } else {
      workspace = 'capture';
      await ensureSensorPreview();
    }
    message = statusMessage;
  }

  async function saveProjectAction(): Promise<void> {
    if (!project || busy || capturing || processing || photoLocalizationActive) return;
    projectManagerError = '';
    busy = true;
    try {
      if (settingsDirty && !(await persistSettings())) {
        throw new Error('Project settings could not be saved');
      }
      project = await saveProject(project.path, projectNameDraft);
      projectNameDraft = project.name;
      await refreshProjectCatalog();
      message = `${project.name} saved.`;
    } catch (error) {
      projectManagerError = errorText(error);
      message = projectManagerError;
    } finally {
      busy = false;
    }
  }

  async function openProjectAction(entry: ProjectCatalogEntry): Promise<void> {
    if (!project || busy || capturing || processing || photoLocalizationActive) return;
    if (entry.id === project.id) {
      closeProjectManager();
      return;
    }
    if (transformDirty && transformSaveMode === 'manual'
      && !window.confirm('Discard the unsaved model pose and open another project?')) return;
    projectManagerError = '';
    busy = true;
    try {
      if (settingsDirty && !(await persistSettings())) {
        throw new Error('Save the current project settings before switching projects');
      }
      const opened = await openProject(entry.path);
      await activateProject(opened, `${opened.name} opened.`);
      projectManagerOpen = false;
    } catch (error) {
      projectManagerError = errorText(error);
      message = projectManagerError;
    } finally {
      busy = false;
    }
  }

  async function deleteProjectAction(entry: ProjectCatalogEntry): Promise<void> {
    if (!project || busy || capturing || processing || photoLocalizationActive) return;
    const details = entry.captureCount || entry.mediaSourceCount || entry.artifactCount
      ? 'Its captures, imported media, and reconstructed outputs will be permanently deleted.'
      : 'The empty project folder will be permanently deleted.';
    if (!window.confirm(`Delete “${entry.name}”? ${details}`)) return;
    if (entry.id === project.id && transformDirty && transformSaveMode === 'manual'
      && !window.confirm('This project also has an unsaved model pose. Delete it anyway?')) return;
    projectManagerError = '';
    busy = true;
    try {
      const deletingCurrent = entry.id === project.id;
      const active = await deleteProject(entry.path);
      localStorage.removeItem(transformStorageKey(entry.id));
      if (deletingCurrent) await activateProject(active, `${entry.name} deleted. ${active.name} is now open.`);
      else message = `${entry.name} deleted.`;
      await refreshProjectCatalog();
      projectNameDraft = active.name;
    } catch (error) {
      projectManagerError = errorText(error);
      message = projectManagerError;
    } finally {
      busy = false;
    }
  }

  async function newProjectAction(): Promise<void> {
    if (busy || capturing || processing || photoLocalizationActive) return;
    projectManagerError = '';
    busy = true;
    try {
      const created = await createProject(newProjectName);
      await activateProject(created, `${created.name} created.`);
      newProjectName = '';
      newProjectFormOpen = false;
      await refreshProjectCatalog();
      projectManagerOpen = false;
      await discoverSensors();
    } catch (error) {
      projectManagerError = errorText(error);
      message = projectManagerError;
    } finally {
      busy = false;
    }
  }

  async function exportPointCloud(): Promise<void> {
    if (!project || !artifactReady('pointCloud')) return;
    const destination = await save({ title: 'Export metric point cloud', defaultPath: 'scan-cloud.ply', filters: [{ name: 'PLY point cloud', extensions: ['ply'] }] });
    if (!destination) return;
    try {
      message = `Aligned point cloud exported to ${await exportPly(project.path, destination, cloudTransform)}.`;
    } catch (error) { message = errorText(error); }
  }

  async function exportMesh(): Promise<void> {
    if (!project || !artifactReady('texturedMesh')) return;
    const destination = await save({ title: 'Export textured mesh bundle', defaultPath: 'scan-mesh.obj', filters: [{ name: 'Wavefront OBJ', extensions: ['obj'] }] });
    if (!destination) return;
    try {
      message = `Aligned OBJ, MTL, and texture exported beside ${await exportTexturedMesh(project.path, destination, cloudTransform)}.`;
    } catch (error) { message = errorText(error); }
  }

  async function exportSplat(): Promise<void> {
    if (!project || !artifactReady('gaussianSplat')) return;
    const destination = await save({ title: 'Export metric 2D Gaussian surface', defaultPath: 'scan-2dgs.ply', filters: [{ name: 'Gaussian PLY', extensions: ['ply'] }] });
    if (!destination) return;
    try {
      message = `Gaussian surface and coordinate sidecars exported to ${await exportGaussianSplat(project.path, destination)}.`;
    } catch (error) { message = errorText(error); }
  }

  onMount(() => {
    loadEditorPreferences();
    void (async () => {
      try {
        project = await currentProject();
        [runtime, sensors] = await Promise.all([
          runtimeInfo().catch(() => null),
          availableSensors().catch(() => [])
        ]);
        lastSensorScanAt = performance.now();
        const fallback = await selectSupportedFallback();
        sensor = await captureStatus().catch(() => null);
        if (project.activeJob || project.processingStatus === 'processing' || project.processingStatus === 'failed') {
          activeJob = await latestArtifactJob(project.path).catch(() => null);
          if (activeJob) workspace = 'reconstruct';
        }
        if (workspace === 'reconstruct' && readyArtifacts === 0) {
          captureDraftFrame = parsePointPacket(await loadCaptureDraft(project.path));
        }
        message = fallback
          ? `${fallback} selected because the previous camera backend is not installed.`
          : runtime?.sensorStatus ?? 'RGB-D workspace ready.';
        await refreshTexturePhotos(project.path);
        texturePhotoProgress = await supplementalPhotoProgress(project.path);
        if (texturePhotoProgress?.status === 'running') {
          photoLocalizationActive = true;
          photoProgressProcessed = texturePhotoProgress.processedPhotos;
          workspace = 'reconstruct';
          message = texturePhotoProgress.detail;
          startPhotoProgressPolling(project.path);
        }
        if (workspace === 'capture' && !processing) await ensureSensorPreview();
        statusTimer = window.setInterval(() => void pollStatus(), 300);
        geometryTimer = window.setInterval(() => void pollLiveGeometry(), 75);
      } catch (error) {
        fatalError = errorText(error);
        message = fatalError;
      }
    })();
  });

  onDestroy(() => {
    if (statusTimer) window.clearInterval(statusTimer);
    if (geometryTimer) window.clearInterval(geometryTimer);
    if (settingsTimer) window.clearTimeout(settingsTimer);
    stopPhotoProgressPolling();
  });
</script>

<svelte:head><title>ScanLan · Gaussian Reconstruction</title></svelte:head>
<svelte:window on:keydown={handleEditShortcut} />

<div class="app-shell">
  <header class="topbar">
    <div class="brand"><span class="brand-mark">SL</span><div><strong>ScanLan</strong><small>Photos · video · RGB-D reconstruction</small></div></div>
    <button class="project-title" on:click={() => showProjectManager()} disabled={!project} title="Manage projects"><span>ACTIVE PROJECT</span><strong>{project?.name ?? 'Loading…'}</strong></button>
    <div class="runtime-state">
      <span class:ready={Boolean(runtime?.sensorWorkerAvailable)}><i></i>Capture</span>
      <span class:ready={Boolean(runtime?.reconstructionWorkerAvailable)}><i></i>Reconstruct</span>
      <span class:ready={Boolean(runtime?.splatWorkerAvailable)}><i></i>Gaussian CUDA</span>
    </div>
    <div class="header-actions">
      <button class="ghost compact" on:click={() => showProjectManager()} disabled={busy || !project}>Projects</button>
      <button class="ghost compact" on:click={() => showProjectManager(true)} disabled={busy || capturing || processing || photoLocalizationActive}>New project</button>
    </div>
  </header>

  {#if projectManagerOpen && project}
    <div class="modal-backdrop">
      <button class="modal-dismiss" aria-label="Close project manager" on:click={closeProjectManager}></button>
      <div class="project-manager" role="dialog" aria-modal="true" aria-labelledby="project-manager-title">
        <header>
          <div><span>PROJECT LIBRARY</span><h2 id="project-manager-title">Manage projects</h2></div>
          <button class="dialog-close" on:click={closeProjectManager} disabled={busy} aria-label="Close">×</button>
        </header>

        <div class="current-project-editor">
          <div><span>ACTIVE PROJECT</span><small>Settings, captures, and outputs are saved automatically.</small></div>
          <form on:submit|preventDefault={saveProjectAction}>
            <input bind:value={projectNameDraft} maxlength="80" aria-label="Active project name" disabled={busy || capturing || processing}/>
            <button class="primary" type="submit" disabled={busy || capturing || processing || !projectNameDraft.trim()}>Save name</button>
          </form>
        </div>

        <div class="project-library-heading">
          <strong>All projects</strong>
          <span>{projectCatalog.length} project{projectCatalog.length === 1 ? '' : 's'}</span>
        </div>
        <div class="project-library" class:loading={projectCatalogLoading}>
          {#if projectCatalogLoading && projectCatalog.length === 0}
            <div class="project-library-empty"><div class="spinner"></div><p>Loading projects…</p></div>
          {:else if projectCatalog.length === 0}
            <div class="project-library-empty"><p>No saved projects yet.</p></div>
          {:else}
            {#each projectCatalog as entry (entry.id)}
              <article class:active={entry.id === project.id}>
                <div class="project-library-icon">{entry.name.slice(0, 2).toUpperCase()}</div>
                <div class="project-library-copy">
                  <div><strong>{entry.name}</strong>{#if entry.id === project.id}<span>CURRENT</span>{/if}</div>
                  <small>Updated {formatProjectDate(entry.modifiedAt)}</small>
                  <p>{entry.captureCount} capture{entry.captureCount === 1 ? '' : 's'} · {entry.mediaSourceCount} media · {entry.artifactCount} output{entry.artifactCount === 1 ? '' : 's'}{entry.frameCount ? ` · ${entry.frameCount.toLocaleString()} frames` : ''}</p>
                </div>
                <div class="project-library-actions">
                  <button class="ghost" on:click={() => openProjectAction(entry)} disabled={busy || entry.id === project.id}>{entry.id === project.id ? 'Current' : 'Open'}</button>
                  <button class="danger" on:click={() => deleteProjectAction(entry)} disabled={busy}>Remove</button>
                </div>
              </article>
            {/each}
          {/if}
        </div>

        {#if newProjectFormOpen}
          <form class="new-project-form" on:submit|preventDefault={newProjectAction}>
            <div><span>NEW PROJECT</span><small>Your current project stays safely on disk.</small></div>
            <input bind:value={newProjectName} maxlength="80" placeholder="Project name" aria-label="New project name" disabled={busy}/>
            <button class="primary" type="submit" disabled={busy || !newProjectName.trim()}>Create</button>
            <button class="ghost" type="button" on:click={() => newProjectFormOpen = false} disabled={busy}>Cancel</button>
          </form>
        {:else}
          <button class="new-project-button" on:click={() => newProjectFormOpen = true} disabled={busy || capturing || processing || photoLocalizationActive}>+ New project</button>
        {/if}
        {#if projectManagerError}<p class="project-manager-error">{projectManagerError}</p>{/if}
      </div>
    </div>
  {/if}

  <nav class="workflow" aria-label="Workflow">
    <button class:active={workspace === 'capture'} class:done={completedCaptures > 0} on:click={() => void showCaptureWorkspace()}>
      <span>01</span><div><strong>Capture</strong><small>{capturing ? 'Recording now' : `${completedCaptures} take${completedCaptures === 1 ? '' : 's'}`}</small></div>
    </button>
    <button class:active={workspace === 'reconstruct'} class:done={readyArtifacts > 0} on:click={() => void showReconstructWorkspace()} disabled={capturing}>
      <span>02</span><div><strong>Reconstruct</strong><small>{processing ? activeJob?.stage.replaceAll('_', ' ') : 'Points · mesh · 2DGS'}</small></div>
    </button>
    <button class:active={workspace === 'inspect'} class:done={readyArtifacts > 0} on:click={() => void showInspectWorkspace()} disabled={capturing || (!processing && readyArtifacts === 0)}>
      <span>03</span><div><strong>Edit & export</strong><small>{readyArtifacts ? `${readyArtifacts} output${readyArtifacts === 1 ? '' : 's'} ready` : 'No output yet'}</small></div>
    </button>
  </nav>

  <main>
    <section class="viewport">
      <PointCloudPreview
        points={viewerPackedFrame ? [] : previewPoints}
        packedFrame={viewerPackedFrame}
        processing={processing}
        live={liveSensor || Boolean(viewerPackedFrame)}
        liveLabel={previewing ? 'Live camera point cloud' : liveSensor ? 'Live reconstruction' : 'Saved live reconstruction draft'}
        emptyDetail={liveSensor ? sensor?.trackingStatus ?? '' : ''}
        pointSize={0.026}
        opacity={0.95}
        showColors={true}
        renderMode={viewerRenderMode}
        mesh={viewerMesh}
        splatBytes={liveSensor ? null : previewSplat}
        {meshViewMode}
        assetLoading={assetLoading}
        floorPickMode={floorPickMode && canEditModel}
        cloudTransform={viewerCloudTransform}
        {gizmoAnchor}
        editMode={editMode && canEditModel}
        {gizmoMode}
        {rotationSnapDegrees}
        onFloorDetected={setFloorTransform}
        onFloorMessage={(value) => message = value}
        onTransformChanged={handleGizmoTransform}
        onTransformCommitted={commitGizmoTransform}
      />

      {#if liveSensor && sensor}
        <div class="live-metrics">
          <div><span>{capturing ? 'Tracking' : 'Camera'}</span><strong class:good={capturing ? sensor.tracking : sensor.sensorConnected}>{capturing ? sensor.tracking ? 'LOCKED' : 'SEARCHING' : sensor.sensorConnected ? 'CONNECTED' : 'WAITING'}</strong></div>
          <div><span>{capturing ? 'Tracker' : 'Stream'}</span><strong>{(capturing ? sensor.trackingFps : sensor.streamFps).toFixed(1)} fps</strong></div>
          <div><span>{capturing ? 'Keyframes' : 'Frames seen'}</span><strong>{capturing ? sensor.liveIntegratedFrameCount : sensor.liveProcessedFrameCount}</strong></div>
          <div><span>Overlap</span><strong>{Math.round(sensor.trackingOverlap * 100)}%</strong></div>
          <div><span>Depth error</span><strong>{sensor.depthRmseMm ? `${sensor.depthRmseMm.toFixed(1)} mm` : '—'}</strong></div>
          <div><span>Queue drops</span><strong>{sensor.trackingQueueDropCount + sensor.mappingDropCount}</strong></div>
        </div>
      {/if}

      {#if processing && activeJob}
        <div class="job-overlay">
          <div><span>{activeJob.stage.replaceAll('_', ' ')}</span><strong>{Math.round(activeJob.progress * 100)}%</strong></div>
          <div class="progress"><i style={`width:${Math.round(activeJob.progress * 100)}%`}></i></div>
          {#if activeJob.iteration !== null}
            <div class="job-quality">
              <span>Iteration <strong>{activeJob.iteration.toLocaleString()} / {activeJob.totalIterations?.toLocaleString()}</strong></span>
              <span>Current loss <strong>{activeJob.loss?.toFixed(4) ?? '—'}</strong></span>
              <span>Rolling loss <strong>{activeJob.smoothedLoss?.toFixed(4) ?? 'warming up'}</strong></span>
            </div>
          {/if}
          <p>{activeJob.detail}</p>
        </div>
      {/if}
    </section>

    <aside>
      {#if !project}
        <section class="panel"><div class="spinner"></div><h2>Starting ScanLan</h2><p>{message}</p></section>
      {:else if workspace === 'capture'}
        <section class="panel panel-heading">
          <div><span>RGB-D SOURCE</span><h2>{liveSensor ? sensor?.sensorName ?? 'Live camera' : 'Camera & live fusion'}</h2></div>
          <button class="icon-button" on:click={discoverSensors} disabled={discovering || selectingSensor || capturing || processing} title="Refresh cameras">↻</button>
        </section>

        <section class="panel settings">
          <label>Capture source
            <select value={currentSensorKey} on:change={chooseSensor} disabled={capturing || processing || discovering || selectingSensor}>
              {#if !sensors.some((item) => sensorKey(item) === currentSensorKey)}
                <option value={currentSensorKey}>{project.settings.sensorKind.replaceAll('_', ' ')} · configured</option>
              {/if}
              {#each sensors as candidate}
                <option value={sensorKey(candidate)}>{candidate.name}{candidate.connection === 'network' ? ` · ${candidate.address}` : ''}{candidate.connected ? ' · Connected' : ''}</option>
              {/each}
            </select>
          </label>
          <div class="setting-grid">
            <label>Camera family
              <select value={project.settings.sensorKind} on:change={chooseSensorKind} disabled={capturing || processing || selectingSensor}>
                <option value="kinect_v2">Kinect v2</option>
                <option value="azure_kinect">Azure Kinect DK</option>
                <option value="femto_mega">Femto Mega</option>
              </select>
            </label>
            <label>Realtime view
              <select value={project.settings.liveReconstruction} on:change={(event) => updateSetting('liveReconstruction', inputValue(event) as LiveReconstructionMode)} disabled={capturing || processing}>
                <option value="points">Points · fastest</option>
                <option value="mesh">Mesh · 1 Hz</option>
              </select>
            </label>
            <label>Archive rate
              <select value={project.settings.captureFps} on:change={(event) => updateSetting('captureFps', Number(inputValue(event)))} disabled={capturing || processing}>
                <option value={5}>5 fps</option><option value={10}>10 fps</option><option value={15}>15 fps</option><option value={30}>30 fps</option>
              </select>
            </label>
            <label>Depth limit
              <div class="unit-input"><input type="number" min="0.8" max="8" step="0.1" value={project.settings.maxDepthM} on:change={(event) => updateSetting('maxDepthM', Number(inputValue(event)))} disabled={capturing || processing}/><span>m</span></div>
            </label>
            <label>Fusion voxel
              <div class="unit-input"><input type="number" min="3" max="40" step="1" value={project.settings.voxelSizeMm} on:change={(event) => updateSetting('voxelSizeMm', Number(inputValue(event)))} disabled={capturing || processing}/><span>mm</span></div>
            </label>
            {#if project.settings.sensorKind !== 'kinect_v2'}
              <label>Depth FOV
                <select value={project.settings.depthFieldOfView} on:change={(event) => updateSetting('depthFieldOfView', inputValue(event) as DepthFieldOfView)} disabled={capturing || processing}>
                  <option value="narrow">Narrow</option><option value="wide">Wide</option>
                </select>
              </label>
              <label>Depth sampling
                <select value={project.settings.depthBinned ? 'binned' : 'full'} on:change={(event) => updateSetting('depthBinned', inputValue(event) === 'binned')} disabled={capturing || processing}>
                  <option value="full">Full resolution</option><option value="binned">2×2 binned · faster</option>
                </select>
              </label>
            {/if}
          </div>
          {#if project.settings.sensorKind === 'femto_mega'}
            <label>Connection
              <select value={project.settings.sensorConnection} on:change={(event) => updateSetting('sensorConnection', inputValue(event) as 'usb' | 'network')} disabled={capturing || processing}>
                <option value="usb">USB</option><option value="network">Network</option>
              </select>
            </label>
            {#if project.settings.sensorConnection === 'network'}
              <label>Camera address<input value={project.settings.sensorAddress} placeholder="192.168.1.10" on:change={(event) => updateSetting('sensorAddress', inputValue(event))} disabled={capturing || processing}/></label>
            {/if}
          {/if}
          <label class="toggle"><input type="checkbox" checked={project.settings.useImu} on:change={(event) => updateSetting('useImu', inputChecked(event))} disabled={capturing || processing || project.settings.sensorKind === 'kinect_v2'}/><span></span><div><strong>IMU motion prior</strong><small>Improves fast-rotation initialization</small></div></label>
        </section>

        {#if !capturing}
          <section class="panel connection-card" class:connected={selectedSensorConnected} class:warning={!selectedSensorConnected}>
            <div class="tracking-title"><i></i><div>
              <strong>{selectedSensorConnected ? 'Connected' : project.settings.sensorKind === 'kinect_v2' ? 'Ready to open' : 'Camera not connected'}</strong>
              <small>{selectedSensor?.name ?? project.settings.sensorKind.replaceAll('_', ' ')}</small>
            </div></div>
            <p>{selectedSensorConnected
              ? previewing
                ? `${sensor?.sensorStatus ?? 'Live point-cloud preview is active.'} Capture starts recording immediately.`
                : `${selectedSensor?.connection.toUpperCase()}${selectedSensor?.serial ? ` · ${selectedSensor.serial}` : ''} · Starting live point-cloud preview…`
              : project.settings.sensorKind === 'kinect_v2'
                ? 'Kinect v2 has no passive connection query; it is verified when capture starts.'
                : 'Plug in the selected camera or refresh the source list. Connection state is checked automatically.'}</p>
          </section>
        {/if}

        {#if liveSensor && sensor}
          <section class="panel tracking-card" class:warning={!sensor.tracking}>
            <div class="tracking-title"><i></i><div><strong>{sensor.trackingStatus}</strong><small>{sensor.liveReconstructionBackend ?? 'Realtime engine'}</small></div></div>
            <div class="mini-grid">
              <div><span>Sensor</span><strong>{sensor.streamFps.toFixed(1)} fps</strong></div>
              <div><span>{capturing ? 'Raw archive' : 'Recording'}</span><strong>{capturing ? sensor.frameCount : 'OFF'}</strong></div>
              <div><span>{capturing ? 'Tracked' : 'Frames seen'}</span><strong>{capturing ? Math.max(0, sensor.liveProcessedFrameCount - sensor.liveRejectedFrameCount) : sensor.liveProcessedFrameCount}</strong></div>
              <div><span>Rejected</span><strong>{sensor.liveRejectedFrameCount}</strong></div>
              <div><span>Source drops</span><strong>{sensor.sourceDropCount}</strong></div>
            </div>
            <p>Raw RGB-D stays recoverable for the offline pass. Rejected live poses never enter the fused map; hold a previously scanned view steady to relocalize.</p>
          </section>
        {/if}

        <button class:stop={capturing} class="capture-button" on:click={captureAction} disabled={busy || selectingSensor || processing || photoLocalizationActive || (!capturing && (mediaSourceCount > 0 || (runtime && !runtime.sensorWorkerAvailable)))}>
          <i></i><span>{capturing ? 'Stop & save take' : busy ? 'Starting recording…' : 'Start capture'}</span>
        </button>
        <button class="ghost full" on:click={addMediaSource} disabled={busy || capturing || processing || completedCaptures > 0 || !runtime?.splatWorkerAvailable}>Import photos or video for Gaussian splatting…</button>

        <section class="panel takes">
          <div class="section-title"><span>RECORDED TAKES</span><strong>{totalFrames.toLocaleString()} raw frames</strong></div>
          {#if project.phases.length === 0}
            <p class="empty-copy">No RGB-D takes yet. Tracking runs at sensor rate; the archive rate only controls frames kept for the production pass.</p>
          {:else}
            {#each project.phases as capture, index}
              <article>
                <span class="take-number">{String(index + 1).padStart(2, '0')}</span>
                <div><strong>{capture.name}</strong><small>{capture.frameCount.toLocaleString()} raw frames · {formatDuration(capture.durationSeconds)}</small></div>
                <button on:click={() => removeCaptureAction(capture.id, capture.name)} disabled={busy || capturing || processing}>Delete</button>
              </article>
            {/each}
          {/if}
        </section>

      {:else if workspace === 'reconstruct'}
        <section class="panel panel-heading"><div><span>PRODUCTION PASS</span><h2>Reconstruction outputs</h2></div><strong class="take-total">{mediaOnlyProject ? `${mediaSourceCount} media source${mediaSourceCount === 1 ? '' : 's'}` : `${completedCaptures} take${completedCaptures === 1 ? '' : 's'}`}</strong></section>

        <section class="panel target-list">
          <label class:active={buildPointCloud}><input type="checkbox" bind:checked={buildPointCloud} disabled={processing || mediaOnlyProject}/><span class="target-icon">P</span><div><strong>Metric point cloud</strong><small>{mediaOnlyProject ? 'Requires calibrated RGB-D capture' : 'Filtered colored PLY · quickest'}</small></div><i>{artifactReady('pointCloud') ? 'READY' : ''}</i></label>
          <label class:active={buildTexturedMesh}><input type="checkbox" bind:checked={buildTexturedMesh} disabled={processing || mediaOnlyProject}/><span class="target-icon">M</span><div><strong>Textured triangle mesh</strong><small>{mediaOnlyProject ? 'Requires calibrated RGB-D capture' : 'TSDF surface · OBJ/MTL/PNG'}</small></div><i>{artifactReady('texturedMesh') ? 'READY' : ''}</i></label>
          <label class:active={buildGaussianSplat}><input type="checkbox" bind:checked={buildGaussianSplat} disabled={processing || !runtime?.splatWorkerAvailable}/><span class="target-icon">G</span><div><strong>{mediaOnlyProject ? 'Photoreal 3D Gaussian splat' : '2D Gaussian surface'}</strong><small>{mediaOnlyProject ? 'COLMAP cameras · anisotropic 3DGS · SH degree 3' : 'Depth-aware discs · metric PLY'}</small></div><i>{artifactReady('gaussianSplat') ? 'READY' : runtime?.splatWorkerAvailable ? '' : 'CUDA RUNTIME MISSING'}</i></label>
          {#if buildGaussianSplat}
            <label class="iterations"><span>Training iterations</span><input type="range" min="5000" max="60000" step="5000" bind:value={splatIterations} disabled={processing}/><strong>{Number(splatIterations).toLocaleString()}</strong></label>
          {/if}
        </section>

        {#if mediaOnlyProject}
          <section class="panel pipeline-note">
            <strong>Photo/video source</strong>
            <p>ScanLan selects sharp, non-duplicate video frames, solves and bundle-adjusts cameras with COLMAP, undistorts every registered view, then trains exposure-compensated 3D Gaussians. Weak or disconnected camera solutions fail visibly.</p>
            <div><span>Imported</span><strong>{mediaSourceCount} source{mediaSourceCount === 1 ? '' : 's'}</strong></div>
            <button class="ghost full" on:click={addMediaSource} disabled={busy || processing}>Add more photos or video…</button>
          </section>
        {/if}

        {#if !mediaOnlyProject}
          <section class="panel pipeline-note">
            <strong>High-resolution texture photos</strong>
            <p>After an initial mesh build, add overlapping DSLR or phone photos. ScanLan detects depth-backed feature matches, validates each camera pose, and uses accepted photos during the next mesh rebuild.</p>
            {#if texturePhotoProgress}
              <div class="texture-progress" class:error={texturePhotoProgress.status === 'failed'}>
                <div class="texture-progress-title">
                  <span>{texturePhotoProgress.status === 'running' ? texturePhotoProgress.stage.replaceAll('_', ' ') : 'Last localization'}</span>
                  <strong>{Math.round(texturePhotoProgress.progress * 100)}%</strong>
                </div>
                <div class="progress"><i style={`width:${Math.round(texturePhotoProgress.progress * 100)}%`}></i></div>
                <small>{texturePhotoProgress.detail}</small>
              </div>
            {/if}
            {#if texturePhotos.length}
              <div class="texture-summary">
                <span><strong>{localizedTexturePhotoCount}</strong> ready to bake</span>
                <span><strong>{rejectedTexturePhotoCount}</strong> rejected</span>
                {#if pendingTexturePhotoCount}<span><strong>{pendingTexturePhotoCount}</strong> pending</span>{/if}
              </div>
              <div class="texture-photo-list">
                {#each texturePhotos as photo (photo.id)}
                  <article class="texture-photo" class:rejected={photo.status === 'rejected'} class:pending={photo.status === 'queued' || photo.status === 'localizing'}>
                    <span class="photo-state">{photo.status === 'localized' ? 'READY' : photo.status === 'rejected' ? 'REJECTED' : photo.status === 'localizing' ? 'MATCHING' : 'QUEUED'}</span>
                    <div class="photo-copy">
                      <strong title={photo.sourcePath ?? photo.path}>{photo.name}</strong>
                      {#if photo.status === 'localized'}
                        <small>{photo.inlierCount ?? '—'} inliers · {photo.reprojectionRmsePixels?.toFixed(2) ?? '—'} px RMSE</small>
                      {:else if photo.status === 'rejected'}
                        <small title={photo.error}>{photo.error ?? 'Camera pose did not pass validation'}</small>
                      {:else}
                        <small>{photo.status === 'localizing' ? 'Feature matching and geometric validation' : 'Waiting for localization'}</small>
                      {/if}
                    </div>
                    <div class="photo-quality">
                      <strong>{photo.status === 'localized' ? (photo.qualityScore === undefined ? 'Ready' : `${photo.qualityScore}/100`) : photo.status === 'rejected' ? 'Not used' : 'Waiting'}</strong>
                      <small>{photo.qualityLabel ?? (photo.status === 'localized' ? 'Localized' : photo.status === 'rejected' ? 'Rejected' : 'Queued')}</small>
                    </div>
                    <button on:click={() => removeTexturePhoto(photo)} disabled={busy || processing || photoLocalizationActive} aria-label={`Remove ${photo.name}`}>Remove</button>
                  </article>
                {/each}
              </div>
            {/if}
            <button class="ghost full" on:click={addTexturePhotos} disabled={busy || processing || photoLocalizationActive || !project?.artifacts.texturedMesh}>{photoLocalizationActive ? 'Localization running…' : 'Add and localize photos…'}</button>
          </section>

          <section class="panel pipeline-note">
            <strong>One trajectory, three representations</strong>
            <p>All outputs share the same quality-gated RGB-D poses. The final pass stabilizes the trajectory, fuses a weighted TSDF, and only then builds the selected representations.</p>
            <div><span>Source</span><strong>{totalFrames.toLocaleString()} archived frames</strong></div>
            <div><span>Compute</span><strong>{runtime?.reconstructionWorkerAvailable ? 'CUDA preferred' : 'Runtime missing'}</strong></div>
          </section>
        {/if}

        {#if activeJob}
          <section class="panel job-card" class:error={activeJob.status === 'failed'}>
            <div class="section-title"><span>{activeJob.status.toUpperCase()}</span><strong>{Math.round(activeJob.progress * 100)}%</strong></div>
            <h3>{activeJob.stage.replaceAll('_', ' ')}</h3>
            <p>{activeJob.error ?? activeJob.detail}</p>
            <div class="progress"><i style={`width:${Math.round(activeJob.progress * 100)}%`}></i></div>
            {#if activeJob.iteration !== null}
              <div class="job-quality">
                <span>Iteration <strong>{activeJob.iteration.toLocaleString()} / {activeJob.totalIterations?.toLocaleString()}</strong></span>
                <span>Current <strong>{activeJob.loss?.toFixed(4) ?? '—'}</strong></span>
                <span>Rolling <strong>{activeJob.smoothedLoss?.toFixed(4) ?? 'warming up'}</strong></span>
              </div>
            {/if}
            <div class="job-meta"><span>{activeJob.computeBackend ?? 'Waiting for worker'}</span><span>{activeJob.etaSeconds ? `~${formatDuration(activeJob.etaSeconds)}` : ''}</span></div>
            {#if processing}
              <button class="ghost full" on:click={cancelBuild}>Cancel safely</button>
            {:else if activeJob.resumable && ['failed', 'cancelled'].includes(activeJob.status)}
              <div class="button-row"><button class="primary" on:click={() => startBuild(true)}>Resume checkpoint</button><button class="ghost" on:click={discardBuild}>Discard</button></div>
            {/if}
          </section>
        {/if}

        <button class="primary full build-button" on:click={() => startBuild(false)} disabled={busy || processing || photoLocalizationActive || (completedCaptures === 0 && mediaSourceCount === 0) || (!buildPointCloud && !buildTexturedMesh && !buildGaussianSplat)}>{processing ? 'Reconstruction running…' : photoLocalizationActive ? 'Localizing texture photos…' : readyArtifacts ? 'Rebuild selected outputs' : mediaOnlyProject ? 'Solve cameras & build AAA splat' : 'Build selected outputs'}</button>

      {:else}
        <section class="panel panel-heading"><div><span>RESULT</span><h2>Edit & export</h2></div><strong class="take-total">{readyArtifacts} ready</strong></section>
        <section class="panel view-switcher">
          <button class:active={renderMode === 'points'} disabled={!artifactReady('pointCloud')} on:click={() => loadResult('points')}><span>P</span><div><strong>Points</strong><small>{formatCount(project.pointCount)}</small></div></button>
          <button class:active={renderMode === 'mesh'} disabled={!artifactReady('texturedMesh')} on:click={() => loadResult('mesh')}><span>M</span><div><strong>Mesh</strong><small>{formatCount(project.meshTriangleCount)} tris</small></div></button>
          <button class:active={renderMode === 'splat'} disabled={!artifactReady('gaussianSplat')} on:click={() => loadResult('splat')}><span>G</span><div><strong>2DGS</strong><small>Metric surface</small></div></button>
        </section>
        {#if renderMode === 'mesh'}
          <section class="panel settings"><label>Mesh display<select bind:value={meshViewMode}><option value="surface">Textured</option><option value="surface-wireframe">Texture + wire</option><option value="wireframe">Wireframe</option><option value="shaded">Shaded</option></select></label></section>
        {/if}
        {#if renderMode !== 'splat'}
          <section class="panel edit-tools">
            <div class="section-title"><span>MODEL POSE</span><strong class:edited={hasEditPose}>{hasEditPose ? 'EDITED' : 'ORIGINAL'}</strong></div>
            <div class="edit-actions">
              <button class:active={floorPickMode} disabled={!canEditModel} on:click={() => { floorPickMode = !floorPickMode; editMode = false; }}>{floorPickMode ? 'Cancel floor pick' : 'Pick floor'}</button>
              <button class:active={editMode} disabled={!canEditModel} on:click={() => { editMode = !editMode; floorPickMode = false; }}>{editMode ? 'Close gizmo' : 'Transform gizmo'}</button>
            </div>
            {#if editMode && canEditModel}
              <div class="gizmo-modes">
                <button class:active={gizmoMode === 'translate'} on:click={() => setGizmoMode('translate')}>Move <kbd>W</kbd></button>
                <button class:active={gizmoMode === 'rotate'} on:click={() => setGizmoMode('rotate')}>Rotate <kbd>E</kbd></button>
                <button class:active={gizmoMode === 'scale'} on:click={() => setGizmoMode('scale')}>Scale <kbd>R</kbd></button>
              </div>
            {/if}
            <div class="edit-options">
              <label><span>Rotation snap</span>
                <select value={rotationSnapDegrees} on:change={(event) => setRotationSnap(Number(inputValue(event)))}>
                  <option value={0}>Off · smooth</option>
                  <option value={1}>1°</option>
                  <option value={5}>5°</option>
                  <option value={15}>15°</option>
                </select>
              </label>
              <label><span>Pose saving</span>
                <select value={transformSaveMode} on:change={(event) => setTransformSaveMode(inputValue(event) as TransformSaveMode)}>
                  <option value="manual">Manual</option>
                  <option value="auto">Automatic</option>
                </select>
              </label>
            </div>
            {#if transformSaveMode === 'manual'}
              <button class="save-pose" class:dirty={transformDirty} disabled={!transformDirty} on:click={saveTransform}>{transformDirty ? 'Save pose' : 'Pose saved'}</button>
            {/if}
            <button class="reset-pose" disabled={!hasEditPose} on:click={resetEditPose}>Reset pose</button>
            <p>{transformSaveMode === 'manual' ? 'Manipulations stay in memory until Save pose, so you can edit without persistence work between tools.' : 'Each completed manipulation is saved automatically.'} The current pose is applied to point-cloud and mesh exports.</p>
          </section>
        {/if}
        <section class="panel result-stats">
          <div><span>Points</span><strong>{formatCount(project.pointCount)}</strong></div>
          <div><span>Triangles</span><strong>{formatCount(project.meshTriangleCount)}</strong></div>
          <div><span>Frames used</span><strong>{project.framesUsed ?? '—'}</strong></div>
          <div><span>Confidence</span><strong>{project.confidenceLabel ?? '—'}</strong></div>
          <p>{project.confidenceDetail ?? 'Build an output to see trajectory and coverage quality.'}</p>
        </section>
        <section class="panel export-list">
          <button on:click={exportPointCloud} disabled={!artifactReady('pointCloud')}><span>P</span><div><strong>Point cloud PLY</strong><small>Metric colored vertices</small></div><i>Export…</i></button>
          <button on:click={exportMesh} disabled={!artifactReady('texturedMesh')}><span>M</span><div><strong>Textured OBJ bundle</strong><small>OBJ + MTL + PNG</small></div><i>Export…</i></button>
          <button on:click={exportSplat} disabled={!artifactReady('gaussianSplat')}><span>G</span><div><strong>2D Gaussian PLY</strong><small>Canonical metric splat + sidecars</small></div><i>Export…</i></button>
        </section>
      {/if}
    </aside>
  </main>

  <footer class:error={Boolean(fatalError)}>
    <span class="status-dot" class:busy={busy || capturing || processing || photoLocalizationActive}></span>
    <strong>{capturing ? 'LIVE' : processing ? 'BUILDING' : photoLocalizationActive ? 'LOCALIZING' : fatalError ? 'ERROR' : 'READY'}</strong>
    <p>{message}</p>
    {#if sensor?.imuActive}<span class="footer-metric">IMU {sensor.imuRateHz.toFixed(0)} Hz</span>{/if}
    {#if sensor?.liveReconstructionBackend}<span class="footer-metric">{sensor.liveReconstructionBackend}</span>{/if}
  </footer>
</div>

<style>
  :global(*) { box-sizing: border-box; }
  :global(html, body, #app) { width: 100%; height: 100%; margin: 0; overflow: hidden; }
  :global(body) { background: #071019; color: #dce9ee; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  :global(button), :global(input), :global(select) { font: inherit; }
  :global(button) { color: inherit; }

  .app-shell { --panel: #0c1823; --panel-soft: #101f2b; --line: rgba(155, 199, 215, 0.13); --muted: #78909d; --cyan: #63c7e7; --mint: #62d6ba; --amber: #efb366; display: grid; grid-template-rows: 68px 68px minmax(0, 1fr) 38px; width: 100%; height: 100%; background: radial-gradient(circle at 45% -20%, rgba(40, 112, 139, .15), transparent 42%), #071019; }
  .topbar { display: grid; grid-template-columns: minmax(230px, .8fr) minmax(220px, 1fr) auto auto; align-items: center; gap: 24px; padding: 0 24px; border-bottom: 1px solid var(--line); background: rgba(7, 16, 25, .9); }
  .brand, .project-title, .runtime-state, .panel-heading, .tracking-title, .section-title, .job-meta, .button-row { display: flex; align-items: center; }
  .brand { gap: 11px; }
  .brand-mark { display: grid; place-items: center; width: 35px; height: 35px; border: 1px solid rgba(99,199,231,.45); border-radius: 10px; background: linear-gradient(145deg, rgba(99,199,231,.18), rgba(98,214,186,.06)); color: #80d7ef; font-size: 12px; font-weight: 850; letter-spacing: .06em; }
  .brand div, .project-title { display: grid; gap: 2px; }
  .brand strong { font-size: 16px; letter-spacing: .01em; }
  .brand small, .project-title span { color: var(--muted); font-size: 10px; letter-spacing: .08em; text-transform: uppercase; }
  .project-title { justify-items: start; min-width: 0; padding: 8px 10px; border-radius: 8px; background: transparent; text-align: left; }
  .project-title:hover:not(:disabled) { background: rgba(99,199,231,.06); }
  .project-title strong { max-width: 360px; overflow: hidden; font-size: 13px; text-overflow: ellipsis; white-space: nowrap; }
  .header-actions { display: flex; gap: 7px; }
  .runtime-state { gap: 8px; }
  .runtime-state span { display: flex; align-items: center; gap: 6px; padding: 7px 9px; border: 1px solid var(--line); border-radius: 8px; color: #708590; font-size: 10px; font-weight: 750; letter-spacing: .04em; text-transform: uppercase; }
  .runtime-state span i { width: 6px; height: 6px; border-radius: 50%; background: #53636b; }
  .runtime-state span.ready { color: #a8c4cf; }
  .runtime-state span.ready i { background: var(--mint); box-shadow: 0 0 10px rgba(98,214,186,.5); }
  button { border: 0; cursor: pointer; }
  button:disabled, input:disabled, select:disabled { cursor: not-allowed; opacity: .43; }
  .ghost { padding: 10px 13px; border: 1px solid var(--line); border-radius: 9px; background: rgba(255,255,255,.02); color: #a9bfca; font-size: 12px; font-weight: 700; }
  .ghost:hover:not(:disabled) { border-color: rgba(99,199,231,.38); background: rgba(99,199,231,.07); }
  .ghost.compact { white-space: nowrap; }
  .ghost.full, .primary.full { width: 100%; }
  .primary { padding: 11px 15px; border-radius: 9px; background: linear-gradient(135deg, #3ba8cc, #42bda2); color: #041018; font-size: 12px; font-weight: 850; }

  .modal-backdrop { position: fixed; z-index: 100; inset: 0; display: grid; place-items: center; padding: 24px; background: rgba(2, 8, 13, .72); backdrop-filter: blur(8px); }
  .modal-dismiss { position: absolute; inset: 0; width: 100%; height: 100%; background: transparent; cursor: default; }
  .project-manager { position: relative; z-index: 1; display: grid; grid-template-rows: auto auto auto minmax(150px, 1fr) auto auto; gap: 14px; width: min(780px, calc(100vw - 48px)); max-height: min(760px, calc(100vh - 48px)); padding: 20px; overflow: hidden; border: 1px solid rgba(155,199,215,.2); border-radius: 16px; background: linear-gradient(150deg, #10202c, #091722); box-shadow: 0 30px 90px rgba(0,0,0,.5); }
  .project-manager > header { display: flex; align-items: center; justify-content: space-between; }
  .project-manager > header div { display: grid; gap: 4px; }
  .project-manager > header span, .current-project-editor span, .new-project-form span { color: var(--cyan); font-size: 9px; font-weight: 850; letter-spacing: .11em; }
  .dialog-close { display: grid; place-items: center; width: 34px; height: 34px; border: 1px solid var(--line); border-radius: 9px; background: rgba(255,255,255,.025); color: #91a9b4; font-size: 19px; }
  .current-project-editor { display: grid; grid-template-columns: minmax(180px, .8fr) minmax(260px, 1.2fr); align-items: end; gap: 16px; padding: 13px; border: 1px solid rgba(99,199,231,.2); border-radius: 11px; background: rgba(99,199,231,.055); }
  .current-project-editor > div, .new-project-form > div { display: grid; gap: 4px; }
  .current-project-editor small, .new-project-form small { color: #758d98; font-size: 9px; line-height: 1.4; }
  .current-project-editor form { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; }
  .current-project-editor .primary { height: 36px; padding-top: 0; padding-bottom: 0; }
  .project-library-heading { display: flex; justify-content: space-between; color: #8da4ae; font-size: 10px; }
  .project-library-heading span { color: #617985; }
  .project-library { min-height: 150px; overflow-y: auto; border: 1px solid var(--line); border-radius: 11px; background: #08151f; scrollbar-color: #263d49 transparent; }
  .project-library.loading { opacity: .7; }
  .project-library article { display: grid; grid-template-columns: 40px minmax(0, 1fr) auto; align-items: center; gap: 12px; padding: 13px; border-bottom: 1px solid var(--line); }
  .project-library article:last-child { border-bottom: 0; }
  .project-library article.active { background: rgba(98,214,186,.055); }
  .project-library-icon { display: grid; place-items: center; width: 40px; height: 40px; border: 1px solid rgba(99,199,231,.18); border-radius: 10px; background: rgba(99,199,231,.07); color: #7ccce5; font-size: 10px; font-weight: 850; }
  .project-library-copy { min-width: 0; }
  .project-library-copy > div { display: flex; align-items: center; gap: 8px; }
  .project-library-copy strong { overflow: hidden; font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
  .project-library-copy span { padding: 2px 5px; border-radius: 4px; background: rgba(98,214,186,.12); color: var(--mint); font-size: 7px; font-weight: 850; letter-spacing: .08em; }
  .project-library-copy small, .project-library-copy p { color: #657d88; font-size: 8px; }
  .project-library-copy p { margin-top: 4px; }
  .project-library-actions { display: flex; gap: 6px; }
  .project-library-actions button { padding: 7px 10px; font-size: 9px; }
  .danger { border: 1px solid rgba(226,120,103,.2); border-radius: 8px; background: rgba(226,120,103,.06); color: #c88d84; font-size: 9px; font-weight: 750; }
  .danger:hover:not(:disabled) { border-color: rgba(226,120,103,.4); background: rgba(226,120,103,.11); }
  .project-library-empty { display: grid; place-items: center; align-content: center; min-height: 150px; color: #718994; font-size: 10px; }
  .project-library-empty .spinner { width: 18px; height: 18px; margin-bottom: 8px; }
  .new-project-form { display: grid; grid-template-columns: minmax(150px, .7fr) minmax(180px, 1fr) auto auto; align-items: end; gap: 8px; padding-top: 2px; }
  .new-project-form .primary, .new-project-form .ghost { height: 36px; padding-top: 0; padding-bottom: 0; }
  .new-project-button { width: 100%; min-height: 40px; border: 1px dashed rgba(99,199,231,.25); border-radius: 9px; background: rgba(99,199,231,.035); color: var(--cyan); font-size: 10px; font-weight: 800; }
  .new-project-button:hover:not(:disabled) { border-color: rgba(99,199,231,.45); background: rgba(99,199,231,.075); }
  .project-manager-error { color: #df9388; font-size: 10px; }

  .workflow { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1px; padding: 0 24px; border-bottom: 1px solid var(--line); background: #09131d; }
  .workflow button { position: relative; display: flex; align-items: center; gap: 12px; padding: 0 18px; background: transparent; color: #708792; text-align: left; }
  .workflow button::after { position: absolute; right: 0; bottom: -1px; left: 0; height: 2px; background: transparent; content: ''; }
  .workflow button.active { color: #d6e7ed; background: rgba(99,199,231,.045); }
  .workflow button.active::after { background: var(--cyan); box-shadow: 0 -3px 12px rgba(99,199,231,.28); }
  .workflow button > span { color: #49606b; font-family: ui-monospace, monospace; font-size: 11px; font-weight: 800; }
  .workflow button.done > span { color: var(--mint); }
  .workflow button div { display: grid; gap: 3px; }
  .workflow button strong { font-size: 12px; }
  .workflow button small { color: #607783; font-size: 10px; }

  main { display: grid; grid-template-columns: minmax(0, 1fr) 390px; min-height: 0; }
  .viewport { position: relative; min-width: 0; min-height: 0; padding: 14px; border-right: 1px solid var(--line); }
  .viewport :global(.viewer) { border-radius: 14px; }
  aside { min-height: 0; padding: 14px; overflow-x: hidden; overflow-y: auto; background: #09131d; scrollbar-color: #263d49 transparent; }
  .panel { margin-bottom: 12px; padding: 15px; border: 1px solid var(--line); border-radius: 12px; background: linear-gradient(150deg, rgba(17,34,46,.94), rgba(11,24,35,.94)); box-shadow: 0 12px 35px rgba(0,0,0,.08); }
  .panel-heading { justify-content: space-between; gap: 12px; padding: 10px 3px 13px; border: 0; border-radius: 0; background: transparent; box-shadow: none; }
  .panel-heading > div { display: grid; gap: 4px; }
  .panel-heading span, .section-title span { color: var(--cyan); font-size: 9px; font-weight: 850; letter-spacing: .11em; }
  h2, h3, p { margin: 0; }
  h2 { font-size: 18px; letter-spacing: -.02em; }
  h3 { margin: 8px 0 4px; font-size: 14px; text-transform: capitalize; }
  .icon-button { display: grid; place-items: center; width: 34px; height: 34px; border: 1px solid var(--line); border-radius: 9px; background: rgba(255,255,255,.025); color: var(--cyan); font-size: 18px; }

  .settings { display: grid; gap: 13px; }
  .settings label { display: grid; gap: 6px; color: #8ba2ad; font-size: 10px; font-weight: 720; letter-spacing: .03em; }
  .setting-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 10px; }
  select, input { width: 100%; min-width: 0; height: 36px; padding: 0 10px; outline: none; border: 1px solid rgba(147,193,211,.16); border-radius: 8px; background: #091722; color: #c8dce4; font-size: 11px; }
  select:focus, input:focus { border-color: rgba(99,199,231,.55); box-shadow: 0 0 0 2px rgba(99,199,231,.08); }
  .unit-input { position: relative; }
  .unit-input input { padding-right: 35px; }
  .unit-input span { position: absolute; top: 50%; right: 10px; color: #637b86; transform: translateY(-50%); }
  .toggle { grid-template-columns: auto auto 1fr; align-items: center; cursor: pointer; }
  .toggle input { position: absolute; width: 1px; height: 1px; opacity: 0; }
  .toggle > span { position: relative; width: 34px; height: 19px; border-radius: 20px; background: #263945; transition: .2s; }
  .toggle > span::after { position: absolute; top: 3px; left: 3px; width: 13px; height: 13px; border-radius: 50%; background: #8499a3; transition: .2s; content: ''; }
  .toggle input:checked + span { background: rgba(98,214,186,.28); }
  .toggle input:checked + span::after { left: 18px; background: var(--mint); }
  .toggle div { display: grid; gap: 2px; }
  .toggle strong { color: #bfd1d8; font-size: 11px; }
  .toggle small { color: #657c87; font-size: 9px; font-weight: 500; }

  .capture-button { display: flex; align-items: center; justify-content: center; gap: 10px; width: 100%; height: 50px; margin-bottom: 12px; border-radius: 12px; background: linear-gradient(135deg, #49c7a9, #4bb5d3); color: #041119; font-size: 13px; font-weight: 900; box-shadow: 0 10px 30px rgba(56,176,169,.16); }
  .capture-button i { width: 11px; height: 11px; border: 2px solid currentColor; border-radius: 50%; }
  .capture-button.stop { background: linear-gradient(135deg, #e27867, #e9a159); color: #1c0b07; }
  .capture-button.stop i { border-radius: 2px; background: currentColor; }
  .tracking-card.warning { border-color: rgba(239,179,102,.3); }
  .connection-card.connected { border-color: rgba(98,214,186,.28); }
  .connection-card.warning { border-color: rgba(239,179,102,.24); }
  .tracking-title { gap: 10px; }
  .tracking-title > i { width: 9px; height: 9px; border-radius: 50%; background: var(--mint); box-shadow: 0 0 12px rgba(98,214,186,.5); }
  .tracking-card.warning .tracking-title > i { background: var(--amber); }
  .connection-card.warning .tracking-title > i { background: var(--amber); box-shadow: 0 0 12px rgba(239,179,102,.35); }
  .tracking-title div { display: grid; gap: 3px; }
  .tracking-title strong { font-size: 11px; }
  .tracking-title small { color: var(--muted); font-size: 9px; }
  .tracking-card > p, .connection-card > p, .empty-copy, .pipeline-note p, .job-card p, .result-stats p { margin-top: 11px; color: #708792; font-size: 10px; line-height: 1.55; }
  .mini-grid, .result-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; margin-top: 12px; overflow: hidden; border: 1px solid var(--line); border-radius: 8px; background: var(--line); }
  .mini-grid div, .result-stats > div { display: grid; gap: 3px; padding: 9px; background: #0b1924; }
  .mini-grid span, .result-stats span, .pipeline-note div span { color: #617985; font-size: 9px; }
  .mini-grid strong, .result-stats strong, .pipeline-note div strong { font-size: 11px; }

  .section-title { justify-content: space-between; }
  .section-title > strong, .take-total { color: #8199a5; font-size: 10px; }
  .takes article { display: grid; grid-template-columns: 28px 1fr auto; align-items: center; gap: 9px; padding: 11px 0; border-bottom: 1px solid var(--line); }
  .takes article:last-child { padding-bottom: 0; border-bottom: 0; }
  .take-number { color: #4d6570; font-family: ui-monospace, monospace; font-size: 10px; }
  .takes article div { display: grid; gap: 3px; }
  .takes article strong { font-size: 11px; }
  .takes article small { color: #6d8590; font-size: 9px; }
  .takes article button { padding: 5px 7px; background: transparent; color: #997b7a; font-size: 9px; }

  .target-list { display: grid; gap: 8px; }
  .target-list > label { display: grid; grid-template-columns: auto 34px 1fr auto; align-items: center; gap: 9px; min-height: 58px; padding: 9px; border: 1px solid var(--line); border-radius: 9px; background: #0a1823; cursor: pointer; }
  .target-list > label.active { border-color: rgba(99,199,231,.33); background: rgba(49,124,151,.09); }
  .target-list > label > input[type=checkbox] { width: 14px; height: 14px; accent-color: var(--cyan); }
  .target-icon, .view-switcher button > span, .export-list button > span { display: grid; place-items: center; width: 32px; height: 32px; border-radius: 8px; background: rgba(99,199,231,.08); color: var(--cyan); font-size: 10px; font-weight: 900; }
  .target-list label div { display: grid; gap: 3px; }
  .target-list label div strong { font-size: 11px; }
  .target-list label div small { color: #687f8a; font-size: 9px; }
  .target-list label > i { max-width: 80px; color: var(--mint); font-size: 8px; font-style: normal; font-weight: 800; text-align: right; }
  .target-list .iterations { grid-template-columns: auto 1fr auto; min-height: auto; }
  .iterations input { height: 18px; padding: 0; accent-color: var(--cyan); }
  .pipeline-note > strong { font-size: 12px; }
  .pipeline-note div { display: flex; justify-content: space-between; padding-top: 9px; }
  .pipeline-note .texture-progress { display: grid; gap: 8px; margin-top: 12px; padding: 10px; border: 1px solid rgba(99,199,231,.2); border-radius: 9px; background: rgba(99,199,231,.045); }
  .pipeline-note .texture-progress.error { border-color: rgba(226,120,103,.3); background: rgba(226,120,103,.045); }
  .pipeline-note .texture-progress-title { display: flex; align-items: center; padding: 0; text-transform: capitalize; }
  .pipeline-note .texture-progress-title span { color: #8aa3ae; }
  .pipeline-note .texture-progress-title strong { color: var(--cyan); font-size: 10px; }
  .pipeline-note .texture-progress.error .texture-progress-title strong { color: #e28a7d; }
  .pipeline-note .texture-progress .progress { display: block; width: 100%; padding: 0; }
  .pipeline-note .texture-progress > small { overflow: hidden; color: #718894; font-size: 9px; line-height: 1.4; text-overflow: ellipsis; }
  .pipeline-note .texture-summary { display: flex; gap: 14px; justify-content: flex-start; margin-top: 10px; padding: 0; color: #6d8590; font-size: 9px; }
  .pipeline-note .texture-summary span { color: #718894; }
  .pipeline-note .texture-summary strong { margin-right: 3px; color: #b7ccd5; }
  .pipeline-note .texture-photo-list { display: grid; max-height: 310px; margin-top: 9px; padding: 0; overflow-y: auto; border-top: 1px solid var(--line); scrollbar-color: #263d49 transparent; }
  .pipeline-note .texture-photo { display: grid; grid-template-columns: 48px minmax(0, 1fr) 58px auto; align-items: center; gap: 8px; min-width: 0; padding: 9px 0; border-bottom: 1px solid var(--line); }
  .pipeline-note .texture-photo .photo-state { padding: 4px 3px; border-radius: 5px; background: rgba(98,214,186,.1); color: var(--mint); font-size: 7px; font-weight: 850; text-align: center; }
  .pipeline-note .texture-photo.rejected .photo-state { background: rgba(226,120,103,.1); color: #d98a80; }
  .pipeline-note .texture-photo.pending .photo-state { background: rgba(239,179,102,.1); color: var(--amber); }
  .pipeline-note .texture-photo .photo-copy, .pipeline-note .texture-photo .photo-quality { display: grid; min-width: 0; gap: 3px; padding: 0; }
  .pipeline-note .texture-photo .photo-copy strong, .pipeline-note .texture-photo .photo-copy small { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .pipeline-note .texture-photo .photo-copy strong { font-size: 10px; }
  .pipeline-note .texture-photo .photo-copy small { color: #657d88; font-size: 8px; }
  .pipeline-note .texture-photo .photo-quality { text-align: right; }
  .pipeline-note .texture-photo .photo-quality strong { color: #b7ccd5; font-size: 9px; }
  .pipeline-note .texture-photo .photo-quality small { color: #657d88; font-size: 7px; }
  .pipeline-note .texture-photo > button { padding: 4px 2px; background: transparent; color: #987b7b; font-size: 8px; }
  .pipeline-note > .ghost { margin-top: 11px; }
  .job-card.error { border-color: rgba(226,120,103,.35); }
  .progress { height: 5px; overflow: hidden; border-radius: 6px; background: #172a35; }
  .progress i { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--cyan), var(--mint)); transition: width .25s linear; }
  .job-card .progress { margin: 11px 0 8px; }
  .job-quality { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 6px 12px; margin: 9px 0; color: #687f8a; font-size: 8px; }
  .job-quality strong { margin-left: 3px; color: #a9bec8; font-size: 9px; }
  .job-meta { justify-content: space-between; gap: 8px; margin-bottom: 11px; color: #687f8a; font-size: 9px; }
  .button-row { gap: 8px; }
  .button-row button { flex: 1; }
  .build-button { height: 47px; }

  .view-switcher { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; }
  .view-switcher button { display: grid; justify-items: center; gap: 7px; padding: 10px 4px; border: 1px solid var(--line); border-radius: 9px; background: #0a1823; }
  .view-switcher button.active { border-color: rgba(99,199,231,.4); background: rgba(99,199,231,.09); }
  .view-switcher button div { display: grid; gap: 2px; text-align: center; }
  .view-switcher button strong { font-size: 10px; }
  .view-switcher button small { color: #687f8a; font-size: 8px; }
  .edit-tools { display: grid; gap: 9px; }
  .edit-tools .section-title strong.edited { color: var(--amber); }
  .edit-actions, .gizmo-modes { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
  .gizmo-modes { grid-template-columns: repeat(3, 1fr); }
  .edit-options { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; padding-top: 2px; }
  .edit-options label { display: grid; gap: 5px; color: #78909d; font-size: 9px; font-weight: 700; }
  .edit-options select { height: 32px; padding: 0 8px; font-size: 9px; }
  .edit-tools button { min-height: 35px; padding: 7px 8px; border: 1px solid var(--line); border-radius: 8px; background: #0a1823; color: #9eb5bf; font-size: 9px; font-weight: 750; }
  .edit-tools button:hover:not(:disabled), .edit-tools button.active { border-color: rgba(99,199,231,.42); background: rgba(99,199,231,.09); color: #d2e5ec; }
  .edit-tools button.active { box-shadow: inset 0 0 0 1px rgba(99,199,231,.08); }
  .edit-tools .save-pose.dirty { border-color: rgba(98,214,186,.4); background: rgba(98,214,186,.1); color: var(--mint); }
  .edit-tools kbd { margin-left: 3px; padding: 1px 4px; border: 1px solid rgba(155,199,215,.16); border-radius: 4px; color: var(--cyan); font-family: ui-monospace, monospace; font-size: 8px; }
  .edit-tools .reset-pose { min-height: 30px; color: #7f97a2; }
  .edit-tools > p { color: #687f8a; font-size: 9px; line-height: 1.45; }
  .result-stats { padding: 0; }
  .result-stats p { grid-column: 1 / -1; margin: 0; padding: 10px; background: #0b1924; }
  .export-list { display: grid; gap: 7px; }
  .export-list button { display: grid; grid-template-columns: 34px 1fr auto; align-items: center; gap: 9px; padding: 10px; border: 1px solid var(--line); border-radius: 9px; background: #0a1823; text-align: left; }
  .export-list button:hover:not(:disabled) { border-color: rgba(99,199,231,.36); }
  .export-list button div { display: grid; gap: 3px; }
  .export-list button strong { font-size: 10px; }
  .export-list button small { color: #687f8a; font-size: 8px; }
  .export-list button > i { color: var(--cyan); font-size: 9px; font-style: normal; }

  .live-metrics { position: absolute; top: 26px; right: 26px; display: grid; grid-template-columns: repeat(3, minmax(80px, 1fr)); gap: 1px; overflow: hidden; border: 1px solid rgba(141,195,214,.16); border-radius: 9px; background: rgba(5,14,22,.72); box-shadow: 0 12px 35px rgba(0,0,0,.22); backdrop-filter: blur(12px); }
  .live-metrics div { display: grid; gap: 3px; padding: 8px 10px; background: rgba(10,26,37,.78); }
  .live-metrics span { color: #6f8792; font-size: 8px; text-transform: uppercase; }
  .live-metrics strong { font-size: 10px; }
  .live-metrics strong.good { color: var(--mint); }
  .job-overlay { position: absolute; right: 26px; bottom: 26px; width: min(420px, calc(100% - 52px)); padding: 13px; border: 1px solid rgba(141,195,214,.17); border-radius: 10px; background: rgba(5,14,22,.82); backdrop-filter: blur(12px); }
  .job-overlay > div:first-child { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 10px; text-transform: capitalize; }
  .job-overlay .job-quality { justify-content: flex-start; }
  .job-overlay p { margin-top: 8px; color: #78909c; font-size: 9px; }

  footer { display: flex; align-items: center; gap: 8px; padding: 0 24px; border-top: 1px solid var(--line); background: #08121b; color: #758c97; font-size: 9px; }
  footer strong { color: #9bb1bb; font-size: 9px; letter-spacing: .08em; }
  footer p { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  footer.error p { color: #d8988e; }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--mint); }
  .status-dot.busy { background: var(--amber); animation: pulse 1s infinite; }
  .footer-metric { padding-left: 12px; border-left: 1px solid var(--line); color: #607984; }
  .spinner { width: 24px; height: 24px; margin-bottom: 12px; border: 2px solid rgba(99,199,231,.16); border-top-color: var(--cyan); border-radius: 50%; animation: spin .8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes pulse { 50% { opacity: .35; } }

  @media (max-width: 1120px) {
    main { grid-template-columns: minmax(0, 1fr) 340px; }
    .topbar { grid-template-columns: auto 1fr auto; }
    .project-title { display: none; }
    .live-metrics { grid-template-columns: repeat(2, minmax(80px, 1fr)); }
  }
</style>
