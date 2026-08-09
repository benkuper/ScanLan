<script lang="ts">
  import { onDestroy, onMount, tick } from 'svelte';
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
    loadLiveReconstructionGuidance,
    loadLiveReconstructionMesh,
    loadLiveReconstructionOverlay,
    localizeSupplementalPhotos,
    loadPreview,
    loadPreviewMesh,
    openProject,
    removeCapture,
    removeMediaSource,
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
    BoundingBoxClip,
    CaptureSettings,
    CaptureStatus,
    CloudTransform,
    DepthFieldOfView,
    LiveReconstructionMode,
    LiveOverlayMode,
    LiveReconstructionGuidance,
    MediaRestartStage,
    MediaSourceSummary,
    MeshRepairProfile,
    MeshViewMode,
    PackedPreviewFrame,
    PreviewMesh,
    PreviewPoint,
    ProjectCatalogEntry,
    ProjectSummary,
    RgbResolution,
    RuntimeInfo,
    SensorKind
  } from './lib/types';

  type Workspace = 'capture' | 'reconstruct' | 'inspect';
  type RenderMode = 'points' | 'mesh' | 'splat';
  type TransformSaveMode = 'auto' | 'manual';
  type ExportKind = 'points' | 'mesh' | 'splat';

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
  let clippingEnabled = false;
  let clipEditMode = false;
  let clipGizmoMode: 'translate' | 'scale' = 'scale';
  let clipBounds: BoundingBoxClip | null = null;

  let previewPoints: PreviewPoint[] = [];
  let packedPreviewFrame: PackedPreviewFrame | null = null;
  let captureDraftFrame: PackedPreviewFrame | null = null;
  let previewMesh: PreviewMesh | null = null;
  let liveMesh: PreviewMesh | null = null;
  let previewSplat: Uint8Array | null = null;
  let assetLoading: RenderMode | null = null;
  let exporting: ExportKind | null = null;

  let buildPointCloud = true;
  let buildTexturedMesh = true;
  let buildGaussianSplat = false;
  let splatIterations = 30_000;
  let mediaRestartStage: MediaRestartStage = 'reuse';
  let rebuildRgbdPreparation = false;

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
  let liveOverlayMode: LiveOverlayMode = 'normal';
  let liveGuidance: LiveReconstructionGuidance | null = null;
  let lastGuidanceAt = 0;
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
  let canClipModel = false;
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
  $: if (mediaSourceCount === 0) mediaRestartStage = 'reuse';
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
      ? liveOverlayMode === 'normal' && project?.settings.liveReconstruction === 'mesh' ? 'mesh' : 'points'
      : renderMode;
  $: viewerMesh = capturing && liveOverlayMode === 'normal' ? liveMesh : previewing ? null : previewMesh;
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
    && (renderMode === 'splat'
      ? Boolean(previewSplat?.byteLength)
      : previewPoints.length > 0 || Boolean(previewMesh?.positions.length));
  $: canClipModel = canEditModel;
  $: hasEditPose = !isIdentityTransform(cloudTransform);
  $: gizmoAnchor = modelCenter(renderMode, previewPoints, previewMesh, previewSplat);
  $: viewerCloudTransform = workspace === 'inspect'
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

  function formatByteSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
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
  const clipStorageKey = (projectId: string) => `scanlan-bounding-box:${projectId}`;
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

  function cloneClipBounds(bounds: BoundingBoxClip): BoundingBoxClip {
    return { min: [...bounds.min], max: [...bounds.max] };
  }

  function validClipBounds(value: unknown): BoundingBoxClip | null {
    if (!value || typeof value !== 'object') return null;
    const candidate = value as Partial<BoundingBoxClip>;
    const min = validVector(candidate.min, [Number.NaN, Number.NaN, Number.NaN]);
    const max = validVector(candidate.max, [Number.NaN, Number.NaN, Number.NaN]);
    return min.every((item, axis) => Number.isFinite(item) && max[axis] - item >= 0.001)
      ? { min, max }
      : null;
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
    clippingEnabled = false;
    clipEditMode = false;
    clipBounds = null;
    const stored = localStorage.getItem(transformStorageKey(projectId));
    if (stored) {
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
    const storedClip = localStorage.getItem(clipStorageKey(projectId));
    if (storedClip) {
      try {
        const parsed = JSON.parse(storedClip) as { enabled?: unknown; bounds?: unknown };
        const bounds = validClipBounds(parsed.bounds);
        if (bounds) {
          clipBounds = bounds;
          clippingEnabled = parsed.enabled === true;
        }
      } catch {
        localStorage.removeItem(clipStorageKey(projectId));
      }
    }
  }

  function persistTransform(): void {
    if (!project) return;
    if (isIdentityTransform(cloudTransform)) localStorage.removeItem(transformStorageKey(project.id));
    else localStorage.setItem(transformStorageKey(project.id), JSON.stringify(cloudTransform));
    savedCloudTransform = cloneTransform(cloudTransform);
    transformDirty = false;
  }

  function persistClip(): void {
    if (!project || !clipBounds) return;
    localStorage.setItem(clipStorageKey(project.id), JSON.stringify({
      enabled: clippingEnabled,
      bounds: clipBounds
    }));
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

  function modelCenter(
    mode: RenderMode,
    points: PreviewPoint[],
    mesh: PreviewMesh | null,
    splat: Uint8Array | null
  ): [number, number, number] {
    let count = 0;
    let stride = 1;
    let pointAt: (index: number) => [number, number, number];
    if (mode === 'mesh' && mesh?.positions.length) {
      count = Math.floor(mesh.positions.length / 3);
      stride = Math.max(1, Math.floor(count / 100_000));
      pointAt = (index) => [mesh.positions[index * 3], mesh.positions[index * 3 + 1], mesh.positions[index * 3 + 2]];
    } else if (mode === 'splat' && splat?.byteLength) {
      count = Math.floor(splat.byteLength / 32);
      stride = Math.max(1, Math.floor(count / 100_000));
      const view = new DataView(splat.buffer, splat.byteOffset, splat.byteLength);
      pointAt = (index) => [
        view.getFloat32(index * 32, true),
        view.getFloat32(index * 32 + 4, true),
        view.getFloat32(index * 32 + 8, true)
      ];
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

  function transformedPosition(position: [number, number, number], transform: CloudTransform): [number, number, number] {
    const [xAngle, yAngle, zAngle] = transform.rotation.map((value) => value * Math.PI / 180);
    const sx = Math.sin(xAngle * 0.5);
    const cx = Math.cos(xAngle * 0.5);
    const sy = Math.sin(yAngle * 0.5);
    const cy = Math.cos(yAngle * 0.5);
    const sz = Math.sin(zAngle * 0.5);
    const cz = Math.cos(zAngle * 0.5);
    const qx = sx * cy * cz + cx * sy * sz;
    const qy = cx * sy * cz - sx * cy * sz;
    const qz = cx * cy * sz + sx * sy * cz;
    const qw = cx * cy * cz - sx * sy * sz;
    const [x, y, z] = position.map((value, axis) => value * transform.scale[axis]);
    const tx = 2 * (qy * z - qz * y);
    const ty = 2 * (qz * x - qx * z);
    const tz = 2 * (qx * y - qy * x);
    return [
      x + qw * tx + (qy * tz - qz * ty) + transform.position[0],
      y + qw * ty + (qz * tx - qx * tz) + transform.position[1],
      z + qw * tz + (qx * ty - qy * tx) + transform.position[2]
    ];
  }

  function fittedClipBounds(): BoundingBoxClip | null {
    let count = 0;
    let pointAt: (index: number) => [number, number, number];
    if (renderMode === 'mesh' && previewMesh?.positions.length) {
      count = Math.floor(previewMesh.positions.length / 3);
      pointAt = (index) => [
        previewMesh!.positions[index * 3],
        previewMesh!.positions[index * 3 + 1],
        previewMesh!.positions[index * 3 + 2]
      ];
    } else if (renderMode === 'splat' && previewSplat?.byteLength) {
      count = Math.floor(previewSplat.byteLength / 32);
      const view = new DataView(previewSplat.buffer, previewSplat.byteOffset, previewSplat.byteLength);
      pointAt = (index) => [
        view.getFloat32(index * 32, true),
        view.getFloat32(index * 32 + 4, true),
        view.getFloat32(index * 32 + 8, true)
      ];
    } else if (previewPoints.length) {
      count = previewPoints.length;
      pointAt = (index) => previewPoints[index].position;
    } else {
      return null;
    }
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (let index = 0; index < count; index += 1) {
      const point = transformedPosition(pointAt(index), cloudTransform);
      if (point.some((value) => !Number.isFinite(value))) continue;
      for (let axis = 0; axis < 3; axis += 1) {
        min[axis] = Math.min(min[axis], point[axis]);
        max[axis] = Math.max(max[axis], point[axis]);
      }
    }
    if (min.some((value) => !Number.isFinite(value))) return null;
    for (let axis = 0; axis < 3; axis += 1) {
      const padding = Math.max(0.005, (max[axis] - min[axis]) * 0.005);
      min[axis] -= padding;
      max[axis] += padding;
    }
    return { min, max };
  }

  function setClippingEnabled(enabled: boolean): void {
    if (enabled && !clipBounds) {
      clipBounds = fittedClipBounds();
      if (!clipBounds) {
        message = 'Load a result before enabling bounding-box clipping.';
        return;
      }
    }
    clippingEnabled = enabled;
    clipEditMode = enabled;
    editMode = false;
    floorPickMode = false;
    persistClip();
    message = enabled
      ? 'Bounding-box clipping enabled after the model pose. Drag the box or edit its limits.'
      : 'Bounding-box clipping disabled; exports will include the full result.';
  }

  function fitBoundingBox(): void {
    const bounds = fittedClipBounds();
    if (!bounds) {
      message = 'Load a result before fitting the bounding box.';
      return;
    }
    clipBounds = bounds;
    clippingEnabled = true;
    persistClip();
    message = 'Bounding box fitted around the transformed result.';
  }

  function setClipGizmoMode(mode: 'translate' | 'scale'): void {
    clipGizmoMode = mode;
    clipEditMode = true;
    editMode = false;
    floorPickMode = false;
  }

  function handleClipBoundsChanged(bounds: BoundingBoxClip): void {
    clipBounds = cloneClipBounds(bounds);
  }

  function commitClipBounds(): void {
    persistClip();
    message = 'Bounding box updated; previews and exports use these final-space limits.';
  }

  function setClipCoordinate(side: 'min' | 'max', axis: number, value: string): void {
    if (!clipBounds) return;
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return;
    const next = cloneClipBounds(clipBounds);
    if (side === 'min') next.min[axis] = Math.min(numeric, next.max[axis] - 0.001);
    else next.max[axis] = Math.max(numeric, next.min[axis] + 0.001);
    clipBounds = next;
    clippingEnabled = true;
    persistClip();
  }

  function setGizmoMode(mode: 'translate' | 'rotate' | 'scale'): void {
    gizmoMode = mode;
    editMode = true;
    floorPickMode = false;
    clipEditMode = false;
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
    if (event.key === 'Escape' && (editMode || floorPickMode || clipEditMode)) {
      editMode = false;
      floorPickMode = false;
      clipEditMode = false;
      return;
    }
    if (!canEditModel || event.repeat || isTextEntryTarget(event.target)) return;
    if (clipEditMode) {
      const clipMode = event.key.toLowerCase() === 'w'
        ? 'translate'
        : event.key.toLowerCase() === 'r'
          ? 'scale'
          : null;
      if (!clipMode) return;
      event.preventDefault();
      setClipGizmoMode(clipMode);
      return;
    }
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

  function updateDepthFieldOfView(value: DepthFieldOfView): void {
    const settings = project?.settings;
    if (!settings) return;
    if (settings.sensorKind !== 'kinect_v2'
      && value === 'wide' && !settings.depthBinned
      && settings.sensorFps > 15) {
      updateSetting('sensorFps', 0);
    }
    if (settings.sensorKind === 'azure_kinect'
      && (settings.rgbResolution === '2160p' || settings.rgbResolution === '3072p')
      && settings.sensorFps === 0
      && (value !== 'wide' || settings.depthBinned)) {
      updateSetting('rgbResolution', 'auto');
    }
    updateSetting('depthFieldOfView', value);
  }

  function updateDepthBinning(value: boolean): void {
    const settings = project?.settings;
    if (!settings) return;
    if (settings.sensorKind !== 'kinect_v2'
      && settings.depthFieldOfView === 'wide' && !value
      && settings.sensorFps > 15) {
      updateSetting('sensorFps', 0);
    }
    if (settings.sensorKind === 'azure_kinect'
      && (settings.rgbResolution === '2160p' || settings.rgbResolution === '3072p')
      && settings.sensorFps === 0
      && value) {
      updateSetting('rgbResolution', 'auto');
    }
    updateSetting('depthBinned', value);
  }

  function updateSensorFps(value: number): void {
    const settings = project?.settings;
    if (!settings) return;
    if (settings.sensorKind === 'azure_kinect'
      && (settings.rgbResolution === '2160p' || settings.rgbResolution === '3072p')
      && (value === 30 || (value === 0 && !(settings.depthFieldOfView === 'wide' && !settings.depthBinned)))) {
      updateSetting('rgbResolution', 'auto');
    }
    updateSetting('sensorFps', value);
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
      ...(changingFamily ? cameraDefaults(candidate.kind) : {}),
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
      ...cameraDefaults(kind),
      sensorKind: kind,
      sensorId: '',
      sensorConnection: 'usb',
      sensorAddress: '',
      useImu: kind !== 'kinect_v2'
    };
  }

  function cameraDefaults(kind: SensorKind): Partial<CaptureSettings> {
    if (kind === 'femto_mega') {
      return {
        rgbResolution: 'auto',
        sensorFps: 0,
        rgbGain: 30,
        rgbBrightness: 0,
        rgbContrast: 32,
        rgbSaturation: 54,
        rgbSharpness: 6,
        rgbBacklightCompensation: false,
        imuAccelRateHz: 0,
        imuAccelRangeG: 0,
        imuGyroRateHz: 0,
        imuGyroRangeDps: 0
      };
    }
    return {
      rgbResolution: 'auto',
      sensorFps: 0,
      rgbGain: 0,
      rgbBrightness: 128,
      rgbContrast: 5,
      rgbSaturation: 32,
      rgbSharpness: 2,
      rgbBacklightCompensation: false,
      imuAccelRateHz: 0,
      imuAccelRangeG: 0,
      imuGyroRateHz: 0,
      imuGyroRangeDps: 0
    };
  }

  function effectiveSensorFps(settings: CaptureSettings): number {
    if (settings.sensorFps > 0) return settings.sensorFps;
    return settings.depthFieldOfView === 'wide' && !settings.depthBinned ? 15 : 30;
  }

  function rgbExposureLimitUs(settings: CaptureSettings): number {
    const sensorFps = effectiveSensorFps(settings);
    return sensorFps <= 5 ? 190_000 : sensorFps <= 15 ? 60_000 : 30_000;
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

  function setLiveOverlayMode(mode: LiveOverlayMode): void {
    if (liveOverlayMode === mode) return;
    liveOverlayMode = mode;
    packedPreviewFrame = null;
    lastPreviewFrame = 0;
    void pollLiveGeometry();
  }

  async function pollLiveGeometry(): Promise<void> {
    if (!project || !liveSensor || geometryInFlight) return;
    geometryInFlight = true;
    try {
      const source = capturing && liveOverlayMode !== 'normal'
        ? loadLiveReconstructionOverlay(liveOverlayMode, lastPreviewFrame)
        : loadLivePreviewFrame(lastPreviewFrame);
      const packet = parsePointPacket(await source);
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
      const now = performance.now();
      if (capturing && now - lastGuidanceAt >= 500) {
        lastGuidanceAt = now;
        liveGuidance = await loadLiveReconstructionGuidance();
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
    clipEditMode = false;
    await ensureSensorPreview();
  }

  async function showReconstructWorkspace(): Promise<void> {
    if (!project || capturing) return;
    workspace = 'reconstruct';
    editMode = false;
    floorPickMode = false;
    clipEditMode = false;
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
    if (project.settings.lingbotDepthRefinement && !runtime?.splatWorkerAvailable) {
      message = 'LingBot depth refinement requires the packaged CUDA runtime.';
      return;
    }
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
    if (settingsDirty && !(await persistSettings())) return;
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
      activeJob = await startArtifactJob(project.path, targets, splatIterations, {
        mediaRestart: mediaRestartStage,
        rebuildRgbd: rebuildRgbdPreparation
      });
      workspace = 'reconstruct';
      const restartDetail = mediaRestartStage === 'decode'
        ? ' Media decoding and analysis will be rebuilt.'
        : mediaRestartStage === 'analysis'
          ? ' Decoded media will be reused; camera analysis will be rebuilt.'
          : rebuildRgbdPreparation
            ? mediaSourceCount > 0
              ? ' Decoded media will be reused; RGB-D tracking and downstream analysis will be rebuilt.'
              : ' RGB-D tracking, fusion, and downstream outputs will be rebuilt.'
            : mediaSourceCount > 0
              ? ' Valid decoded and analyzed data will be reused.'
              : ' Valid reconstruction caches will be reused.';
      message = (mediaOnlyProject
        ? 'Started photo/video camera solving and photoreal Gaussian reconstruction.'
        : mediaSourceCount > 0
          ? 'Started metric RGB-D reconstruction with high-resolution media enhancement.'
          : 'Started quality-gated RGB-D reconstruction.') + restartDetail;
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
      buildPointCloud = completedCaptures > 0;
      buildTexturedMesh = completedCaptures > 0;
      buildGaussianSplat = true;
      workspace = 'reconstruct';
      message = completedCaptures > 0
        ? `Imported ${paths.length} high-resolution source${paths.length === 1 ? '' : 's'}. They will enhance point colors, mesh textures, and splat appearance.`
        : `Imported ${paths.length} source${paths.length === 1 ? '' : 's'}. Ready to solve cameras and train a photoreal 3D Gaussian splat.`;
    } catch (error) {
      message = errorText(error);
    } finally {
      busy = false;
    }
  }

  async function removeImportedMediaSource(source: MediaSourceSummary): Promise<void> {
    if (!project || busy || processing || capturing) return;
    const invalidation = readyArtifacts > 0
      ? ' Existing reconstruction outputs will be marked stale.'
      : '';
    if (!window.confirm(`Remove ${source.name} from this project? ScanLan's imported copy will be deleted; the original file remains untouched.${invalidation}`)) return;
    busy = true;
    try {
      project = await removeMediaSource(project.path, source.id);
      previewPoints = [];
      previewMesh = null;
      previewSplat = null;
      lastBuildSplatSignature = '';
      captureDraftFrame = completedCaptures > 0
        ? parsePointPacket(await loadCaptureDraft(project.path).catch(() => new ArrayBuffer(0)))
        : null;
      message = `${source.name} removed. ${project.mediaSources.length} imported source${project.mediaSources.length === 1 ? '' : 's'} remaining.${invalidation}`;
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
      localStorage.removeItem(clipStorageKey(entry.id));
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
    if (!project || exporting || !artifactReady('pointCloud')) return;
    const destination = await save({ title: 'Export metric point cloud', defaultPath: 'scan-cloud.ply', filters: [{ name: 'PLY point cloud', extensions: ['ply'] }] });
    if (!destination) return;
    exporting = 'points';
    message = `Exporting${clippingEnabled ? ' clipped' : ''} point cloud…`;
    await tick();
    try {
      message = `Aligned${clippingEnabled ? ' and clipped' : ''} point cloud exported to ${await exportPly(project.path, destination, cloudTransform, clippingEnabled ? clipBounds : null)}.`;
    } catch (error) {
      message = `Point cloud export failed: ${errorText(error)}`;
    } finally {
      exporting = null;
    }
  }

  async function exportMesh(): Promise<void> {
    if (!project || exporting || !artifactReady('texturedMesh')) return;
    const destination = await save({ title: 'Export textured mesh bundle', defaultPath: 'scan-mesh.obj', filters: [{ name: 'Wavefront OBJ', extensions: ['obj'] }] });
    if (!destination) return;
    exporting = 'mesh';
    message = `Exporting${clippingEnabled ? ' clipped' : ''} textured mesh bundle…`;
    await tick();
    try {
      message = `Aligned${clippingEnabled ? ' and clipped' : ''} OBJ, MTL, and texture exported beside ${await exportTexturedMesh(project.path, destination, cloudTransform, clippingEnabled ? clipBounds : null)}.`;
    } catch (error) {
      message = `Mesh export failed: ${errorText(error)}`;
    } finally {
      exporting = null;
    }
  }

  async function exportSplat(): Promise<void> {
    if (!project || exporting || !artifactReady('gaussianSplat')) return;
    const splatKind = mediaOnlyProject ? 'photoreal 3D Gaussian splat' : 'metric 2D Gaussian surface';
    const destination = await save({ title: `Export ${splatKind}`, defaultPath: mediaOnlyProject ? 'scan-3dgs.ply' : 'scan-2dgs.ply', filters: [{ name: 'Gaussian PLY', extensions: ['ply'] }] });
    if (!destination) return;
    exporting = 'splat';
    message = `Exporting${clippingEnabled ? ' clipped' : ''} ${splatKind}…`;
    await tick();
    try {
      message = `Aligned${clippingEnabled ? ' and clipped' : ''} Gaussian surface and coordinate sidecars exported to ${await exportGaussianSplat(project.path, destination, cloudTransform, clippingEnabled ? clipBounds : null)}.`;
    } catch (error) {
      message = `Gaussian export failed: ${errorText(error)}`;
    } finally {
      exporting = null;
    }
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
      <span>02</span><div><strong>Reconstruct</strong><small>{processing ? activeJob?.stage.replaceAll('_', ' ') : 'Points · mesh · Gaussian splat'}</small></div>
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
        splatRepresentation={mediaOnlyProject ? '3d' : '2d'}
        {meshViewMode}
        assetLoading={assetLoading}
        floorPickMode={floorPickMode && canEditModel}
        cloudTransform={viewerCloudTransform}
        {gizmoAnchor}
        editMode={editMode && canEditModel}
        {gizmoMode}
        {rotationSnapDegrees}
        clipBounds={workspace === 'inspect' && clippingEnabled ? clipBounds : null}
        clipEditMode={workspace === 'inspect' && clipEditMode}
        {clipGizmoMode}
        onFloorDetected={setFloorTransform}
        onFloorMessage={(value) => message = value}
        onTransformChanged={handleGizmoTransform}
        onTransformCommitted={commitGizmoTransform}
        onClipBoundsChanged={handleClipBoundsChanged}
        onClipBoundsCommitted={commitClipBounds}
      />

      {#if capturing}
        <div class="live-overlays" aria-label="Live reconstruction view">
          <button class:active={liveOverlayMode === 'normal'} on:click={() => setLiveOverlayMode('normal')}>Normal</button>
          <button class:active={liveOverlayMode === 'coverage'} on:click={() => setLiveOverlayMode('coverage')}>Coverage</button>
          <button class:active={liveOverlayMode === 'tracking'} on:click={() => setLiveOverlayMode('tracking')}>Tracking</button>
          <button class:active={liveOverlayMode === 'confidence'} on:click={() => setLiveOverlayMode('confidence')}>Confidence</button>
        </div>
      {/if}

      {#if liveSensor && sensor}
        <div class="live-metrics">
          <div><span>{capturing ? 'Tracking' : 'Camera'}</span><strong class:good={capturing ? sensor.trackingState === 'tracking' || sensor.trackingState === 'relocalized' : sensor.sensorConnected}>{capturing ? sensor.trackingState.toUpperCase() : sensor.sensorConnected ? 'CONNECTED' : 'WAITING'}</strong></div>
          <div><span>{capturing ? 'Tracker' : 'Stream'}</span><strong>{(capturing ? sensor.trackingFps : sensor.streamFps).toFixed(1)} fps</strong></div>
          <div><span>{capturing ? 'Keyframes' : 'Frames seen'}</span><strong>{capturing ? sensor.liveIntegratedFrameCount : sensor.liveProcessedFrameCount}</strong></div>
          <div><span>Confidence</span><strong>{Math.round(sensor.trackingConfidence * 100)}%</strong></div>
          <div><span>Overlap</span><strong>{Math.round(sensor.trackingOverlap * 100)}%</strong></div>
          <div><span>Depth error</span><strong>{sensor.depthRmseMm ? `${sensor.depthRmseMm.toFixed(1)} mm` : '—'}</strong></div>
          <div><span>Queue drops</span><strong>{sensor.trackingQueueDropCount + sensor.mappingDropCount}</strong></div>
          <div><span>Map update</span><strong>{sensor.mapUpdateHz.toFixed(1)} Hz</strong></div>
          <div><span>Submaps</span><strong>{sensor.residentSubmapCount} GPU / {sensor.hostCachedSubmapCount} host</strong></div>
          <div><span>Live memory</span><strong>{formatByteSize(sensor.allocatedLiveMapBytes)}</strong></div>
          <div><span>Pose age</span><strong>{sensor.poseLatencyMs == null ? '—' : `${sensor.poseLatencyMs.toFixed(0)} ms`}</strong></div>
          <div><span>Pressure</span><strong>{sensor.degradationLevel ? `LEVEL ${sensor.degradationLevel}` : 'NORMAL'}</strong></div>
          <div><span>Loop closures</span><strong>{sensor.loopClosureCount}{sensor.loopCorrectionActive ? ' · SMOOTHING' : ''}</strong></div>
        </div>
        {#if capturing && liveGuidance?.coverage?.guidance?.length}
          <div class="live-guidance">
            <strong>{liveGuidance.coverage.guidance[0]}</strong>
            <span>{Math.round(liveGuidance.coverage.observedRatio * 100)}% well observed · {Math.round(liveGuidance.coverage.singleViewRatio * 100)}% single-view</span>
          </div>
        {/if}
      {/if}

      {#if processing && activeJob}
        <div class="job-overlay">
          <div><span>{activeJob.stage.replaceAll('_', ' ')}</span><strong>{Math.round(activeJob.progress * 100)}%</strong></div>
          <div class="progress"><i style={`width:${Math.round(activeJob.progress * 100)}%`}></i></div>
          {#if activeJob.stageProgress != null}
            <div class="stage-progress-meta"><span>Current stage</span><strong>{Math.round(activeJob.stageProgress * 100)}%</strong></div>
            <div class="progress stage-progress"><i style={`width:${Math.round(activeJob.stageProgress * 100)}%`}></i></div>
          {/if}
          {#if activeJob.iteration !== null}
            <div class="job-quality">
              <span>Iteration <strong>{activeJob.iteration.toLocaleString()} / {activeJob.totalIterations?.toLocaleString()}</strong></span>
              <span>Current loss <strong>{activeJob.loss?.toFixed(4) ?? '—'}</strong></span>
              <span>Rolling loss <strong>{activeJob.smoothedLoss?.toFixed(4) ?? 'warming up'}</strong></span>
            </div>
          {/if}
          <p>{activeJob.detail}</p>
          <div class="job-meta">
            <span>{activeJob.computeBackend ?? 'Waiting for worker'}</span>
            <span>{activeJob.elapsedSeconds != null ? `${formatDuration(activeJob.elapsedSeconds)} elapsed` : ''}</span>
          </div>
        </div>
      {/if}
    </section>

    <aside>
      {#if !project}
        <details class="panel collapsible-panel" open>
          <summary><span>SCANLAN</span><strong>STARTING</strong></summary>
          <div class="collapsible-body startup-panel"><div class="spinner"></div><h2>Starting ScanLan</h2><p>{message}</p></div>
        </details>
      {:else if workspace === 'capture'}
        <header class="workspace-heading">
          <div><span>RGB-D SOURCE</span><h2>{liveSensor ? sensor?.sensorName ?? 'Live camera' : 'Camera & live fusion'}</h2></div>
          <button class="icon-button" on:click={discoverSensors} disabled={discovering || selectingSensor || capturing || processing} title="Refresh cameras">↻</button>
        </header>

        <details class="panel collapsible-panel" open>
          <summary><span>CAPTURE SETTINGS</span><strong>{project.settings.sensorKind.replaceAll('_', ' ').toUpperCase()}</strong></summary>
          <div class="collapsible-body settings">
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
            {#if project.settings.sensorKind !== 'kinect_v2'}
              <label>Sensor rate
                <select value={project.settings.sensorFps} on:change={(event) => updateSensorFps(Number(inputValue(event)))} disabled={capturing || processing}>
                  <option value={0}>Auto · fastest</option><option value={5}>5 fps</option><option value={15}>15 fps</option>{#if project.settings.sensorKind === 'femto_mega'}<option value={25} disabled={project.settings.depthFieldOfView === 'wide' && !project.settings.depthBinned}>25 fps</option>{/if}<option value={30} disabled={project.settings.depthFieldOfView === 'wide' && !project.settings.depthBinned}>30 fps</option>
                </select>
              </label>
            {/if}
            <label>Depth limit
              <div class="unit-input"><input type="number" min="0.8" max="8" step="0.1" value={project.settings.maxDepthM} on:change={(event) => updateSetting('maxDepthM', Number(inputValue(event)))} disabled={capturing || processing}/><span>m</span></div>
            </label>
            <label>Fusion voxel
              <div class="unit-input"><input type="number" min="3" max="40" step="1" value={project.settings.voxelSizeMm} on:change={(event) => updateSetting('voxelSizeMm', Number(inputValue(event)))} disabled={capturing || processing}/><span>mm</span></div>
            </label>
            <label>Live map budget
              <div class="unit-input"><input type="number" min="256" max="4096" step="128" value={project.settings.liveMapMemoryMib} on:change={(event) => updateSetting('liveMapMemoryMib', Number(inputValue(event)))} disabled={capturing || processing}/><span>MiB</span></div>
            </label>
            {#if project.settings.sensorKind !== 'kinect_v2'}
              <label>Depth FOV
                <select value={project.settings.depthFieldOfView} on:change={(event) => updateDepthFieldOfView(inputValue(event) as DepthFieldOfView)} disabled={capturing || processing}>
                  <option value="narrow">Narrow</option><option value="wide">Wide</option>
                </select>
              </label>
              <label>Depth sampling
                <select value={project.settings.depthBinned ? 'binned' : 'full'} on:change={(event) => updateDepthBinning(inputValue(event) === 'binned')} disabled={capturing || processing}>
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
          </div>
        </details>

        <details class="panel advanced-settings" open>
          <summary><span>RGB CAMERA</span><strong>{project.settings.sensorKind === 'kinect_v2' ? 'FIXED BY SDK' : 'SENSOR + ARCHIVE'}</strong></summary>
          {#if project.settings.sensorKind === 'kinect_v2'}
            <p>Kinect v2 has a fixed 1920×1080, 30 fps color stream. Its public SDK exposes the current exposure, gain, gamma, and frame interval as read-only values, so ScanLan cannot safely override them.</p>
          {:else}
            <div class="advanced-body">
              <label>Sensor RGB resolution
                <select value={project.settings.rgbResolution} on:change={(event) => updateSetting('rgbResolution', inputValue(event) as RgbResolution)} disabled={capturing || processing}>
                  <option value="auto">Automatic · best compatible</option>
                  <option value="720p">1280×720</option>
                  <option value="1080p">1920×1080</option>
                  <option value="1440p">2560×1440</option>
                  {#if project.settings.sensorKind === 'azure_kinect'}<option value="1536p">2048×1536 · 4:3</option>{/if}
                  <option value="2160p" disabled={project.settings.sensorKind === 'azure_kinect' && (project.settings.sensorFps === 30 || (project.settings.sensorFps === 0 && !(project.settings.depthFieldOfView === 'wide' && !project.settings.depthBinned)))}>3840×2160{project.settings.sensorKind === 'azure_kinect' ? ' · 5/15 fps' : ''}</option>
                  {#if project.settings.sensorKind === 'azure_kinect'}<option value="3072p" disabled={project.settings.sensorFps === 30 || (project.settings.sensorFps === 0 && !(project.settings.depthFieldOfView === 'wide' && !project.settings.depthBinned))}>4096×3072 · 5/15 fps</option>{/if}
                </select>
              </label>

              <label class="toggle"><input type="checkbox" checked={project.settings.rgbAutoExposure} on:change={(event) => updateSetting('rgbAutoExposure', inputChecked(event))} disabled={capturing || processing}/><span></span><div><strong>Auto exposure</strong><small>Disable to lock exposure and gain across the take</small></div></label>
              {#if !project.settings.rgbAutoExposure}
                <div class="setting-grid">
                  <label class="slider-control"><span>Exposure <output>{(project.settings.rgbExposureUs / 1000).toFixed(1)} ms</output></span>
                    <input type="range" min="100" max={rgbExposureLimitUs(project.settings)} step="100" value={project.settings.rgbExposureUs} on:input={(event) => updateSetting('rgbExposureUs', Number(inputValue(event)))} disabled={capturing || processing}/>
                  </label>
                  <label class="slider-control"><span>Gain <output>{project.settings.rgbGain}</output></span>
                    <input type="range" min={project.settings.sensorKind === 'femto_mega' ? 1 : 0} max={project.settings.sensorKind === 'femto_mega' ? 240 : 255} step="1" value={project.settings.rgbGain} on:input={(event) => updateSetting('rgbGain', Number(inputValue(event)))} disabled={capturing || processing}/>
                  </label>
                </div>
              {/if}

              <label class="toggle"><input type="checkbox" checked={project.settings.rgbAutoWhiteBalance} on:change={(event) => updateSetting('rgbAutoWhiteBalance', inputChecked(event))} disabled={capturing || processing}/><span></span><div><strong>Auto white balance</strong><small>Disable to keep color temperature consistent between frames</small></div></label>
              {#if !project.settings.rgbAutoWhiteBalance}
                <label class="slider-control"><span>White balance <output>{project.settings.rgbWhiteBalanceK} K</output></span>
                  <input type="range" min="2000" max={project.settings.sensorKind === 'femto_mega' ? 11000 : 12500} step="10" value={project.settings.rgbWhiteBalanceK} on:input={(event) => updateSetting('rgbWhiteBalanceK', Number(inputValue(event)))} disabled={capturing || processing}/>
                </label>
              {/if}

              <label class="toggle"><input type="checkbox" checked={project.settings.rgbColorAdjustmentsEnabled} on:change={(event) => updateSetting('rgbColorAdjustmentsEnabled', inputChecked(event))} disabled={capturing || processing}/><span></span><div><strong>Manual image processing</strong><small>Apply deterministic brightness, contrast, color, and sharpening</small></div></label>
              {#if project.settings.rgbColorAdjustmentsEnabled}
                <div class="setting-grid compact-settings">
                  <label class="slider-control"><span>Brightness <output>{project.settings.rgbBrightness}</output></span><input type="range" min="0" max={project.settings.sensorKind === 'femto_mega' ? 128 : 255} step="1" value={project.settings.rgbBrightness} on:input={(event) => updateSetting('rgbBrightness', Number(inputValue(event)))} disabled={capturing || processing}/></label>
                  <label class="slider-control"><span>Contrast <output>{project.settings.rgbContrast}</output></span><input type="range" min={project.settings.sensorKind === 'femto_mega' ? 1 : 0} max={project.settings.sensorKind === 'femto_mega' ? 60 : 10} step="1" value={project.settings.rgbContrast} on:input={(event) => updateSetting('rgbContrast', Number(inputValue(event)))} disabled={capturing || processing}/></label>
                  <label class="slider-control"><span>Saturation <output>{project.settings.rgbSaturation}</output></span><input type="range" min={project.settings.sensorKind === 'femto_mega' ? 1 : 0} max={project.settings.sensorKind === 'femto_mega' ? 80 : 63} step="1" value={project.settings.rgbSaturation} on:input={(event) => updateSetting('rgbSaturation', Number(inputValue(event)))} disabled={capturing || processing}/></label>
                  <label class="slider-control"><span>Sharpness <output>{project.settings.rgbSharpness}</output></span><input type="range" min={project.settings.sensorKind === 'femto_mega' ? 1 : 0} max={project.settings.sensorKind === 'femto_mega' ? 15 : 4} step="1" value={project.settings.rgbSharpness} on:input={(event) => updateSetting('rgbSharpness', Number(inputValue(event)))} disabled={capturing || processing}/></label>
                </div>
                {#if project.settings.sensorKind === 'azure_kinect'}
                  <label class="toggle"><input type="checkbox" checked={project.settings.rgbBacklightCompensation} on:change={(event) => updateSetting('rgbBacklightCompensation', inputChecked(event))} disabled={capturing || processing}/><span></span><div><strong>Backlight compensation</strong><small>Lift a dark subject against a bright background</small></div></label>
                {/if}
              {/if}

              <div class="setting-grid">
                <label>Anti-flicker
                  <select value={project.settings.rgbPowerlineHz} on:change={(event) => updateSetting('rgbPowerlineHz', Number(inputValue(event)))} disabled={capturing || processing}>
                    <option value={0}>Camera default</option><option value={50}>50 Hz</option><option value={60}>60 Hz</option>
                  </select>
                </label>
                <label>JPEG quality
                  <div class="unit-input"><input type="number" min="60" max="100" step="1" value={project.settings.rgbJpegQuality} on:change={(event) => updateSetting('rgbJpegQuality', Number(inputValue(event)))} disabled={capturing || processing}/><span>%</span></div>
                </label>
                <label>Archived RGB size
                  <select value={project.settings.maxRgbDimension} on:change={(event) => updateSetting('maxRgbDimension', Number(inputValue(event)))} disabled={capturing || processing}>
                    <option value={0}>Native sensor size</option><option value={3840}>Max 3840 px</option><option value={2560}>Max 2560 px</option><option value={1920}>Max 1920 px</option><option value={1280}>Max 1280 px</option>
                  </select>
                </label>
              </div>
              <p>For texture quality, use native archive size and JPEG 95–100. Lock exposure and white balance after the preview looks correct; excessive exposure causes motion blur even when depth tracking remains stable.</p>
            </div>
          {/if}
        </details>

        <details class="panel advanced-settings" open>
          <summary><span>IMU</span><strong>{project.settings.sensorKind === 'femto_mega' ? 'CONFIGURABLE' : project.settings.sensorKind === 'azure_kinect' ? 'FIXED PROFILE' : 'UNAVAILABLE'}</strong></summary>
          <div class="advanced-body">
            <label class="toggle"><input type="checkbox" checked={project.settings.useImu} on:change={(event) => updateSetting('useImu', inputChecked(event))} disabled={capturing || processing || project.settings.sensorKind === 'kinect_v2'}/><span></span><div><strong>IMU motion prior</strong><small>Improves fast-rotation initialization</small></div></label>
            {#if project.settings.useImu && project.settings.sensorKind === 'femto_mega'}
              <div class="setting-grid">
                <label>Accelerometer rate
                  <select value={project.settings.imuAccelRateHz} on:change={(event) => updateSetting('imuAccelRateHz', Number(inputValue(event)))} disabled={capturing || processing}>
                    <option value={0}>Device default</option><option value={50}>50 Hz</option><option value={100}>100 Hz</option><option value={200}>200 Hz</option><option value={500}>500 Hz</option><option value={1000}>1000 Hz</option><option value={2000}>2000 Hz</option>
                  </select>
                </label>
                <label>Accelerometer range
                  <select value={project.settings.imuAccelRangeG} on:change={(event) => updateSetting('imuAccelRangeG', Number(inputValue(event)))} disabled={capturing || processing}>
                    <option value={0}>Device default</option><option value={2}>±2 g · most sensitive</option><option value={4}>±4 g</option><option value={8}>±8 g</option><option value={16}>±16 g</option>
                  </select>
                </label>
                <label>Gyroscope rate
                  <select value={project.settings.imuGyroRateHz} on:change={(event) => updateSetting('imuGyroRateHz', Number(inputValue(event)))} disabled={capturing || processing}>
                    <option value={0}>Device default</option><option value={50}>50 Hz</option><option value={100}>100 Hz</option><option value={200}>200 Hz</option><option value={500}>500 Hz</option><option value={1000}>1000 Hz</option><option value={2000}>2000 Hz</option>
                  </select>
                </label>
                <label>Gyroscope range
                  <select value={project.settings.imuGyroRangeDps} on:change={(event) => updateSetting('imuGyroRangeDps', Number(inputValue(event)))} disabled={capturing || processing}>
                    <option value={0}>Device default</option><option value={125}>±125 °/s · most sensitive</option><option value={250}>±250 °/s</option><option value={500}>±500 °/s</option><option value={1000}>±1000 °/s</option><option value={2000}>±2000 °/s</option>
                  </select>
                </label>
              </div>
              <p>A faster rate resolves quick motion better. A narrower range gives finer quantization; choose ±8 g only if ±2/±4 g clips during abrupt movement. For handheld scanning, ±4 g and ±500 °/s are a balanced starting point.</p>
            {:else if project.settings.useImu && project.settings.sensorKind === 'azure_kinect'}
              <p>Azure Kinect exposes its factory-calibrated IMU stream but no SDK controls for sample rate or full-scale range. ScanLan drains it at the device rate and reports the measured rate below.</p>
            {:else if project.settings.sensorKind === 'kinect_v2'}
              <p>Kinect v2 does not contain an SDK-accessible IMU.</p>
            {/if}
          </div>
        </details>

        {#if !capturing}
          <details class="panel collapsible-panel connection-card" class:connected={selectedSensorConnected} class:warning={!selectedSensorConnected} open>
            <summary><span>DEVICE</span><strong class:good={selectedSensorConnected}>{selectedSensorConnected ? 'CONNECTED' : project.settings.sensorKind === 'kinect_v2' ? 'READY TO OPEN' : 'NOT CONNECTED'}</strong></summary>
            <div class="collapsible-body status-body">
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
            </div>
          </details>
        {/if}

        {#if liveSensor && sensor}
          <details class="panel collapsible-panel tracking-card" class:warning={!sensor.tracking} open>
            <summary><span>LIVE TRACKING</span><strong class:good={sensor.tracking}>{sensor.tracking ? 'LOCKED' : 'SEARCHING'}</strong></summary>
            <div class="collapsible-body status-body">
              <div class="tracking-title"><i></i><div><strong>{sensor.trackingStatus}</strong><small>{sensor.liveReconstructionBackend ?? 'Realtime engine'}</small></div></div>
              <div class="mini-grid">
                <div><span>Sensor</span><strong>{sensor.streamFps.toFixed(1)} fps</strong></div>
                <div><span>{capturing ? 'Raw archive' : 'Recording'}</span><strong>{capturing ? sensor.frameCount : 'OFF'}</strong></div>
                <div><span>{capturing ? 'Tracked' : 'Frames seen'}</span><strong>{capturing ? Math.max(0, sensor.liveProcessedFrameCount - sensor.liveRejectedFrameCount) : sensor.liveProcessedFrameCount}</strong></div>
                <div><span>Rejected</span><strong>{sensor.liveRejectedFrameCount}</strong></div>
                <div><span>Source drops</span><strong>{sensor.sourceDropCount}</strong></div>
              </div>
              <p>Raw RGB-D stays recoverable for the offline pass. Rejected live poses never enter the fused map; hold a previously scanned view steady to relocalize.</p>
            </div>
          </details>
        {/if}

        <button class:stop={capturing} class="capture-button" on:click={captureAction} disabled={busy || selectingSensor || processing || photoLocalizationActive || (!capturing && (mediaSourceCount > 0 || (runtime && !runtime.sensorWorkerAvailable)))}>
          <i></i><span>{capturing ? 'Stop & save take' : busy ? 'Starting recording…' : 'Start capture'}</span>
        </button>
        <button class="ghost full" on:click={addMediaSource} disabled={busy || capturing || processing || !runtime?.splatWorkerAvailable}>Import high-quality photos or video…</button>

        <details class="panel collapsible-panel takes" open>
          <summary><span>RECORDED TAKES</span><strong>{totalFrames.toLocaleString()} RAW FRAMES</strong></summary>
          <div class="collapsible-body">
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
          </div>
        </details>

      {:else if workspace === 'reconstruct'}
        <header class="workspace-heading"><div><span>PRODUCTION PASS</span><h2>Reconstruction outputs</h2></div><strong class="take-total">{mediaOnlyProject ? `${mediaSourceCount} media source${mediaSourceCount === 1 ? '' : 's'}` : `${completedCaptures} take${completedCaptures === 1 ? '' : 's'}`}</strong></header>

        <details class="panel collapsible-panel" open>
          <summary><span>OUTPUTS</span><strong>{readyArtifacts} READY</strong></summary>
          <div class="collapsible-body target-list">
            <label class:active={buildPointCloud}><input type="checkbox" bind:checked={buildPointCloud} disabled={processing || mediaOnlyProject}/><span class="target-icon">P</span><div><strong>Metric point cloud</strong><small>{mediaOnlyProject ? 'Requires calibrated RGB-D capture' : 'Filtered colored PLY · quickest'}</small></div><i>{artifactReady('pointCloud') ? 'READY' : ''}</i></label>
            <label class:active={buildTexturedMesh}><input type="checkbox" bind:checked={buildTexturedMesh} disabled={processing || mediaOnlyProject}/><span class="target-icon">M</span><div><strong>Textured triangle mesh</strong><small>{mediaOnlyProject ? 'Requires calibrated RGB-D capture' : 'TSDF surface · OBJ/MTL/PNG'}</small></div><i>{artifactReady('texturedMesh') ? 'READY' : ''}</i></label>
            <label class:active={buildGaussianSplat}><input type="checkbox" bind:checked={buildGaussianSplat} disabled={processing || !runtime?.splatWorkerAvailable}/><span class="target-icon">G</span><div><strong>{mediaOnlyProject ? 'Photoreal 3D Gaussian splat' : '2D Gaussian surface'}</strong><small>{mediaOnlyProject ? 'LingBot dense geometry · COLMAP refinement · SH degree 3' : 'Depth-aware discs · metric PLY'}</small></div><i>{artifactReady('gaussianSplat') ? 'READY' : runtime?.splatWorkerAvailable ? '' : 'CUDA RUNTIME MISSING'}</i></label>
            {#if buildGaussianSplat}
              <label class="iterations"><span>Training iterations</span><input type="range" min="5000" max="60000" step="5000" bind:value={splatIterations} disabled={processing}/><strong>{Number(splatIterations).toLocaleString()}</strong></label>
            {/if}
          </div>
        </details>

        {#if !mediaOnlyProject}
          <details class="panel collapsible-panel" open>
            <summary><span>DEPTH REFINEMENT</span><strong>{project.settings.lingbotDepthRefinement ? 'LINGBOT V0.5' : 'SENSOR ONLY'}</strong></summary>
            <div class="collapsible-body settings mesh-repair-settings">
              <label class="toggle"><input type="checkbox" checked={project.settings.lingbotDepthRefinement} on:change={(event) => updateSetting('lingbotDepthRefinement', inputChecked(event))} disabled={processing || !runtime?.splatWorkerAvailable}/><span></span><div><strong>Fill RGB-D depth gaps with LingBot-Depth</strong><small>Runs after camera tracking; preserves measured depth and accepts only metric, RGB-aligned, multi-view-consistent predictions</small></div></label>
              <p>Generated pixels carry lower fusion weight and are exported with confidence and provenance masks. Camera poses always remain tied to the original sensor depth.</p>
              {#if !runtime?.splatWorkerAvailable}
                <p class="warning">The packaged CUDA runtime is required for offline LingBot-Depth inference.</p>
              {:else if project.depthRefinement?.enabled}
                <div class="repair-result" class:warning={(project.depthRefinement.acceptedFrameCount ?? 0) < (project.depthRefinement.frameCount ?? 0)}>
                  <span>LAST QUALITY GATE</span>
                  <div><strong>{project.depthRefinement.acceptedFrameCount ?? 0}</strong><small>frames accepted</small></div>
                  <div><strong>{project.depthRefinement.generatedPixelCount ?? 0}</strong><small>pixels accepted</small></div>
                  <div><strong>{Math.max(0, (project.depthRefinement.frameCount ?? 0) - (project.depthRefinement.acceptedFrameCount ?? 0))}</strong><small>frames rejected</small></div>
                  <div><strong>{Math.round((project.depthRefinement.generatedFusionWeight ?? 0) * 100)}%</strong><small>fusion weight</small></div>
                </div>
              {/if}
            </div>
          </details>
        {/if}

        {#if buildTexturedMesh && !mediaOnlyProject}
          <details class="panel collapsible-panel" open>
            <summary><span>MESH REPAIR</span><strong>DEPTH-AWARE</strong></summary>
            <div class="collapsible-body settings mesh-repair-settings">
            <label class="toggle"><input type="checkbox" checked={project.settings.repairMesh} on:change={(event) => updateSetting('repairMesh', inputChecked(event))} disabled={processing}/><span></span><div><strong>Repair mesh before texturing</strong><small>Fixes topology and fills only holes supported by captured depth</small></div></label>
            {#if project.settings.repairMesh}
              <label>Repair profile
                <select value={project.settings.meshRepairProfile} on:change={(event) => updateSetting('meshRepairProfile', inputValue(event) as MeshRepairProfile)} disabled={processing}>
                  <option value="faithful">Faithful · preserve measured geometry</option>
                  <option value="architectural">Architectural · planar wall patches</option>
                  <option value="natural">Natural · smoothly faired patches</option>
                </select>
              </label>
              <p>{project.settings.meshRepairProfile === 'architectural' ? 'Projects new wall and floor patch vertices onto their fitted plane.' : project.settings.meshRepairProfile === 'natural' ? 'Fairs new patch vertices for rounded and organic surfaces.' : 'Triangulates supported holes with no smoothing of measured vertices.'} Doorways and depth-confirmed free space stay open.</p>
              <label class="toggle"><input type="checkbox" checked={project.settings.fillInferredMeshHoles} on:change={(event) => updateSetting('fillInferredMeshHoles', inputChecked(event))} disabled={processing}/><span></span><div><strong>Fill inferred holes</strong><small>More complete, but may fill boundaries without direct depth support</small></div></label>
              <label class="toggle"><input type="checkbox" checked={project.settings.produceWatertightMesh} on:change={(event) => updateSetting('produceWatertightMesh', inputChecked(event))} disabled={processing}/><span></span><div><strong>Also produce watertight copy</strong><small>Separate PLY for fabrication; intentional openings may be sealed</small></div></label>
            {/if}
            {#if artifactReady('texturedMesh') && project.meshRepairReportPath}
              <div class="repair-result" class:warning={project.meshRepairFallback}>
                <span>{project.meshRepairFallback ? 'UNREPAIRED FALLBACK' : `${project.meshRepairProfile?.toUpperCase()} COMPLETE`}</span>
                <div><strong>{project.meshRepairHolesFilled ?? 0}</strong><small>holes filled</small></div>
                <div><strong>{project.meshRepairOpeningsPreserved ?? 0}</strong><small>openings preserved</small></div>
                <div><strong>{project.meshRepairUnknownPreserved ?? 0}</strong><small>unknown preserved</small></div>
                <div><strong>{project.meshRepairDefectsFixed ?? 0}</strong><small>defects fixed</small></div>
                <small class="report-path" title={project.meshRepairReportPath}>{project.meshRepairReportPath}</small>
              </div>
              {/if}
            </div>
          </details>
        {/if}

        {#if mediaSourceCount > 0}
          <details class="panel collapsible-panel" open>
            <summary><span>MEDIA SOURCES</span><strong>{mediaSourceCount} IMPORTED</strong></summary>
            <div class="collapsible-body pipeline-note">
              <p>{mediaOnlyProject ? 'LingBot-Map supplies dense depth and a continuous trajectory; COLMAP validates and refines it before high-resolution Gaussian training.' : 'High-resolution frames are localized against metric RGB-D landmarks, then enhance point colors, mesh textures, and splat appearance.'}</p>
              <div class="media-source-list">
                {#each project.mediaSources as source (source.id)}
                  <article class="media-source">
                    <span class="media-kind">{source.kind.toUpperCase()}</span>
                    <div class="media-source-copy">
                      <strong title={source.name}>{source.name}</strong>
                      <small>{formatByteSize(source.byteSize)} · imported {formatProjectDate(source.createdAt)}</small>
                    </div>
                    <button on:click={() => removeImportedMediaSource(source)} disabled={busy || processing || capturing} aria-label={`Remove ${source.name}`}>Remove</button>
                  </article>
                {/each}
              </div>
              <button class="ghost full" on:click={addMediaSource} disabled={busy || processing}>Add more photos or video…</button>
            </div>
          </details>
        {/if}

        {#if !mediaOnlyProject}
          <details class="panel collapsible-panel" open>
            <summary><span>TEXTURE PHOTOS</span><strong>{localizedTexturePhotoCount} READY</strong></summary>
            <div class="collapsible-body pipeline-note">
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
            </div>
          </details>
        {/if}

        <details class="panel collapsible-panel" open>
          <summary><span>REBUILD START</span><strong>{rebuildRgbdPreparation ? 'RGB-D SOURCE' : mediaRestartStage === 'decode' ? 'MEDIA DECODE' : mediaRestartStage === 'analysis' ? 'MEDIA ANALYSIS' : 'REUSE CACHE'}</strong></summary>
          <div class="collapsible-body settings rebuild-policy">
            {#if mediaSourceCount > 0}
              <label>Start media preparation from
                <select bind:value={mediaRestartStage} disabled={processing}>
                  <option value="reuse">Cached analysis · fastest</option>
                  <option value="analysis">Camera analysis · keep decoded frames</option>
                  <option value="decode">Media decode · discard prepared frames</option>
                </select>
              </label>
            {/if}
            {#if completedCaptures > 0}
              <label class="toggle"><input type="checkbox" bind:checked={rebuildRgbdPreparation} disabled={processing}/><span></span><div><strong>Re-run RGB-D tracking and fusion</strong><small>Discard cached poses and geometry, while keeping decoded media</small></div></label>
            {/if}
            <p class="cache-policy-note">Later-stage data is discarded automatically from the selected start point. Changed sources or settings still invalidate incompatible cached data.</p>
          </div>
        </details>

        {#if activeJob}
          <details class="panel collapsible-panel job-card" class:error={activeJob.status === 'failed'} open>
            <summary><span>{activeJob.status.toUpperCase()}</span><strong>{Math.round(activeJob.progress * 100)}%</strong></summary>
            <div class="collapsible-body">
            <h3>{activeJob.stage.replaceAll('_', ' ')}</h3>
            <p>{activeJob.error ?? activeJob.detail}</p>
            <div class="progress"><i style={`width:${Math.round(activeJob.progress * 100)}%`}></i></div>
            {#if activeJob.stageProgress != null}
              <div class="stage-progress-meta"><span>Current stage</span><strong>{Math.round(activeJob.stageProgress * 100)}%</strong></div>
              <div class="progress stage-progress"><i style={`width:${Math.round(activeJob.stageProgress * 100)}%`}></i></div>
            {/if}
            {#if activeJob.iteration !== null}
              <div class="job-quality">
                <span>Iteration <strong>{activeJob.iteration.toLocaleString()} / {activeJob.totalIterations?.toLocaleString()}</strong></span>
                <span>Current <strong>{activeJob.loss?.toFixed(4) ?? '—'}</strong></span>
                <span>Rolling <strong>{activeJob.smoothedLoss?.toFixed(4) ?? 'warming up'}</strong></span>
              </div>
            {/if}
            <div class="job-meta"><span>{activeJob.computeBackend ?? 'Waiting for worker'}</span><span>{activeJob.stageEtaSeconds ? `stage ~${formatDuration(activeJob.stageEtaSeconds)}` : activeJob.etaSeconds ? `~${formatDuration(activeJob.etaSeconds)}` : ''}</span></div>
            {#if processing}
              <button class="ghost full" on:click={cancelBuild}>Cancel safely</button>
            {:else if activeJob.resumable && ['failed', 'cancelled'].includes(activeJob.status)}
              <div class="button-row"><button class="primary" on:click={() => startBuild(true)}>Resume checkpoint</button><button class="ghost" on:click={discardBuild}>Discard</button></div>
              {/if}
            </div>
          </details>
        {/if}

        <button class="primary full build-button" on:click={() => startBuild(false)} disabled={busy || processing || photoLocalizationActive || (completedCaptures === 0 && mediaSourceCount === 0) || (!buildPointCloud && !buildTexturedMesh && !buildGaussianSplat)}>{processing ? 'Reconstruction running…' : photoLocalizationActive ? 'Localizing texture photos…' : readyArtifacts ? 'Rebuild selected outputs' : mediaOnlyProject ? 'Solve cameras & build AAA splat' : mediaSourceCount > 0 ? 'Build hybrid high-quality outputs' : 'Build selected outputs'}</button>

      {:else}
        <header class="inspector-heading"><div><span>RESULT</span><h2>Edit & export</h2></div><strong class="take-total">{readyArtifacts} ready</strong></header>
        <details class="panel collapsible-panel" open>
          <summary><span>REPRESENTATION</span><strong>{renderMode === 'points' ? 'POINTS' : renderMode === 'mesh' ? 'MESH' : mediaOnlyProject ? '3DGS' : '2DGS'}</strong></summary>
          <div class="collapsible-body view-switcher">
            <button class:active={renderMode === 'points'} disabled={!artifactReady('pointCloud')} on:click={() => loadResult('points')}><span>P</span><div><strong>Points</strong><small>{formatCount(project.pointCount)}</small></div></button>
            <button class:active={renderMode === 'mesh'} disabled={!artifactReady('texturedMesh')} on:click={() => loadResult('mesh')}><span>M</span><div><strong>Mesh</strong><small>{formatCount(project.meshTriangleCount)} tris</small></div></button>
            <button class:active={renderMode === 'splat'} disabled={!artifactReady('gaussianSplat')} on:click={() => loadResult('splat')}><span>G</span><div><strong>{mediaOnlyProject ? '3DGS' : '2DGS'}</strong><small>{mediaOnlyProject ? 'Photoreal volume' : 'Metric surface'}</small></div></button>
          </div>
        </details>
        {#if renderMode === 'mesh'}
          <details class="panel collapsible-panel" open>
            <summary><span>DISPLAY</span><strong>{meshViewMode.replaceAll('-', ' ').toUpperCase()}</strong></summary>
            <div class="collapsible-body settings"><label>Mesh display<select bind:value={meshViewMode}><option value="surface">Textured</option><option value="surface-wireframe">Texture + wire</option><option value="wireframe">Wireframe</option><option value="shaded">Shaded</option></select></label></div>
          </details>
        {/if}
        <details class="panel collapsible-panel edit-tools" open>
          <summary><span>MODEL POSE</span><strong class:edited={hasEditPose}>{hasEditPose ? 'EDITED' : 'ORIGINAL'}</strong></summary>
          <div class="collapsible-body edit-tools-body">
            <div class="edit-actions">
              <button class:active={floorPickMode} disabled={!canEditModel} on:click={() => { floorPickMode = !floorPickMode; editMode = false; clipEditMode = false; }}>{floorPickMode ? 'Cancel floor pick' : 'Pick floor'}</button>
              <button class:active={editMode} disabled={!canEditModel} on:click={() => { editMode = !editMode; floorPickMode = false; clipEditMode = false; }}>{editMode ? 'Close gizmo' : 'Transform gizmo'}</button>
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
          </div>
        </details>
        <details class="panel collapsible-panel edit-tools clip-tools" open>
          <summary><span>BOUNDING BOX</span><strong class:edited={clippingEnabled}>{clippingEnabled ? 'CLIPPING' : 'OFF'}</strong></summary>
          <div class="collapsible-body edit-tools-body">
            <div class="edit-actions">
              <button class:active={clippingEnabled} disabled={!canClipModel && !clipBounds} on:click={() => setClippingEnabled(!clippingEnabled)}>{clippingEnabled ? 'Disable clipping' : 'Enable clipping'}</button>
              <button class:active={clipEditMode} disabled={!clippingEnabled || !clipBounds} on:click={() => { clipEditMode = !clipEditMode; editMode = false; floorPickMode = false; }}>{clipEditMode ? 'Close box gizmo' : 'Edit box'}</button>
            </div>
            <button class="reset-pose" disabled={!canClipModel} on:click={fitBoundingBox}>Fit to transformed result</button>
            {#if clippingEnabled && clipBounds}
              {#if clipEditMode}
                <div class="gizmo-modes">
                  <button class:active={clipGizmoMode === 'translate'} on:click={() => setClipGizmoMode('translate')}>Move <kbd>W</kbd></button>
                  <button class:active={clipGizmoMode === 'scale'} on:click={() => setClipGizmoMode('scale')}>Resize <kbd>R</kbd></button>
                </div>
              {/if}
              <div class="clip-grid">
                <span></span><small>MIN</small><small>MAX</small>
                <b>X</b><input aria-label="Minimum X" type="number" step="0.01" value={clipBounds.min[0]} on:change={(event) => setClipCoordinate('min', 0, inputValue(event))}><input aria-label="Maximum X" type="number" step="0.01" value={clipBounds.max[0]} on:change={(event) => setClipCoordinate('max', 0, inputValue(event))}>
                <b>Y</b><input aria-label="Minimum Y" type="number" step="0.01" value={clipBounds.min[1]} on:change={(event) => setClipCoordinate('min', 1, inputValue(event))}><input aria-label="Maximum Y" type="number" step="0.01" value={clipBounds.max[1]} on:change={(event) => setClipCoordinate('max', 1, inputValue(event))}>
                <b>Z</b><input aria-label="Minimum Z" type="number" step="0.01" value={clipBounds.min[2]} on:change={(event) => setClipCoordinate('min', 2, inputValue(event))}><input aria-label="Maximum Z" type="number" step="0.01" value={clipBounds.max[2]} on:change={(event) => setClipCoordinate('max', 2, inputValue(event))}>
              </div>
            {/if}
          </div>
        </details>
        <details class="panel collapsible-panel" open>
          <summary><span>OUTPUT STATS</span><strong>{project.confidenceLabel ?? '—'}</strong></summary>
          <div class="collapsible-body result-stats">
            <div><span>Points</span><strong>{formatCount(project.pointCount)}</strong></div>
            <div><span>Triangles</span><strong>{formatCount(project.meshTriangleCount)}</strong></div>
            <div><span>Frames used</span><strong>{project.framesUsed ?? '—'}</strong></div>
            <div><span>Confidence</span><strong>{project.confidenceLabel ?? '—'}</strong></div>
          </div>
        </details>
        <details class="panel collapsible-panel" open>
          <summary><span>EXPORT</span><strong>{readyArtifacts} AVAILABLE</strong></summary>
          <div class="collapsible-body export-list">
            <button class:exporting={exporting === 'points'} aria-busy={exporting === 'points'} on:click={exportPointCloud} disabled={Boolean(exporting) || !artifactReady('pointCloud')}><span>P</span><div><strong>Point cloud PLY</strong><small>Metric colored vertices</small></div><i>{exporting === 'points' ? 'Exporting…' : 'Export…'}</i></button>
            <button class:exporting={exporting === 'mesh'} aria-busy={exporting === 'mesh'} on:click={exportMesh} disabled={Boolean(exporting) || !artifactReady('texturedMesh')}><span>M</span><div><strong>Textured OBJ bundle</strong><small>OBJ + MTL + PNG</small></div><i>{exporting === 'mesh' ? 'Exporting…' : 'Export…'}</i></button>
            <button class:exporting={exporting === 'splat'} aria-busy={exporting === 'splat'} on:click={exportSplat} disabled={Boolean(exporting) || !artifactReady('gaussianSplat')}><span>G</span><div><strong>{mediaOnlyProject ? '3D Gaussian PLY' : '2D Gaussian PLY'}</strong><small>{mediaOnlyProject ? 'Photoreal splat + sidecars' : 'Aligned metric splat + sidecars'}</small></div><i>{exporting === 'splat' ? 'Exporting…' : 'Export…'}</i></button>
            {#if exporting}
              <div class="export-feedback" role="status" aria-live="polite">
                <div><strong>{exporting === 'points' ? 'Exporting point cloud' : exporting === 'mesh' ? 'Exporting textured mesh' : `Exporting ${mediaOnlyProject ? '3D Gaussian splat' : '2D Gaussian surface'}`}</strong><small>Applying the saved pose{clippingEnabled ? ' and clipping bounds' : ''}, then writing to disk.</small></div>
                <span class="export-progress"><i></i></span>
              </div>
            {/if}
          </div>
        </details>
      {/if}
    </aside>
  </main>

  <footer class:error={Boolean(fatalError)}>
    <span class="status-dot" class:busy={busy || capturing || processing || photoLocalizationActive || Boolean(exporting)}></span>
    <strong>{capturing ? 'LIVE' : processing ? 'BUILDING' : photoLocalizationActive ? 'LOCALIZING' : exporting ? 'EXPORTING' : fatalError ? 'ERROR' : 'READY'}</strong>
    <p>{message}</p>
    {#if sensor?.imuActive}<span class="footer-metric">IMU {sensor.imuRateHz.toFixed(0)} Hz</span>{/if}
    {#if sensor?.liveReconstructionBackend}<span class="footer-metric">{sensor.liveReconstructionBackend}</span>{/if}
  </footer>
</div>

<style>
  :global(*) { box-sizing: border-box; }
  :global(html, body, #app) { width: 100%; height: 100%; margin: 0; overflow: hidden; }
  :global(body) { background: #101216; color: #d7dbe1; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  :global(button), :global(input), :global(select) { font: inherit; }
  :global(button) { color: inherit; }

  .app-shell { --panel: #191c21; --panel-soft: #1e2228; --line: #2b3038; --muted: #8a929d; --cyan: #6c9eff; --mint: #54b78d; --amber: #d2a04f; display: grid; grid-template-rows: 64px 58px minmax(0, 1fr) 34px; gap: 8px; width: 100%; height: 100%; padding: 8px; background: #101216; }
  .topbar { display: grid; grid-template-columns: minmax(230px, .8fr) minmax(220px, 1fr) auto auto; align-items: center; gap: 24px; padding: 0 16px; border: 1px solid var(--line); border-radius: 6px; background: #15181d; }
  .brand, .project-title, .runtime-state, .tracking-title, .job-meta, .button-row { display: flex; align-items: center; }
  .brand { gap: 11px; }
  .brand-mark { display: grid; place-items: center; width: 34px; height: 34px; border: 1px solid #3a4658; border-radius: 5px; background: #20252c; color: var(--cyan); font-size: 12px; font-weight: 850; letter-spacing: .06em; }
  .brand div, .project-title { display: grid; gap: 2px; }
  .brand strong { font-size: 16px; letter-spacing: .01em; }
  .brand small, .project-title span { color: var(--muted); font-size: 10px; letter-spacing: .08em; text-transform: uppercase; }
  .project-title { justify-items: start; min-width: 0; padding: 8px 10px; border-radius: 4px; background: transparent; text-align: left; }
  .project-title:hover:not(:disabled) { background: #20242a; }
  .project-title strong { max-width: 360px; overflow: hidden; font-size: 13px; text-overflow: ellipsis; white-space: nowrap; }
  .header-actions { display: flex; gap: 7px; }
  .runtime-state { gap: 8px; }
  .runtime-state span { display: flex; align-items: center; gap: 6px; padding: 7px 9px; border: 1px solid var(--line); border-radius: 4px; color: #777f8a; font-size: 10px; font-weight: 750; letter-spacing: .04em; text-transform: uppercase; }
  .runtime-state span i { width: 6px; height: 6px; border-radius: 50%; background: #53636b; }
  .runtime-state span.ready { color: #b2b8c1; }
  .runtime-state span.ready i { background: var(--mint); }
  button { border: 0; cursor: pointer; }
  button:disabled, input:disabled, select:disabled { cursor: not-allowed; opacity: .43; }
  .ghost { padding: 10px 13px; border: 1px solid var(--line); border-radius: 5px; background: #1b1f24; color: #b6bbc3; font-size: 12px; font-weight: 700; }
  .ghost:hover:not(:disabled) { border-color: #46536a; background: #22272e; }
  .ghost.compact { white-space: nowrap; }
  .ghost.full, .primary.full { width: 100%; }
  .primary { padding: 11px 15px; border-radius: 5px; background: #4f82e8; color: #f7f9fc; font-size: 12px; font-weight: 800; }
  .primary:hover:not(:disabled) { background: #5b8df0; }

  .modal-backdrop { position: fixed; z-index: 100; inset: 0; display: grid; place-items: center; padding: 24px; background: rgba(5, 6, 8, .82); }
  .modal-dismiss { position: absolute; inset: 0; width: 100%; height: 100%; background: transparent; cursor: default; }
  .project-manager { position: relative; z-index: 1; display: grid; grid-template-rows: auto auto auto minmax(150px, 1fr) auto auto; gap: 14px; width: min(780px, calc(100vw - 48px)); max-height: min(760px, calc(100vh - 48px)); padding: 20px; overflow: hidden; border: 1px solid #343a44; border-radius: 8px; background: #191d22; box-shadow: 0 24px 70px rgba(0,0,0,.45); }
  .project-manager > header { display: flex; align-items: center; justify-content: space-between; }
  .project-manager > header div { display: grid; gap: 4px; }
  .project-manager > header span, .current-project-editor span, .new-project-form span { color: var(--cyan); font-size: 9px; font-weight: 850; letter-spacing: .11em; }
  .dialog-close { display: grid; place-items: center; width: 34px; height: 34px; border: 1px solid var(--line); border-radius: 5px; background: #20242a; color: #a0a7b0; font-size: 19px; }
  .current-project-editor { display: grid; grid-template-columns: minmax(180px, .8fr) minmax(260px, 1.2fr); align-items: end; gap: 16px; padding: 13px; border: 1px solid #343b47; border-radius: 6px; background: #1d2229; }
  .current-project-editor > div, .new-project-form > div { display: grid; gap: 4px; }
  .current-project-editor small, .new-project-form small { color: #758d98; font-size: 9px; line-height: 1.4; }
  .current-project-editor form { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; }
  .current-project-editor .primary { height: 36px; padding-top: 0; padding-bottom: 0; }
  .project-library-heading { display: flex; justify-content: space-between; color: #8da4ae; font-size: 10px; }
  .project-library-heading span { color: #617985; }
  .project-library { min-height: 150px; overflow-y: auto; border: 1px solid var(--line); border-radius: 6px; background: #14171b; scrollbar-color: #343a43 transparent; }
  .project-library.loading { opacity: .7; }
  .project-library article { display: grid; grid-template-columns: 40px minmax(0, 1fr) auto; align-items: center; gap: 12px; padding: 13px; border-bottom: 1px solid var(--line); }
  .project-library article:last-child { border-bottom: 0; }
  .project-library article.active { background: #1d2525; }
  .project-library-icon { display: grid; place-items: center; width: 40px; height: 40px; border: 1px solid #353c48; border-radius: 5px; background: #20252c; color: var(--cyan); font-size: 10px; font-weight: 850; }
  .project-library-copy { min-width: 0; }
  .project-library-copy > div { display: flex; align-items: center; gap: 8px; }
  .project-library-copy strong { overflow: hidden; font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
  .project-library-copy span { padding: 2px 5px; border-radius: 4px; background: rgba(98,214,186,.12); color: var(--mint); font-size: 7px; font-weight: 850; letter-spacing: .08em; }
  .project-library-copy small, .project-library-copy p { color: #657d88; font-size: 8px; }
  .project-library-copy p { margin-top: 4px; }
  .project-library-actions { display: flex; gap: 6px; }
  .project-library-actions button { padding: 7px 10px; font-size: 9px; }
  .danger { border: 1px solid #4b3434; border-radius: 5px; background: #271d1e; color: #d38a81; font-size: 9px; font-weight: 750; }
  .danger:hover:not(:disabled) { border-color: #704747; background: #322223; }
  .project-library-empty { display: grid; place-items: center; align-content: center; min-height: 150px; color: #718994; font-size: 10px; }
  .project-library-empty .spinner { width: 18px; height: 18px; margin-bottom: 8px; }
  .new-project-form { display: grid; grid-template-columns: minmax(150px, .7fr) minmax(180px, 1fr) auto auto; align-items: end; gap: 8px; padding-top: 2px; }
  .new-project-form .primary, .new-project-form .ghost { height: 36px; padding-top: 0; padding-bottom: 0; }
  .new-project-button { width: 100%; min-height: 40px; border: 1px dashed #3a465a; border-radius: 5px; background: #1b2026; color: var(--cyan); font-size: 10px; font-weight: 800; }
  .new-project-button:hover:not(:disabled) { border-color: #53688b; background: #202630; }
  .project-manager-error { color: #df9388; font-size: 10px; }

  .workflow { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1px; overflow: hidden; padding: 0 12px; border: 1px solid var(--line); border-radius: 6px; background: #13161a; }
  .workflow button { position: relative; display: flex; align-items: center; gap: 12px; padding: 0 18px; background: transparent; color: #708792; text-align: left; }
  .workflow button::after { position: absolute; right: 0; bottom: -1px; left: 0; height: 2px; background: transparent; content: ''; }
  .workflow button.active { color: #e2e5e9; background: #191d23; }
  .workflow button.active::after { background: var(--cyan); }
  .workflow button > span { color: #49606b; font-family: ui-monospace, monospace; font-size: 11px; font-weight: 800; }
  .workflow button.done > span { color: var(--mint); }
  .workflow button div { display: grid; gap: 3px; }
  .workflow button strong { font-size: 12px; }
  .workflow button small { color: #607783; font-size: 10px; }

  main { display: grid; grid-template-columns: minmax(0, 1fr) 390px; gap: 8px; min-height: 0; }
  .viewport { position: relative; min-width: 0; min-height: 0; overflow: hidden; border: 1px solid var(--line); border-radius: 6px; background: #0b0d10; }
  .viewport :global(.viewer) { border: 0; border-radius: 0; }
  aside { min-height: 0; padding: 0 3px 0 0; overflow-x: hidden; overflow-y: auto; background: transparent; scrollbar-color: #343a43 transparent; }
  .panel { margin-bottom: 8px; padding: 14px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); }
  .workspace-heading, .inspector-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; min-height: 58px; margin-bottom: 8px; padding: 10px 13px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); }
  .workspace-heading > div, .inspector-heading > div { display: grid; gap: 4px; }
  .workspace-heading span, .inspector-heading span { color: var(--cyan); font-size: 9px; font-weight: 850; letter-spacing: .11em; }
  h2, h3, p { margin: 0; }
  h2 { font-size: 18px; letter-spacing: -.02em; }
  h3 { margin: 8px 0 4px; font-size: 14px; text-transform: capitalize; }
  .icon-button { display: grid; place-items: center; width: 34px; height: 34px; border: 1px solid var(--line); border-radius: 5px; background: #1b1f24; color: var(--cyan); font-size: 18px; }

  .settings { display: grid; gap: 13px; }
  .settings label { display: grid; gap: 6px; color: #8ba2ad; font-size: 10px; font-weight: 720; letter-spacing: .03em; }
  .advanced-settings, .collapsible-panel { padding: 0; overflow: hidden; }
  .advanced-settings summary, .collapsible-panel > summary { display: flex; align-items: center; justify-content: space-between; min-height: 42px; padding: 0 13px; cursor: pointer; list-style: none; }
  .advanced-settings summary::-webkit-details-marker, .collapsible-panel > summary::-webkit-details-marker { display: none; }
  .advanced-settings summary::after, .collapsible-panel > summary::after { width: 12px; margin-left: 9px; color: #747c87; font-size: 12px; content: '›'; transform: rotate(90deg); transition: transform .15s; }
  .advanced-settings:not([open]) summary::after, .collapsible-panel:not([open]) > summary::after { transform: rotate(0deg); }
  .advanced-settings summary:hover, .collapsible-panel > summary:hover { background: #1e2228; }
  .advanced-settings summary span, .collapsible-panel > summary span { color: var(--cyan); font-size: 9px; font-weight: 850; letter-spacing: .11em; }
  .advanced-settings summary strong, .collapsible-panel > summary strong { margin-left: auto; color: #8c949f; font-size: 9px; }
  .collapsible-panel > summary strong.edited { color: var(--amber); }
  .collapsible-panel > summary strong.good { color: var(--mint); }
  .collapsible-body { padding: 13px; border-top: 1px solid var(--line); }
  .startup-panel, .status-body { display: grid; gap: 10px; }
  .advanced-settings > p { padding: 0 15px 15px; color: #708792; font-size: 10px; line-height: 1.55; }
  .advanced-body { display: grid; gap: 13px; padding: 14px 13px 13px; border-top: 1px solid var(--line); }
  .advanced-body > label, .advanced-body .setting-grid label { display: grid; gap: 6px; color: #8ba2ad; font-size: 10px; font-weight: 720; letter-spacing: .03em; }
  .advanced-body > p { color: #708792; font-size: 10px; line-height: 1.55; }
  .compact-settings { padding: 10px; border: 1px solid var(--line); border-radius: 5px; background: #16191e; }
  .slider-control > span { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .slider-control output { color: #c9dde5; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 9px; font-weight: 800; letter-spacing: 0; }
  .slider-control input[type='range'] { height: 18px; padding: 0; border: 0; border-radius: 0; background: transparent; box-shadow: none; accent-color: var(--cyan); cursor: ew-resize; }
  .setting-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 10px; }
  select, input { width: 100%; min-width: 0; height: 36px; padding: 0 10px; outline: none; border: 1px solid #343a43; border-radius: 5px; background: #14171b; color: #d2d6dc; font-size: 11px; }
  select:focus, input:focus { border-color: #5d82c8; box-shadow: 0 0 0 2px rgba(108,158,255,.12); }
  .unit-input { position: relative; }
  .unit-input input { padding-right: 35px; }
  .unit-input span { position: absolute; top: 50%; right: 10px; color: #637b86; transform: translateY(-50%); }
  .toggle { grid-template-columns: auto auto 1fr; align-items: center; cursor: pointer; }
  .toggle input { position: absolute; width: 1px; height: 1px; opacity: 0; }
  .toggle > span { position: relative; width: 34px; height: 19px; border-radius: 10px; background: #343a43; transition: .2s; }
  .toggle > span::after { position: absolute; top: 3px; left: 3px; width: 13px; height: 13px; border-radius: 50%; background: #8499a3; transition: .2s; content: ''; }
  .toggle input:checked + span { background: #315f4d; }
  .toggle input:checked + span::after { left: 18px; background: var(--mint); }
  .toggle div { display: grid; gap: 2px; }
  .toggle strong { color: #bfd1d8; font-size: 11px; }
  .toggle small { color: #657c87; font-size: 9px; font-weight: 500; }
  .rebuild-policy .cache-policy-note { color: #718894; font-size: 9px; line-height: 1.5; }

  .capture-button { display: flex; align-items: center; justify-content: center; gap: 10px; width: 100%; height: 48px; margin-bottom: 9px; border-radius: 5px; background: #3ca178; color: #07140f; font-size: 13px; font-weight: 900; }
  .capture-button i { width: 11px; height: 11px; border: 2px solid currentColor; border-radius: 50%; }
  .capture-button.stop { background: #cf685b; color: #1b0907; }
  .capture-button.stop i { border-radius: 2px; background: currentColor; }
  .tracking-card.warning { border-color: rgba(239,179,102,.3); }
  .connection-card.connected { border-color: rgba(98,214,186,.28); }
  .connection-card.warning { border-color: rgba(239,179,102,.24); }
  .tracking-title { gap: 10px; }
  .tracking-title > i { width: 8px; height: 8px; border-radius: 50%; background: var(--mint); }
  .tracking-card.warning .tracking-title > i { background: var(--amber); }
  .connection-card.warning .tracking-title > i { background: var(--amber); }
  .tracking-title div { display: grid; gap: 3px; }
  .tracking-title strong { font-size: 11px; }
  .tracking-title small { color: var(--muted); font-size: 9px; }
  .status-body > p, .empty-copy, .pipeline-note p, .job-card p { color: #79818c; font-size: 10px; line-height: 1.55; }
  .mini-grid, .result-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; margin-top: 12px; overflow: hidden; border: 1px solid var(--line); border-radius: 5px; background: var(--line); }
  .mini-grid div, .result-stats > div { display: grid; gap: 3px; padding: 9px; background: #171a1f; }
  .mini-grid span, .result-stats span, .pipeline-note div span { color: #617985; font-size: 9px; }
  .mini-grid strong, .result-stats strong, .pipeline-note div strong { font-size: 11px; }

  .take-total { color: #8199a5; font-size: 10px; }
  .takes article { display: grid; grid-template-columns: 28px 1fr auto; align-items: center; gap: 9px; padding: 11px 0; border-bottom: 1px solid var(--line); }
  .takes article:last-child { padding-bottom: 0; border-bottom: 0; }
  .take-number { color: #4d6570; font-family: ui-monospace, monospace; font-size: 10px; }
  .takes article div { display: grid; gap: 3px; }
  .takes article strong { font-size: 11px; }
  .takes article small { color: #6d8590; font-size: 9px; }
  .takes article button { padding: 5px 7px; background: transparent; color: #997b7a; font-size: 9px; }

  .target-list { display: grid; gap: 8px; }
  .target-list > label { display: grid; grid-template-columns: auto 34px 1fr auto; align-items: center; gap: 9px; min-height: 58px; padding: 9px; border: 1px solid var(--line); border-radius: 5px; background: #171a1f; cursor: pointer; }
  .target-list > label.active { border-color: #465d88; background: #1c222c; }
  .target-list > label > input[type=checkbox] { width: 14px; height: 14px; accent-color: var(--cyan); }
  .target-icon, .view-switcher button > span, .export-list button > span { display: grid; place-items: center; width: 32px; height: 32px; border-radius: 4px; background: #242a34; color: var(--cyan); font-size: 10px; font-weight: 900; }
  .target-list label div { display: grid; gap: 3px; }
  .target-list label div strong { font-size: 11px; }
  .target-list label div small { color: #687f8a; font-size: 9px; }
  .target-list label > i { max-width: 80px; color: var(--mint); font-size: 8px; font-style: normal; font-weight: 800; text-align: right; }
  .target-list .iterations { grid-template-columns: auto 1fr auto; min-height: auto; }
  .iterations input { height: 18px; padding: 0; accent-color: var(--cyan); }
  .mesh-repair-settings > p { margin: -3px 0 1px; color: #718894; font-size: 9px; line-height: 1.55; }
  .repair-result { display: grid; grid-template-columns: 1fr repeat(4, auto); align-items: center; gap: 11px; padding: 10px; border: 1px solid #345345; border-radius: 5px; background: #19231f; }
  .repair-result.warning { border-color: #58482f; background: #241f18; }
  .repair-result > span { color: var(--mint); font-size: 8px; font-weight: 850; }
  .repair-result.warning > span { color: var(--amber); }
  .repair-result > div { display: grid; gap: 2px; text-align: right; }
  .repair-result strong { color: #bdd0d7; font-size: 11px; }
  .repair-result small { color: #647c87; font-size: 7px; }
  .repair-result .report-path { grid-column: 1 / -1; overflow: hidden; color: #58717d; font-family: ui-monospace, monospace; text-overflow: ellipsis; white-space: nowrap; }
  .pipeline-note div { display: flex; justify-content: space-between; padding-top: 9px; }
  .pipeline-note .media-source-list { display: grid; max-height: 310px; margin-top: 10px; padding: 0; overflow-y: auto; border-top: 1px solid var(--line); scrollbar-color: #263d49 transparent; }
  .pipeline-note .media-source { display: grid; grid-template-columns: 42px minmax(0, 1fr) auto; align-items: center; gap: 8px; min-width: 0; padding: 9px 0; border-bottom: 1px solid var(--line); }
  .pipeline-note .media-source .media-kind { padding: 4px 3px; border-radius: 5px; background: rgba(98,214,186,.1); color: var(--mint); font-size: 7px; font-weight: 850; text-align: center; }
  .pipeline-note .media-source .media-source-copy { display: grid; min-width: 0; gap: 3px; padding: 0; }
  .pipeline-note .media-source-copy strong, .pipeline-note .media-source-copy small { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .pipeline-note .media-source-copy strong { font-size: 10px; }
  .pipeline-note .media-source-copy small { color: #657d88; font-size: 8px; }
  .pipeline-note .media-source > button { padding: 4px 2px; background: transparent; color: #987b7b; font-size: 8px; }
  .pipeline-note .texture-progress { display: grid; gap: 8px; margin-top: 12px; padding: 10px; border: 1px solid #35445b; border-radius: 5px; background: #1a2029; }
  .pipeline-note .texture-progress.error { border-color: #5a3837; background: #251c1d; }
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
  .progress { height: 4px; overflow: hidden; border-radius: 2px; background: #30353d; }
  .progress i { display: block; height: 100%; border-radius: inherit; background: var(--cyan); transition: width .25s linear; }
  .job-card .progress { margin: 11px 0 8px; }
  .stage-progress-meta { display: flex; justify-content: space-between; gap: 8px; margin-top: 8px; color: #687f8a; font-size: 8px; text-transform: uppercase; letter-spacing: .05em; }
  .job-card .progress.stage-progress, .job-overlay .progress.stage-progress { height: 3px; margin: 4px 0 8px; }
  .progress.stage-progress i { background: #8a76e8; }
  .job-quality { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 6px 12px; margin: 9px 0; color: #687f8a; font-size: 8px; }
  .job-quality strong { margin-left: 3px; color: #a9bec8; font-size: 9px; }
  .job-meta { justify-content: space-between; gap: 8px; margin-bottom: 11px; color: #687f8a; font-size: 9px; }
  .button-row { gap: 8px; }
  .button-row button { flex: 1; }
  .build-button { height: 47px; }

  .view-switcher { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; }
  .view-switcher button { display: grid; justify-items: center; gap: 7px; padding: 10px 4px; border: 1px solid var(--line); border-radius: 5px; background: #171a1f; }
  .view-switcher button.active { border-color: #506b9d; background: #202736; }
  .view-switcher button div { display: grid; gap: 2px; text-align: center; }
  .view-switcher button strong { font-size: 10px; }
  .view-switcher button small { color: #687f8a; font-size: 8px; }
  .edit-tools { display: block; }
  .edit-tools-body { display: grid; gap: 9px; }
  .edit-actions, .gizmo-modes { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
  .gizmo-modes { grid-template-columns: repeat(3, 1fr); }
  .clip-tools .gizmo-modes { grid-template-columns: repeat(2, 1fr); }
  .edit-options { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; padding-top: 2px; }
  .edit-options label { display: grid; gap: 5px; color: #78909d; font-size: 9px; font-weight: 700; }
  .edit-options select { height: 32px; padding: 0 8px; font-size: 9px; }
  .edit-tools button { min-height: 35px; padding: 7px 8px; border: 1px solid var(--line); border-radius: 5px; background: #171a1f; color: #adb3bc; font-size: 9px; font-weight: 750; }
  .edit-tools button:hover:not(:disabled), .edit-tools button.active { border-color: #506b9d; background: #202736; color: #e0e4ea; }
  .edit-tools .save-pose.dirty { border-color: #416c59; background: #1c2a24; color: var(--mint); }
  .edit-tools kbd { margin-left: 3px; padding: 1px 4px; border: 1px solid rgba(155,199,215,.16); border-radius: 4px; color: var(--cyan); font-family: ui-monospace, monospace; font-size: 8px; }
  .edit-tools .reset-pose { min-height: 30px; color: #7f97a2; }
  .clip-grid { display: grid; grid-template-columns: 18px 1fr 1fr; align-items: center; gap: 5px; }
  .clip-grid small { color: #617985; font-size: 7px; font-weight: 800; letter-spacing: .08em; text-align: center; }
  .clip-grid b { color: var(--amber); font-size: 9px; text-align: center; }
  .clip-grid input { width: 100%; min-width: 0; padding: 6px; border: 1px solid var(--line); border-radius: 4px; background: #14171b; color: #c8cdd4; font-size: 9px; font-variant-numeric: tabular-nums; }
  .collapsible-body.result-stats { margin-top: 0; border: 0; border-top: 1px solid var(--line); border-radius: 0; }
  .export-list { display: grid; gap: 7px; }
  .export-list button { display: grid; grid-template-columns: 34px 1fr auto; align-items: center; gap: 9px; padding: 10px; border: 1px solid var(--line); border-radius: 5px; background: #171a1f; text-align: left; }
  .export-list button:hover:not(:disabled) { border-color: rgba(99,199,231,.36); }
  .export-list button.exporting { border-color: rgba(99,199,231,.36); background: rgba(99,199,231,.07); }
  .export-list button div { display: grid; gap: 3px; }
  .export-list button strong { font-size: 10px; }
  .export-list button small { color: #687f8a; font-size: 8px; }
  .export-list button > i { color: var(--cyan); font-size: 9px; font-style: normal; }
  .export-feedback { display: grid; gap: 8px; padding: 9px 10px; border: 1px solid #35445b; border-radius: 5px; background: #1a2029; }
  .export-feedback > div { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
  .export-feedback strong { color: #a9c3cd; font-size: 9px; }
  .export-feedback small { color: #687f8a; font-size: 8px; text-align: right; }
  .export-progress { display: block; height: 3px; overflow: hidden; border-radius: 4px; background: #172a35; }
  .export-progress i { display: block; width: 35%; height: 100%; border-radius: inherit; background: var(--cyan); animation: export-slide 1.1s ease-in-out infinite; }

  .live-metrics { position: absolute; top: 24px; right: 24px; display: grid; grid-template-columns: repeat(3, minmax(80px, 1fr)); gap: 1px; overflow: hidden; border: 1px solid #31363e; border-radius: 5px; background: #31363e; }
  .live-metrics div { display: grid; gap: 3px; padding: 8px 10px; background: #171b20; }
  .live-metrics span { color: #6f8792; font-size: 8px; text-transform: uppercase; }
  .live-metrics strong { font-size: 10px; }
  .live-metrics strong.good { color: var(--mint); }
  .live-overlays { position: absolute; top: 24px; left: 24px; z-index: 3; display: flex; padding: 3px; gap: 2px; border: 1px solid #31363e; border-radius: 5px; background: #171b20e8; }
  .live-overlays button { min-width: 0; padding: 7px 9px; border: 0; color: #82929a; background: transparent; font-size: 9px; text-transform: uppercase; }
  .live-overlays button.active { color: #101719; background: var(--mint); }
  .live-guidance { position: absolute; left: 50%; bottom: 26px; z-index: 3; display: grid; gap: 4px; min-width: 320px; padding: 10px 14px; transform: translateX(-50%); border: 1px solid #31363e; border-radius: 5px; background: #171b20e8; text-align: center; }
  .live-guidance strong { color: #e8f1f3; font-size: 11px; }
  .live-guidance span { color: #82929a; font-size: 9px; }
  .job-overlay { position: absolute; right: 24px; bottom: 24px; width: min(420px, calc(100% - 48px)); padding: 13px; border: 1px solid #31363e; border-radius: 5px; background: #171b20; }
  .job-overlay > div:first-child { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 10px; text-transform: capitalize; }
  .job-overlay .job-quality { justify-content: flex-start; }
  .job-overlay p { margin-top: 8px; color: #78909c; font-size: 9px; }

  footer { display: flex; align-items: center; gap: 8px; padding: 0 14px; border: 1px solid var(--line); border-radius: 6px; background: #14171b; color: #858c96; font-size: 9px; }
  footer strong { color: #9bb1bb; font-size: 9px; letter-spacing: .08em; }
  footer p { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  footer.error p { color: #d8988e; }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--mint); }
  .status-dot.busy { background: var(--amber); animation: pulse 1s infinite; }
  .footer-metric { padding-left: 12px; border-left: 1px solid var(--line); color: #607984; }
  .spinner { width: 24px; height: 24px; margin-bottom: 12px; border: 2px solid rgba(99,199,231,.16); border-top-color: var(--cyan); border-radius: 50%; animation: spin .8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes pulse { 50% { opacity: .35; } }
  @keyframes export-slide { from { transform: translateX(-110%); } to { transform: translateX(300%); } }

  @media (max-width: 1120px) {
    main { grid-template-columns: minmax(0, 1fr) 340px; }
    .topbar { grid-template-columns: auto 1fr auto; }
    .project-title { display: none; }
    .live-metrics { grid-template-columns: repeat(2, minmax(80px, 1fr)); }
  }
</style>
