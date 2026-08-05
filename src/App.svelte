<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { save } from '@tauri-apps/plugin-dialog';
  import PointCloudPreview from './lib/components/PointCloudPreview.svelte';
  import {
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
    latestArtifactJob,
    loadGaussianSplat,
    loadLivePreviewFrame,
    loadLiveReconstructionMesh,
    loadPreview,
    loadPreviewMesh,
    removeCapture,
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
    CaptureSettings,
    CaptureStatus,
    DepthFieldOfView,
    LiveReconstructionMode,
    MeshViewMode,
    PackedPreviewFrame,
    PreviewMesh,
    PreviewPoint,
    ProjectSummary,
    RuntimeInfo,
    SensorKind
  } from './lib/types';

  type Workspace = 'capture' | 'reconstruct' | 'inspect';
  type RenderMode = 'points' | 'mesh' | 'splat';

  let project: ProjectSummary | null = null;
  let sensor: CaptureStatus | null = null;
  let runtime: RuntimeInfo | null = null;
  let sensors: AvailableSensor[] = [];
  let activeJob: ArtifactJob | null = null;
  let workspace: Workspace = 'capture';
  let renderMode: RenderMode = 'points';
  let meshViewMode: MeshViewMode = 'surface';

  let previewPoints: PreviewPoint[] = [];
  let packedPreviewFrame: PackedPreviewFrame | null = null;
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
  let statusInFlight = false;
  let geometryInFlight = false;
  let resultInFlight = false;
  let message = 'Initializing the RGB-D engine…';
  let fatalError = '';
  let statusTimer: number | undefined;
  let settingsTimer: number | undefined;
  let settingsRevision = 0;
  let selectingSensor = false;
  let lastPreviewFrame = 0;
  let lastMeshFrame = 0;
  let lastBuildPreviewAt = 0;
  let completedJobId = '';

  let capturing = false;
  let processing = false;
  let completedCaptures = 0;
  let totalFrames = 0;
  let readyArtifacts = 0;
  let viewerRenderMode: RenderMode = 'points';
  let viewerMesh: PreviewMesh | null = null;
  let viewerPackedFrame: PackedPreviewFrame | null = null;
  let currentSensorKey = '';

  $: capturing = Boolean(sensor?.capturing);
  $: processing = Boolean(activeJob && ['queued', 'running', 'cancelling'].includes(activeJob.status));
  $: completedCaptures = project?.phases.filter((capture) => capture.status === 'complete').length ?? 0;
  $: totalFrames = project?.phases.reduce((sum, capture) => sum + capture.frameCount, 0) ?? 0;
  $: readyArtifacts = project
    ? Object.values(project.artifacts).filter((artifact) => artifact && !artifact.stale && artifact.status === 'ready').length
    : 0;
  $: viewerRenderMode = capturing
    ? project?.settings.liveReconstruction === 'mesh' ? 'mesh' : 'points'
    : renderMode;
  $: viewerMesh = capturing ? liveMesh : previewMesh;
  $: viewerPackedFrame = capturing ? packedPreviewFrame : null;
  $: currentSensorKey = project ? configuredSensorKey(project.settings) : '';

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

  function updateSetting<K extends keyof CaptureSettings>(key: K, value: CaptureSettings[K]): void {
    if (!project || capturing || processing) return;
    project = { ...project, settings: { ...project.settings, [key]: value } };
    settingsRevision += 1;
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
    try {
      const saved = await updateProjectSettings(snapshot.path, snapshot.settings);
      if (revision === settingsRevision) {
        project = saved;
        if (refreshRuntime) runtime = await runtimeInfo().catch(() => runtime);
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
      message = `${candidate.name} selected. ${runtime?.sensorStatus ?? ''}`.trim();
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
      const fallback = await selectSupportedFallback();
      message = fallback
        ? `${fallback} selected because the previous camera backend is not installed.`
        : sensors.length
          ? `${sensors.length} RGB-D source${sensors.length === 1 ? '' : 's'} available.`
          : runtime.sensorStatus;
    } catch (error) {
      message = errorText(error);
    } finally {
      discovering = false;
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
    if (!project || !capturing || geometryInFlight) return;
    geometryInFlight = true;
    try {
      const packet = parsePointPacket(await loadLivePreviewFrame(lastPreviewFrame));
      if (packet && packet.frameCount > lastPreviewFrame) {
        packedPreviewFrame = packet;
        lastPreviewFrame = packet.frameCount;
      }
      if (project.settings.liveReconstruction === 'mesh') {
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
    if (!project || !processing || resultInFlight || performance.now() - lastBuildPreviewAt < 1200) return;
    lastBuildPreviewAt = performance.now();
    resultInFlight = true;
    try {
      if (activeJob?.stage.includes('splat')) {
        const next = await loadGaussianSplat(project.path).catch(() => null);
        if (next?.byteLength) {
          previewSplat = next;
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
    if (statusInFlight || !project) return;
    statusInFlight = true;
    try {
      const wasCapturing = capturing;
      if (forceCapture || wasCapturing) {
        const next = await captureStatus();
        sensor = next;
        project = next.project;
        if (next.error) message = next.error;
        if (next.capturing) await pollLiveGeometry();
        else if (wasCapturing && !next.error) {
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
    } catch (error) {
      message = errorText(error);
    } finally {
      statusInFlight = false;
    }
  }

  async function captureAction(): Promise<void> {
    if (!project || busy || processing) return;
    busy = true;
    try {
      if (capturing) {
        message = 'Finishing the archive and draining reconstruction queues…';
        project = await stopSensorPhase();
      } else {
        if (settingsTimer) window.clearTimeout(settingsTimer);
        settingsTimer = undefined;
        packedPreviewFrame = null;
        liveMesh = null;
        lastPreviewFrame = 0;
        lastMeshFrame = 0;
        message = `Warming realtime ${project.settings.liveReconstruction === 'mesh' ? 'mesh' : 'point'} reconstruction…`;
        project = await startSensorPhase(project.path, project.settings);
      }
      await pollStatus(true);
    } catch (error) {
      message = errorText(error);
    } finally {
      busy = false;
    }
  }

  async function removeCaptureAction(id: string, name: string): Promise<void> {
    if (busy || processing || capturing || !window.confirm(`Delete ${name} and invalidate all reconstructed outputs?`)) return;
    busy = true;
    try {
      project = await removeCapture(id);
      previewPoints = [];
      previewMesh = null;
      previewSplat = null;
      activeJob = null;
      message = `${name} deleted. Existing reconstruction outputs were invalidated.`;
    } catch (error) {
      message = errorText(error);
    } finally {
      busy = false;
    }
  }

  async function startBuild(resume = false): Promise<void> {
    if (!project || busy || capturing) return;
    if (resume && activeJob) {
      busy = true;
      try {
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
      completedJobId = '';
      activeJob = await startArtifactJob(project.path, targets, splatIterations);
      workspace = 'reconstruct';
      message = 'Started quality-gated RGB-D reconstruction.';
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

  async function newProjectAction(): Promise<void> {
    if (capturing || processing) return;
    if (project && (project.phases.length || readyArtifacts) && !window.confirm('Start a new RGB-D scan? The current project remains on disk.')) return;
    busy = true;
    try {
      project = await createProject();
      sensor = null;
      activeJob = null;
      previewPoints = [];
      previewMesh = null;
      previewSplat = null;
      packedPreviewFrame = null;
      liveMesh = null;
      workspace = 'capture';
      message = 'New RGB-D scan ready.';
      await discoverSensors();
    } catch (error) {
      message = errorText(error);
    } finally {
      busy = false;
    }
  }

  async function exportPointCloud(): Promise<void> {
    if (!project || !artifactReady('pointCloud')) return;
    const destination = await save({ title: 'Export metric point cloud', defaultPath: 'scan-cloud.ply', filters: [{ name: 'PLY point cloud', extensions: ['ply'] }] });
    if (!destination) return;
    try {
      message = `Point cloud exported to ${await exportPly(project.path, destination)}.`;
    } catch (error) { message = errorText(error); }
  }

  async function exportMesh(): Promise<void> {
    if (!project || !artifactReady('texturedMesh')) return;
    const destination = await save({ title: 'Export textured mesh bundle', defaultPath: 'scan-mesh.obj', filters: [{ name: 'Wavefront OBJ', extensions: ['obj'] }] });
    if (!destination) return;
    try {
      message = `OBJ, MTL, and texture exported beside ${await exportTexturedMesh(project.path, destination)}.`;
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
    void (async () => {
      try {
        project = await currentProject();
        [runtime, sensors] = await Promise.all([
          runtimeInfo().catch(() => null),
          availableSensors().catch(() => [])
        ]);
        const fallback = await selectSupportedFallback();
        sensor = await captureStatus().catch(() => null);
        if (project.activeJob || project.processingStatus === 'processing' || project.processingStatus === 'failed') {
          activeJob = await latestArtifactJob(project.path).catch(() => null);
          if (activeJob && ['queued', 'running', 'cancelling'].includes(activeJob.status)) workspace = 'reconstruct';
        }
        if (!capturing && readyArtifacts) {
          workspace = 'inspect';
          if (artifactReady('texturedMesh')) await loadResult('mesh');
          else if (artifactReady('pointCloud')) await loadResult('points');
          else if (artifactReady('gaussianSplat')) await loadResult('splat');
        }
        message = fallback
          ? `${fallback} selected because the previous camera backend is not installed.`
          : runtime?.sensorStatus ?? 'RGB-D workspace ready.';
        statusTimer = window.setInterval(() => void pollStatus(), 300);
      } catch (error) {
        fatalError = errorText(error);
        message = fatalError;
      }
    })();
  });

  onDestroy(() => {
    if (statusTimer) window.clearInterval(statusTimer);
    if (settingsTimer) window.clearTimeout(settingsTimer);
  });
</script>

<svelte:head><title>ScanLan · RGB-D Reconstruction</title></svelte:head>

<div class="app-shell">
  <header class="topbar">
    <div class="brand"><span class="brand-mark">SL</span><div><strong>ScanLan</strong><small>Realtime RGB-D reconstruction</small></div></div>
    <div class="project-title"><span>ACTIVE SCAN</span><strong>{project?.name ?? 'Loading…'}</strong></div>
    <div class="runtime-state">
      <span class:ready={Boolean(runtime?.sensorWorkerAvailable)}><i></i>Capture</span>
      <span class:ready={Boolean(runtime?.reconstructionWorkerAvailable)}><i></i>Reconstruct</span>
      <span class:ready={Boolean(runtime?.splatWorkerAvailable)}><i></i>2DGS CUDA</span>
    </div>
    <button class="ghost compact" on:click={newProjectAction} disabled={busy || capturing || processing}>New scan</button>
  </header>

  <nav class="workflow" aria-label="Workflow">
    <button class:active={workspace === 'capture'} class:done={completedCaptures > 0} on:click={() => workspace = 'capture'}>
      <span>01</span><div><strong>Capture</strong><small>{capturing ? 'Recording now' : `${completedCaptures} take${completedCaptures === 1 ? '' : 's'}`}</small></div>
    </button>
    <button class:active={workspace === 'reconstruct'} class:done={readyArtifacts > 0} on:click={() => workspace = 'reconstruct'} disabled={capturing}>
      <span>02</span><div><strong>Reconstruct</strong><small>{processing ? activeJob?.stage.replaceAll('_', ' ') : 'Points · mesh · 2DGS'}</small></div>
    </button>
    <button class:active={workspace === 'inspect'} class:done={readyArtifacts > 0} on:click={() => workspace = 'inspect'} disabled={capturing || (!processing && readyArtifacts === 0)}>
      <span>03</span><div><strong>Inspect & export</strong><small>{readyArtifacts ? `${readyArtifacts} output${readyArtifacts === 1 ? '' : 's'} ready` : 'No output yet'}</small></div>
    </button>
  </nav>

  <main>
    <section class="viewport">
      <PointCloudPreview
        points={capturing ? [] : previewPoints}
        packedFrame={viewerPackedFrame}
        processing={processing}
        live={capturing}
        pointSize={0.026}
        opacity={0.95}
        showColors={true}
        renderMode={viewerRenderMode}
        mesh={viewerMesh}
        splatBytes={capturing ? null : previewSplat}
        {meshViewMode}
        assetLoading={assetLoading}
      />

      {#if capturing && sensor}
        <div class="live-metrics">
          <div><span>Tracking</span><strong class:good={sensor.tracking}>{sensor.tracking ? 'LOCKED' : 'SEARCHING'}</strong></div>
          <div><span>Tracker</span><strong>{sensor.trackingFps.toFixed(1)} fps</strong></div>
          <div><span>Keyframes</span><strong>{sensor.liveIntegratedFrameCount}</strong></div>
          <div><span>Overlap</span><strong>{Math.round(sensor.trackingOverlap * 100)}%</strong></div>
          <div><span>Depth error</span><strong>{sensor.depthRmseMm ? `${sensor.depthRmseMm.toFixed(1)} mm` : '—'}</strong></div>
          <div><span>Queue drops</span><strong>{sensor.trackingQueueDropCount + sensor.mappingDropCount}</strong></div>
        </div>
      {/if}

      {#if processing && activeJob}
        <div class="job-overlay">
          <div><span>{activeJob.stage.replaceAll('_', ' ')}</span><strong>{Math.round(activeJob.progress * 100)}%</strong></div>
          <div class="progress"><i style={`width:${Math.round(activeJob.progress * 100)}%`}></i></div>
          <p>{activeJob.detail}</p>
        </div>
      {/if}
    </section>

    <aside>
      {#if !project}
        <section class="panel"><div class="spinner"></div><h2>Starting ScanLan</h2><p>{message}</p></section>
      {:else if workspace === 'capture'}
        <section class="panel panel-heading">
          <div><span>RGB-D SOURCE</span><h2>{capturing ? sensor?.sensorName ?? 'Capturing' : 'Camera & live fusion'}</h2></div>
          <button class="icon-button" on:click={discoverSensors} disabled={discovering || selectingSensor || capturing || processing} title="Refresh cameras">↻</button>
        </section>

        <section class="panel settings">
          <label>Capture source
            <select value={currentSensorKey} on:change={chooseSensor} disabled={capturing || processing || discovering || selectingSensor}>
              {#if !sensors.some((item) => sensorKey(item) === currentSensorKey)}
                <option value={currentSensorKey}>{project.settings.sensorKind.replaceAll('_', ' ')} · configured</option>
              {/if}
              {#each sensors as candidate}
                <option value={sensorKey(candidate)}>{candidate.name}{candidate.connection === 'network' ? ` · ${candidate.address}` : ''}</option>
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

        {#if capturing && sensor}
          <section class="panel tracking-card" class:warning={!sensor.tracking}>
            <div class="tracking-title"><i></i><div><strong>{sensor.trackingStatus}</strong><small>{sensor.liveReconstructionBackend ?? 'Realtime engine'}</small></div></div>
            <div class="mini-grid">
              <div><span>Sensor</span><strong>{sensor.streamFps.toFixed(1)} fps</strong></div>
              <div><span>Archived</span><strong>{sensor.frameCount}</strong></div>
              <div><span>Rejected</span><strong>{sensor.liveRejectedFrameCount}</strong></div>
              <div><span>Source drops</span><strong>{sensor.sourceDropCount}</strong></div>
            </div>
            <p>Move steadily, keep 40–70% of the previous view visible, and revisit the start before stopping.</p>
          </section>
        {/if}

        <button class:stop={capturing} class="capture-button" on:click={captureAction} disabled={busy || selectingSensor || processing || (!capturing && runtime && !runtime.sensorWorkerAvailable)}>
          <i></i><span>{capturing ? 'Stop & save take' : busy ? 'Starting engine…' : 'Start capture'}</span>
        </button>

        <section class="panel takes">
          <div class="section-title"><span>RECORDED TAKES</span><strong>{totalFrames.toLocaleString()} frames</strong></div>
          {#if project.phases.length === 0}
            <p class="empty-copy">No RGB-D takes yet. Tracking runs at sensor rate; the archive rate only controls frames kept for the production pass.</p>
          {:else}
            {#each project.phases as capture, index}
              <article>
                <span class="take-number">{String(index + 1).padStart(2, '0')}</span>
                <div><strong>{capture.name}</strong><small>{capture.frameCount.toLocaleString()} frames · {formatDuration(capture.durationSeconds)}</small></div>
                <button on:click={() => removeCaptureAction(capture.id, capture.name)} disabled={busy || capturing || processing}>Delete</button>
              </article>
            {/each}
          {/if}
        </section>

      {:else if workspace === 'reconstruct'}
        <section class="panel panel-heading"><div><span>PRODUCTION PASS</span><h2>Reconstruction outputs</h2></div><strong class="take-total">{completedCaptures} take{completedCaptures === 1 ? '' : 's'}</strong></section>

        <section class="panel target-list">
          <label class:active={buildPointCloud}><input type="checkbox" bind:checked={buildPointCloud} disabled={processing}/><span class="target-icon">P</span><div><strong>Metric point cloud</strong><small>Filtered colored PLY · quickest</small></div><i>{artifactReady('pointCloud') ? 'READY' : ''}</i></label>
          <label class:active={buildTexturedMesh}><input type="checkbox" bind:checked={buildTexturedMesh} disabled={processing}/><span class="target-icon">M</span><div><strong>Textured triangle mesh</strong><small>TSDF surface · OBJ/MTL/PNG</small></div><i>{artifactReady('texturedMesh') ? 'READY' : ''}</i></label>
          <label class:active={buildGaussianSplat}><input type="checkbox" bind:checked={buildGaussianSplat} disabled={processing || !runtime?.splatWorkerAvailable}/><span class="target-icon">G</span><div><strong>2D Gaussian surface</strong><small>Depth-aware discs · metric PLY</small></div><i>{artifactReady('gaussianSplat') ? 'READY' : runtime?.splatWorkerAvailable ? '' : 'CUDA RUNTIME MISSING'}</i></label>
          {#if buildGaussianSplat}
            <label class="iterations"><span>Training iterations</span><input type="range" min="5000" max="60000" step="5000" bind:value={splatIterations} disabled={processing}/><strong>{Number(splatIterations).toLocaleString()}</strong></label>
          {/if}
        </section>

        <section class="panel pipeline-note">
          <strong>One trajectory, three representations</strong>
          <p>All outputs share the same quality-gated RGB-D poses. The final pass stabilizes the trajectory, fuses a weighted TSDF, and only then builds the selected representations.</p>
          <div><span>Source</span><strong>{totalFrames.toLocaleString()} archived frames</strong></div>
          <div><span>Compute</span><strong>{runtime?.reconstructionWorkerAvailable ? 'CUDA preferred' : 'Runtime missing'}</strong></div>
        </section>

        {#if activeJob}
          <section class="panel job-card" class:error={activeJob.status === 'failed'}>
            <div class="section-title"><span>{activeJob.status.toUpperCase()}</span><strong>{Math.round(activeJob.progress * 100)}%</strong></div>
            <h3>{activeJob.stage.replaceAll('_', ' ')}</h3>
            <p>{activeJob.error ?? activeJob.detail}</p>
            <div class="progress"><i style={`width:${Math.round(activeJob.progress * 100)}%`}></i></div>
            <div class="job-meta"><span>{activeJob.computeBackend ?? 'Waiting for worker'}</span><span>{activeJob.etaSeconds ? `~${formatDuration(activeJob.etaSeconds)}` : ''}</span></div>
            {#if processing}
              <button class="ghost full" on:click={cancelBuild}>Cancel safely</button>
            {:else if activeJob.resumable && ['failed', 'cancelled'].includes(activeJob.status)}
              <div class="button-row"><button class="primary" on:click={() => startBuild(true)}>Resume checkpoint</button><button class="ghost" on:click={discardBuild}>Discard</button></div>
            {/if}
          </section>
        {/if}

        <button class="primary full build-button" on:click={() => startBuild(false)} disabled={busy || processing || completedCaptures === 0 || (!buildPointCloud && !buildTexturedMesh && !buildGaussianSplat)}>{processing ? 'Reconstruction running…' : readyArtifacts ? 'Rebuild selected outputs' : 'Build selected outputs'}</button>

      {:else}
        <section class="panel panel-heading"><div><span>RESULT</span><h2>Inspect & export</h2></div><strong class="take-total">{readyArtifacts} ready</strong></section>
        <section class="panel view-switcher">
          <button class:active={renderMode === 'points'} disabled={!artifactReady('pointCloud')} on:click={() => loadResult('points')}><span>P</span><div><strong>Points</strong><small>{formatCount(project.pointCount)}</small></div></button>
          <button class:active={renderMode === 'mesh'} disabled={!artifactReady('texturedMesh')} on:click={() => loadResult('mesh')}><span>M</span><div><strong>Mesh</strong><small>{formatCount(project.meshTriangleCount)} tris</small></div></button>
          <button class:active={renderMode === 'splat'} disabled={!artifactReady('gaussianSplat')} on:click={() => loadResult('splat')}><span>G</span><div><strong>2DGS</strong><small>Metric surface</small></div></button>
        </section>
        {#if renderMode === 'mesh'}
          <section class="panel settings"><label>Mesh display<select bind:value={meshViewMode}><option value="surface">Textured</option><option value="surface-wireframe">Texture + wire</option><option value="wireframe">Wireframe</option><option value="shaded">Shaded</option></select></label></section>
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
    <span class="status-dot" class:busy={busy || capturing || processing}></span>
    <strong>{capturing ? 'LIVE' : processing ? 'BUILDING' : fatalError ? 'ERROR' : 'READY'}</strong>
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
  .project-title strong { max-width: 360px; overflow: hidden; font-size: 13px; text-overflow: ellipsis; white-space: nowrap; }
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
  .tracking-title { gap: 10px; }
  .tracking-title > i { width: 9px; height: 9px; border-radius: 50%; background: var(--mint); box-shadow: 0 0 12px rgba(98,214,186,.5); }
  .tracking-card.warning .tracking-title > i { background: var(--amber); }
  .tracking-title div { display: grid; gap: 3px; }
  .tracking-title strong { font-size: 11px; }
  .tracking-title small { color: var(--muted); font-size: 9px; }
  .tracking-card > p, .empty-copy, .pipeline-note p, .job-card p, .result-stats p { margin-top: 11px; color: #708792; font-size: 10px; line-height: 1.55; }
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
  .job-card.error { border-color: rgba(226,120,103,.35); }
  .progress { height: 5px; overflow: hidden; border-radius: 6px; background: #172a35; }
  .progress i { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--cyan), var(--mint)); transition: width .25s linear; }
  .job-card .progress { margin: 11px 0 8px; }
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
