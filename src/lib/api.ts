import { invoke } from '@tauri-apps/api/core';
import type { ArtifactJob, ArtifactTarget, AvailableSensor, CaptureSettings, CaptureStatus, PreviewMesh, PreviewPoint, ProjectSummary, RuntimeInfo } from './types';

const inTauri = () => typeof window !== 'undefined' && Boolean(window.__TAURI_INTERNALS__);

function requireDesktop(): void {
  if (!inTauri()) {
    throw new Error('ScanLan must be launched as the desktop app to access a depth sensor.');
  }
}

export async function runtimeInfo(): Promise<RuntimeInfo> {
  requireDesktop();
  return invoke<RuntimeInfo>('runtime_info');
}

export async function availableSensors(): Promise<AvailableSensor[]> {
  requireDesktop();
  return invoke<AvailableSensor[]>('available_sensors');
}

export async function currentProject(): Promise<ProjectSummary> {
  requireDesktop();
  return invoke<ProjectSummary>('current_project');
}

export async function createProject(): Promise<ProjectSummary> {
  requireDesktop();
  return invoke<ProjectSummary>('create_project');
}

export async function updateProjectSettings(
  projectPath: string,
  settings: CaptureSettings
): Promise<ProjectSummary> {
  requireDesktop();
  return invoke<ProjectSummary>('update_project_settings', { projectPath, settings });
}

export async function startSensorPhase(
  projectPath: string,
  settings: CaptureSettings
): Promise<ProjectSummary> {
  requireDesktop();
  return invoke<ProjectSummary>('start_sensor_phase', { projectPath, settings });
}

export async function stopSensorPhase(): Promise<ProjectSummary> {
  requireDesktop();
  return invoke<ProjectSummary>('stop_sensor_phase');
}

export async function removeCapture(phaseId: string): Promise<ProjectSummary> {
  requireDesktop();
  return invoke<ProjectSummary>('remove_capture', { phaseId });
}

export async function captureStatus(): Promise<CaptureStatus> {
  requireDesktop();
  return invoke<CaptureStatus>('capture_status');
}

export async function loadLivePreviewFrame(afterFrame: number): Promise<ArrayBuffer> {
  requireDesktop();
  const response = await invoke<ArrayBuffer | Uint8Array | number[]>('live_preview_frame', { afterFrame });
  if (response instanceof ArrayBuffer) return response;
  if (ArrayBuffer.isView(response)) {
    return response.buffer.slice(response.byteOffset, response.byteOffset + response.byteLength) as ArrayBuffer;
  }
  return Uint8Array.from(response).buffer;
}

export async function loadLiveReconstructionMesh(afterFrame: number): Promise<{ frameCount: number; mesh: PreviewMesh } | null> {
  requireDesktop();
  const response = await invoke<ArrayBuffer | Uint8Array | number[]>('live_reconstruction_mesh', { afterFrame });
  const bytes = response instanceof ArrayBuffer
    ? new Uint8Array(response)
    : ArrayBuffer.isView(response)
      ? new Uint8Array(response.buffer.slice(response.byteOffset, response.byteOffset + response.byteLength))
      : Uint8Array.from(response);
  if (bytes.byteLength === 0) return null;
  if (bytes.byteLength < 16 || new TextDecoder().decode(bytes.subarray(0, 4)) !== 'K2M2') {
    throw new Error('The live reconstruction mesh has an invalid header.');
  }
  const header = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const frameCount = header.getUint32(4, true);
  const vertexCount = header.getUint32(8, true);
  const indexCount = header.getUint32(12, true);
  const positionStart = 16;
  const colorStart = positionStart + vertexCount * 12;
  const indexStart = colorStart + vertexCount * 3;
  if (vertexCount > 500_000 || indexCount > 450_000 || bytes.byteLength !== indexStart + indexCount * 4) {
    throw new Error('The live reconstruction mesh is incomplete.');
  }
  return {
    frameCount,
    mesh: {
      positions: new Float32Array(bytes.buffer.slice(bytes.byteOffset + positionStart, bytes.byteOffset + colorStart)),
      colors: new Uint8Array(bytes.buffer.slice(bytes.byteOffset + colorStart, bytes.byteOffset + indexStart)),
      indices: new Uint32Array(bytes.buffer.slice(bytes.byteOffset + indexStart, bytes.byteOffset + bytes.byteLength))
    }
  };
}

export async function startArtifactJob(
  projectPath: string,
  targets: ArtifactTarget[],
  iterations = 30_000
): Promise<ArtifactJob> {
  requireDesktop();
  return invoke<ArtifactJob>('start_artifact_job', {
    projectPath, targets, iterations
  });
}

export async function artifactJobStatus(projectPath: string, jobId: string): Promise<ArtifactJob> {
  requireDesktop();
  return invoke<ArtifactJob>('artifact_job_status', { projectPath, jobId });
}

export async function latestArtifactJob(projectPath: string): Promise<ArtifactJob | null> {
  requireDesktop();
  return invoke<ArtifactJob | null>('latest_artifact_job', { projectPath });
}

export async function cancelArtifactJob(projectPath: string, jobId: string): Promise<ArtifactJob> {
  requireDesktop();
  return invoke<ArtifactJob>('cancel_artifact_job', { projectPath, jobId });
}

export async function discardArtifactJob(projectPath: string, jobId: string): Promise<ArtifactJob> {
  requireDesktop();
  return invoke<ArtifactJob>('discard_artifact_job', { projectPath, jobId });
}

export async function resumeArtifactJob(projectPath: string, jobId: string): Promise<ArtifactJob> {
  requireDesktop();
  return invoke<ArtifactJob>('resume_artifact_job', { projectPath, jobId });
}

export async function loadPreview(projectPath: string): Promise<PreviewPoint[]> {
  requireDesktop();
  return invoke<PreviewPoint[]>('load_preview', { projectPath });
}

function normalizeBinary(response: ArrayBuffer | Uint8Array | number[]): Uint8Array {
  if (response instanceof ArrayBuffer) return new Uint8Array(response);
  if (ArrayBuffer.isView(response)) {
    return new Uint8Array(response.buffer.slice(response.byteOffset, response.byteOffset + response.byteLength));
  }
  return Uint8Array.from(response);
}

export async function loadPreviewMesh(projectPath: string): Promise<PreviewMesh> {
  requireDesktop();
  const [geometry, texture] = await Promise.all([
    invoke<ArrayBuffer | Uint8Array | number[]>('load_preview_mesh_geometry', { projectPath }),
    invoke<ArrayBuffer | Uint8Array | number[]>('load_preview_mesh_texture', { projectPath })
  ]);
  const bytes = normalizeBinary(geometry);
  if (bytes.byteLength < 12 || new TextDecoder().decode(bytes.subarray(0, 4)) !== 'K2M1') {
    throw new Error('The reconstructed mesh preview has an invalid header.');
  }
  const header = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const vertexCount = header.getUint32(4, true);
  const indexCount = header.getUint32(8, true);
  const positionStart = 12;
  const uvStart = positionStart + vertexCount * 12;
  const indexStart = uvStart + vertexCount * 8;
  const expectedLength = indexStart + indexCount * 4;
  if (bytes.byteLength !== expectedLength) {
    throw new Error('The reconstructed mesh preview is incomplete.');
  }
  return {
    positions: new Float32Array(bytes.buffer, bytes.byteOffset + positionStart, vertexCount * 3),
    uvs: new Float32Array(bytes.buffer, bytes.byteOffset + uvStart, vertexCount * 2),
    indices: new Uint32Array(bytes.buffer, bytes.byteOffset + indexStart, indexCount),
    texture: normalizeBinary(texture)
  };
}

export async function loadGaussianSplat(projectPath: string): Promise<Uint8Array> {
  requireDesktop();
  const response = await invoke<ArrayBuffer | Uint8Array | number[]>('load_gaussian_splat', { projectPath });
  return normalizeBinary(response);
}

export async function exportPly(projectPath: string, destinationPath: string): Promise<string> {
  requireDesktop();
  return invoke<string>('export_ply', { projectPath, destinationPath });
}

export async function exportTexturedMesh(projectPath: string, destinationPath: string): Promise<string> {
  requireDesktop();
  return invoke<string>('export_textured_mesh', { projectPath, destinationPath });
}

export async function exportGaussianSplat(projectPath: string, destinationPath: string): Promise<string> {
  requireDesktop();
  return invoke<string>('export_gaussian_splat', { projectPath, destinationPath });
}
