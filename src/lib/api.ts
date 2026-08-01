import { invoke } from '@tauri-apps/api/core';
import type { AvailableSensor, CameraFrame, CaptureSettings, CaptureStatus, CloudTransform, PreviewMesh, PreviewPoint, ProjectSummary, RuntimeInfo } from './types';

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

export async function reconstructProject(
  projectPath: string,
  settings: CaptureSettings
): Promise<ProjectSummary> {
  requireDesktop();
  return invoke<ProjectSummary>('reconstruct_project', { projectPath, settings });
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
    positions: new Float32Array(bytes.slice(positionStart, uvStart).buffer),
    uvs: new Float32Array(bytes.slice(uvStart, indexStart).buffer),
    indices: new Uint32Array(bytes.slice(indexStart, expectedLength).buffer),
    texture: normalizeBinary(texture)
  };
}

export async function loadCameraFrames(projectPath: string): Promise<CameraFrame[]> {
  requireDesktop();
  return invoke<CameraFrame[]>('load_camera_frames', { projectPath });
}

export async function applyCloudTransform(
  projectPath: string,
  transform: CloudTransform
): Promise<PreviewPoint[]> {
  requireDesktop();
  return invoke<PreviewPoint[]>('apply_cloud_transform', { projectPath, transform });
}

export async function exportPly(projectPath: string, destinationPath: string): Promise<string> {
  requireDesktop();
  return invoke<string>('export_ply', { projectPath, destinationPath });
}

export async function exportTexturedMesh(projectPath: string, destinationPath: string): Promise<string> {
  requireDesktop();
  return invoke<string>('export_textured_mesh', { projectPath, destinationPath });
}
