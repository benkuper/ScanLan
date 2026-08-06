use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureSettings {
    pub capture_fps: u32,
    pub max_depth_m: f32,
    pub voxel_size_mm: u32,
    pub sensor_kind: String,
    pub sensor_id: String,
    pub sensor_connection: String,
    pub sensor_address: String,
    pub use_imu: bool,
    pub depth_field_of_view: String,
    pub depth_binned: bool,
    pub rgb_jpeg_quality: u8,
    pub max_rgb_dimension: u32,
    pub live_reconstruction: String,
}

fn default_sensor_kind() -> String {
    "kinect_v2".to_string()
}

fn default_sensor_connection() -> String {
    "usb".to_string()
}

fn default_use_imu() -> bool {
    true
}

fn default_depth_field_of_view() -> String {
    "narrow".to_string()
}

fn default_rgb_jpeg_quality() -> u8 {
    92
}

fn default_live_reconstruction() -> String {
    "points".to_string()
}

impl Default for CaptureSettings {
    fn default() -> Self {
        Self {
            capture_fps: 10,
            max_depth_m: 4.2,
            voxel_size_mm: 10,
            sensor_kind: default_sensor_kind(),
            sensor_id: String::new(),
            sensor_connection: default_sensor_connection(),
            sensor_address: String::new(),
            use_imu: default_use_imu(),
            depth_field_of_view: default_depth_field_of_view(),
            depth_binned: false,
            rgb_jpeg_quality: default_rgb_jpeg_quality(),
            max_rgb_dimension: 0,
            live_reconstruction: default_live_reconstruction(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactSummary {
    pub path: String,
    pub status: String,
    pub source_fingerprint: String,
    pub updated_at: String,
    pub metric: bool,
    pub stale: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactCatalog {
    pub point_cloud: Option<ArtifactSummary>,
    pub textured_mesh: Option<ArtifactSummary>,
    pub gaussian_splat: Option<ArtifactSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AvailableSensor {
    pub id: String,
    pub kind: String,
    pub name: String,
    pub connection: String,
    #[serde(default)]
    pub address: String,
    #[serde(default)]
    pub serial: String,
    #[serde(default)]
    pub connected: bool,
    pub supports_imu: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhaseSummary {
    pub id: String,
    pub name: String,
    pub created_at: String,
    pub duration_seconds: u32,
    pub frame_count: u32,
    pub status: String,
    pub overlap_hint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MediaSourceSummary {
    pub id: String,
    pub name: String,
    pub path: String,
    pub kind: String,
    pub byte_size: u64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSummary {
    pub schema_version: u32,
    pub id: String,
    pub name: String,
    pub path: String,
    pub created_at: String,
    pub phases: Vec<PhaseSummary>,
    #[serde(default)]
    pub media_sources: Vec<MediaSourceSummary>,
    pub artifacts: ArtifactCatalog,
    pub active_job: Option<String>,
    pub settings: CaptureSettings,
    pub processing_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub point_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mesh_triangle_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mesh_output_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera_frame_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_score: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames_used: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing_duration_seconds: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectCatalogEntry {
    pub id: String,
    pub name: String,
    pub path: String,
    pub created_at: String,
    pub modified_at: String,
    pub capture_count: usize,
    pub media_source_count: usize,
    pub frame_count: u64,
    pub artifact_count: usize,
    pub processing_status: String,
}

impl ProjectSummary {
    pub fn placeholder() -> Self {
        Self {
            schema_version: 3,
            id: Uuid::new_v4().to_string(),
            name: "Untitled RGB-D scan".to_string(),
            path: String::new(),
            created_at: Utc::now().to_rfc3339(),
            phases: Vec::new(),
            media_sources: Vec::new(),
            artifacts: ArtifactCatalog::default(),
            active_job: None,
            settings: CaptureSettings::default(),
            processing_status: "idle".to_string(),
            processing_error: None,
            point_count: None,
            output_path: None,
            mesh_triangle_count: None,
            mesh_output_path: None,
            camera_frame_count: None,
            confidence_score: None,
            confidence_label: None,
            confidence_detail: None,
            frames_used: None,
            processing_backend: None,
            processing_duration_seconds: None,
        }
    }

    pub fn at_path(path: &Path) -> Self {
        let mut project = Self::placeholder();
        project.path = path.to_string_lossy().into_owned();
        project
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeInfo {
    pub platform: String,
    pub sensor_capabilities: Vec<String>,
    pub sensor_worker_available: bool,
    pub sensor_status: String,
    pub reconstruction_worker_available: bool,
    pub splat_worker_available: bool,
    pub splat_status: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureStatus {
    pub project: ProjectSummary,
    pub preview: Vec<PreviewPoint>,
    pub capturing: bool,
    pub previewing: bool,
    pub sensor_connected: bool,
    pub sensor_paused: bool,
    pub sensor_status: String,
    pub sensor_name: String,
    pub frame_count: u32,
    pub total_frame_count: u32,
    pub preview_point_count: u64,
    pub stream_fps: f32,
    pub tracking: bool,
    pub tracking_status: String,
    pub imu_active: bool,
    pub imu_rate_hz: f32,
    pub live_reconstruction_active: bool,
    pub live_reconstruction_mode: String,
    pub live_processed_frame_count: u32,
    pub live_integrated_frame_count: u32,
    pub live_rejected_frame_count: u32,
    pub live_triangle_count: u64,
    pub tracking_fps: f32,
    pub source_drop_count: u64,
    pub tracking_queue_drop_count: u64,
    pub mapping_drop_count: u64,
    pub tracking_overlap: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_rmse_mm: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_reconstruction_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reconstruction: Option<ReconstructionProgress>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ReconstructionProgress {
    pub stage: String,
    pub detail: String,
    pub progress: f32,
    pub processed_units: u32,
    pub total_units: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eta_seconds: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub point_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_progress: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_eta_seconds: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_seconds: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_timings_seconds: Option<BTreeMap<String, f32>>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct LiveWorkerStatus {
    #[serde(default)]
    pub frame_count: u32,
    pub stream_fps: f32,
    pub tracking: bool,
    pub tracking_status: String,
    #[serde(default)]
    pub sensor_name: String,
    #[serde(default)]
    pub imu_active: bool,
    #[serde(default)]
    pub imu_rate_hz: f32,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct LiveReconstructionStatus {
    #[serde(default)]
    pub active: bool,
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub tracking: bool,
    #[serde(default)]
    pub tracking_status: String,
    #[serde(default)]
    pub processed_frames: u32,
    #[serde(default)]
    pub integrated_frames: u32,
    #[serde(default)]
    pub rejected_frames: u32,
    #[serde(default)]
    pub point_count: u64,
    #[serde(default)]
    pub triangle_count: u64,
    #[serde(default)]
    pub backend: String,
    #[serde(default)]
    pub tracking_fps: f32,
    #[serde(default)]
    pub source_drops: u64,
    #[serde(default)]
    pub tracking_queue_drops: u64,
    #[serde(default)]
    pub mapping_drops: u64,
    #[serde(default)]
    pub overlap: f32,
    #[serde(default)]
    pub depth_rmse_mm: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraModel {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub depth_scale: f64,
    pub max_depth_m: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RgbCameraModel {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub model: String,
    pub distortion: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceRgbManifest {
    pub format: String,
    pub quality: u8,
    pub native_resolution: bool,
    pub dropped_frames: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhaseManifest {
    pub schema_version: u32,
    pub id: String,
    pub name: String,
    pub created_at: String,
    pub frame_count: u32,
    pub duration_seconds: u32,
    pub frame_format: String,
    pub pose_source: String,
    pub camera: CameraModel,
    pub rgb_camera: RgbCameraModel,
    pub rgb_from_depth: [f64; 16],
    pub source_rgb: SourceRgbManifest,
    pub sensor: SensorManifest,
    #[serde(default)]
    pub imu: Option<ImuManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactJob {
    pub id: String,
    pub project_path: String,
    #[serde(default)]
    pub targets: Vec<String>,
    #[serde(default)]
    pub source_kind: String,
    pub stage: String,
    #[serde(default)]
    pub detail: String,
    pub progress: f32,
    #[serde(default)]
    pub iteration: Option<u32>,
    #[serde(default)]
    pub total_iterations: Option<u32>,
    #[serde(default)]
    pub loss: Option<f32>,
    #[serde(default)]
    pub smoothed_loss: Option<f32>,
    #[serde(default)]
    pub eta_seconds: Option<u32>,
    #[serde(default)]
    pub stage_progress: Option<f32>,
    #[serde(default)]
    pub stage_eta_seconds: Option<u32>,
    #[serde(default)]
    pub elapsed_seconds: Option<u32>,
    #[serde(default)]
    pub compute_backend: Option<String>,
    pub status: String,
    pub created_at: String,
    #[serde(default)]
    pub started_at: Option<String>,
    pub updated_at: String,
    #[serde(default)]
    pub source_fingerprint: String,
    #[serde(default)]
    pub log_path: String,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub resumable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SensorManifest {
    pub kind: String,
    pub name: String,
    pub connection: String,
    pub serial: String,
    pub address: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImuManifest {
    pub path: String,
    pub coordinate_frame: String,
    pub acceleration_unit: String,
    pub angular_velocity_unit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewPoint {
    pub position: [f32; 3],
    pub color: [u8; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CloudTransform {
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    #[serde(default = "unit_scale")]
    pub scale: [f32; 3],
}

fn unit_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
