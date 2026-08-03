use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureSettings {
    pub capture_fps: u32,
    pub max_depth_m: f32,
    pub voxel_size_mm: u32,
    pub environment: String,
    #[serde(default = "default_sensor_kind")]
    pub sensor_kind: String,
    #[serde(default)]
    pub sensor_id: String,
    #[serde(default = "default_sensor_connection")]
    pub sensor_connection: String,
    #[serde(default)]
    pub sensor_address: String,
    #[serde(default = "default_use_imu")]
    pub use_imu: bool,
    #[serde(default = "default_depth_field_of_view")]
    pub depth_field_of_view: String,
    #[serde(default)]
    pub depth_binned: bool,
    #[serde(default = "default_rgb_jpeg_quality")]
    pub rgb_jpeg_quality: u8,
    #[serde(default)]
    pub max_rgb_dimension: u32,
    #[serde(default = "default_live_reconstruction")]
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
            voxel_size_mm: 15,
            environment: "outdoor_low_light".to_string(),
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
pub struct MediaSource {
    pub id: String,
    pub kind: String,
    pub name: String,
    pub created_at: String,
    pub path: String,
    #[serde(default)]
    pub originals: Vec<String>,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub image_count: u32,
    #[serde(default)]
    pub metric: bool,
    #[serde(default)]
    pub quality: Option<MediaSourceQuality>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct MediaSourceQuality {
    #[serde(default)]
    pub registered_images: u32,
    #[serde(default)]
    pub total_images: u32,
    #[serde(default)]
    pub reprojection_error: Option<f32>,
    #[serde(default)]
    pub disconnected_components: u32,
    #[serde(default)]
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactSummary {
    pub path: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub source_fingerprint: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub metric: bool,
    #[serde(default)]
    pub stale: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactCatalog {
    #[serde(default)]
    pub point_cloud: Option<ArtifactSummary>,
    #[serde(default)]
    pub textured_mesh: Option<ArtifactSummary>,
    #[serde(default)]
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
pub struct ProjectSummary {
    pub schema_version: u32,
    pub id: String,
    pub name: String,
    pub path: String,
    pub created_at: String,
    pub phases: Vec<PhaseSummary>,
    #[serde(default)]
    pub media_sources: Vec<MediaSource>,
    #[serde(default)]
    pub artifacts: ArtifactCatalog,
    #[serde(default)]
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

impl ProjectSummary {
    pub fn placeholder() -> Self {
        Self {
            schema_version: 2,
            id: Uuid::new_v4().to_string(),
            name: "Untitled phased scan".to_string(),
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
    pub sensor_worker_available: bool,
    pub sensor_connected: bool,
    pub sensor_status: String,
    pub reconstruction_worker_available: bool,
    pub splat_worker_available: bool,
    pub splat_status: String,
    pub ffmpeg_available: bool,
    pub colmap_available: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureStatus {
    pub project: ProjectSummary,
    pub preview: Vec<PreviewPoint>,
    pub capturing: bool,
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
    #[serde(default = "default_rgb_model")]
    pub model: String,
    #[serde(default)]
    pub distortion: Vec<f64>,
}

fn default_rgb_model() -> String {
    "brown_conrady".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceRgbManifest {
    #[serde(default = "default_rgb_format")]
    pub format: String,
    #[serde(default = "default_rgb_jpeg_quality")]
    pub quality: u8,
    #[serde(default)]
    pub native_resolution: bool,
    #[serde(default)]
    pub dropped_frames: u32,
}

fn default_rgb_format() -> String {
    "jpeg".to_string()
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
    #[serde(default)]
    pub rgb_camera: Option<RgbCameraModel>,
    #[serde(default)]
    pub rgb_from_depth: Option<[f64; 16]>,
    #[serde(default)]
    pub source_rgb: Option<SourceRgbManifest>,
    #[serde(default)]
    pub sensor: Option<SensorManifest>,
    #[serde(default)]
    pub imu: Option<ImuManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactJob {
    pub id: String,
    pub project_path: String,
    pub pipeline: String,
    #[serde(default)]
    pub targets: Vec<String>,
    #[serde(default)]
    pub source_ids: Vec<String>,
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
    #[serde(default)]
    pub serial: String,
    #[serde(default)]
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
pub struct CameraFrame {
    pub phase_name: String,
    pub phase_id: String,
    pub frame_index: u32,
    pub timestamp_us: u64,
    pub matrix: [f32; 16],
    pub aspect: f32,
    pub fov_y_degrees: f32,
    pub image_y_up: bool,
    pub texture_frame: bool,
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
