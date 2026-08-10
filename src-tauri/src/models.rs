use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureSettings {
    pub capture_fps: u32,
    #[serde(default)]
    pub sensor_fps: u32,
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
    #[serde(default = "default_rgb_resolution")]
    pub rgb_resolution: String,
    #[serde(default = "default_true")]
    pub rgb_auto_exposure: bool,
    #[serde(default = "default_rgb_exposure_us")]
    pub rgb_exposure_us: u32,
    #[serde(default)]
    pub rgb_gain: i32,
    #[serde(default = "default_true")]
    pub rgb_auto_white_balance: bool,
    #[serde(default = "default_rgb_white_balance_k")]
    pub rgb_white_balance_k: u32,
    #[serde(default)]
    pub rgb_color_adjustments_enabled: bool,
    #[serde(default = "default_rgb_brightness")]
    pub rgb_brightness: i32,
    #[serde(default = "default_rgb_contrast")]
    pub rgb_contrast: i32,
    #[serde(default = "default_rgb_saturation")]
    pub rgb_saturation: i32,
    #[serde(default = "default_rgb_sharpness")]
    pub rgb_sharpness: i32,
    #[serde(default)]
    pub rgb_backlight_compensation: bool,
    #[serde(default)]
    pub rgb_powerline_hz: u32,
    #[serde(default)]
    pub imu_accel_rate_hz: u32,
    #[serde(default)]
    pub imu_accel_range_g: u32,
    #[serde(default)]
    pub imu_gyro_rate_hz: u32,
    #[serde(default)]
    pub imu_gyro_range_dps: u32,
    pub live_reconstruction: String,
    #[serde(default = "default_live_map_memory_mib")]
    pub live_map_memory_mib: u32,
    #[serde(default = "default_repair_mesh")]
    pub repair_mesh: bool,
    #[serde(default = "default_mesh_repair_profile")]
    pub mesh_repair_profile: String,
    #[serde(default)]
    pub fill_inferred_mesh_holes: bool,
    #[serde(default)]
    pub produce_watertight_mesh: bool,
    #[serde(default)]
    pub lingbot_depth_refinement: bool,
    #[serde(default = "default_depth_refinement_backend")]
    pub depth_refinement_backend: String,
    #[serde(default)]
    pub experimental_rgb_preview: bool,
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

fn default_rgb_resolution() -> String {
    "auto".to_string()
}

fn default_true() -> bool {
    true
}

fn default_rgb_exposure_us() -> u32 {
    8_330
}

fn default_rgb_white_balance_k() -> u32 {
    4_500
}

fn default_rgb_brightness() -> i32 {
    128
}

fn default_rgb_contrast() -> i32 {
    5
}

fn default_rgb_saturation() -> i32 {
    32
}

fn default_rgb_sharpness() -> i32 {
    2
}

fn default_live_reconstruction() -> String {
    "points".to_string()
}

fn default_live_map_memory_mib() -> u32 {
    1024
}

fn default_repair_mesh() -> bool {
    true
}

fn default_mesh_repair_profile() -> String {
    "faithful".to_string()
}

fn default_depth_refinement_backend() -> String {
    "off".to_string()
}

impl Default for CaptureSettings {
    fn default() -> Self {
        Self {
            capture_fps: 10,
            sensor_fps: 0,
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
            rgb_resolution: default_rgb_resolution(),
            rgb_auto_exposure: true,
            rgb_exposure_us: default_rgb_exposure_us(),
            rgb_gain: 0,
            rgb_auto_white_balance: true,
            rgb_white_balance_k: default_rgb_white_balance_k(),
            rgb_color_adjustments_enabled: false,
            rgb_brightness: default_rgb_brightness(),
            rgb_contrast: default_rgb_contrast(),
            rgb_saturation: default_rgb_saturation(),
            rgb_sharpness: default_rgb_sharpness(),
            rgb_backlight_compensation: false,
            rgb_powerline_hz: 0,
            imu_accel_rate_hz: 0,
            imu_accel_range_g: 0,
            imu_gyro_rate_hz: 0,
            imu_gyro_range_dps: 0,
            live_reconstruction: default_live_reconstruction(),
            live_map_memory_mib: default_live_map_memory_mib(),
            repair_mesh: default_repair_mesh(),
            mesh_repair_profile: default_mesh_repair_profile(),
            fill_inferred_mesh_holes: false,
            produce_watertight_mesh: false,
            lingbot_depth_refinement: false,
            depth_refinement_backend: default_depth_refinement_backend(),
            experimental_rgb_preview: false,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_repair_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_repair_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_repair_report_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_repair_fallback: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_repair_defects_fixed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_repair_holes_filled: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_repair_openings_preserved: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_repair_unknown_preserved: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub watertight_mesh_output_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_refinement: Option<serde_json::Value>,
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
            mesh_repair_profile: None,
            mesh_repair_status: None,
            mesh_repair_report_path: None,
            mesh_repair_fallback: None,
            mesh_repair_defects_fixed: None,
            mesh_repair_holes_filled: None,
            mesh_repair_openings_preserved: None,
            mesh_repair_unknown_preserved: None,
            watertight_mesh_output_path: None,
            depth_refinement: None,
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
    pub geometry_worker_available: bool,
    pub geometry_status: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureStatus {
    pub project: ProjectSummary,
    pub live_contract_version: u16,
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
    pub tracking_state: String,
    pub tracking_confidence: f32,
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
    pub tracking_queue_depth: u32,
    pub mapping_queue_depth: u32,
    pub tracking_overlap: f32,
    pub pose_uncertainty_mm: Option<f32>,
    pub pose_uncertainty_degrees: Option<f32>,
    pub pose_latency_ms: Option<f32>,
    pub map_update_latency_ms: Option<f32>,
    pub map_update_hz: f32,
    pub allocated_live_map_bytes: u64,
    pub active_voxel_count: u64,
    pub active_surfel_count: u64,
    pub resident_submap_count: u32,
    pub host_cached_submap_count: u32,
    pub dropped_preview_job_count: u64,
    pub degradation_level: u8,
    pub loop_closure_count: u32,
    pub loop_correction_active: bool,
    pub live_scale_status: String,
    pub integration_frozen: bool,
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
    pub contract_version: u16,
    #[serde(default)]
    pub active: bool,
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub tracking: bool,
    #[serde(default)]
    pub tracking_status: String,
    #[serde(default)]
    pub tracking_state: String,
    #[serde(default)]
    pub tracking_confidence: f32,
    #[serde(default)]
    pub processed_frames: u32,
    #[serde(default)]
    pub accepted_frames: u32,
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
    #[serde(default)]
    pub pose_uncertainty_mm: Option<f32>,
    #[serde(default)]
    pub pose_uncertainty_degrees: Option<f32>,
    #[serde(default)]
    pub pose_latency_ms: Option<f32>,
    #[serde(default)]
    pub map_update_latency_ms: Option<f32>,
    #[serde(default)]
    pub map_update_hz: f32,
    #[serde(default)]
    pub allocated_live_map_bytes: u64,
    #[serde(default)]
    pub active_voxel_count: u64,
    #[serde(default)]
    pub active_surfel_count: u64,
    #[serde(default)]
    pub resident_submap_count: u32,
    #[serde(default)]
    pub host_cached_submap_count: u32,
    #[serde(default)]
    pub dropped_preview_jobs: u64,
    #[serde(default)]
    pub tracking_queue_depth: u32,
    #[serde(default)]
    pub mapping_queue_depth: u32,
    #[serde(default)]
    pub degradation_level: u8,
    #[serde(default)]
    pub loop_closure_count: u32,
    #[serde(default)]
    pub loop_correction_active: bool,
    #[serde(default)]
    pub scale_status: String,
    #[serde(default)]
    pub integration_frozen: bool,
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
    #[serde(default)]
    pub media_restart: String,
    #[serde(default)]
    pub rebuild_rgbd: bool,
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
    #[serde(default)]
    pub rgb_preview_active: bool,
    #[serde(default)]
    pub rgb_preview_scale_status: Option<String>,
    #[serde(default)]
    pub rgb_preview_confidence: Option<f32>,
    #[serde(default)]
    pub rgb_preview_drift_risk: Option<f32>,
    #[serde(default)]
    pub rgb_preview_submap_count: Option<u32>,
    #[serde(default)]
    pub rgb_preview_accepted_frames: Option<u32>,
    #[serde(default)]
    pub rgb_preview_rejected_frames: Option<u32>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoundingBoxClip {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

fn unit_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
