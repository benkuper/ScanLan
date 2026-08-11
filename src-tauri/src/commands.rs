use crate::models::{
    AvailableSensor, BoundingBoxClip, CaptureSettings, CaptureStatus, CloudTransform,
    LiveReconstructionStatus, LiveWorkerStatus, MediaSourceSummary, PreviewPoint,
    ProjectCatalogEntry, ProjectSummary, ReconstructionProgress, RuntimeInfo,
};
use crate::storage;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tauri::{ipc::Response, AppHandle, Manager, State};
use uuid::Uuid;

pub struct ActiveCapture {
    child: Child,
    sensor_relay: Option<thread::JoinHandle<()>>,
    live_reconstruction: Option<Child>,
    realtime: Arc<Mutex<RealtimeEngineSnapshot>>,
    project_root: PathBuf,
    phase_root: PathBuf,
    phase_id: String,
    phase_name: String,
    settings: CaptureSettings,
}

impl Drop for ActiveCapture {
    fn drop(&mut self) {
        // Child handles do not terminate their processes when dropped. Make
        // every early-return path fail closed so a disk or state error cannot
        // leave a camera, relay, or reconstruction worker running invisibly.
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            self.child.kill().ok();
            self.child.wait().ok();
        }
        drain_sensor_relay(self, Duration::from_millis(750));
        stop_live_reconstruction(self, Duration::from_secs(1));
        drain_sensor_relay(self, Duration::from_millis(750));
    }
}

#[derive(Clone)]
struct LiveGeometryFrame {
    frame_count: u32,
    packet: Arc<Vec<u8>>,
}

#[derive(Clone, Default)]
struct RealtimeEngineSnapshot {
    updated: Option<Instant>,
    status: LiveReconstructionStatus,
    camera_points: Option<LiveGeometryFrame>,
    points: Option<LiveGeometryFrame>,
    coverage_points: Option<LiveGeometryFrame>,
    tracking_points: Option<LiveGeometryFrame>,
    mesh: Option<LiveGeometryFrame>,
    coverage: Option<serde_json::Value>,
    submaps: Option<serde_json::Value>,
    error: Option<String>,
}

#[derive(Clone, Copy)]
enum RealtimeGeometry {
    CameraPoints,
    FusedPoints,
    CoveragePoints,
    TrackingPoints,
    Mesh,
}

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RealtimeEngineStatusMessage {
    #[serde(default)]
    contract_version: u16,
    #[serde(default)]
    active: bool,
    #[serde(default)]
    state: String,
    #[serde(default)]
    detail: String,
    #[serde(default)]
    backend: String,
    #[serde(default)]
    processed_frames: u32,
    #[serde(default)]
    accepted_frames: u32,
    #[serde(default)]
    rejected_frames: u32,
    #[serde(default)]
    integrated_frames: u32,
    #[serde(default)]
    point_count: u64,
    #[serde(default)]
    triangle_count: u64,
    #[serde(default)]
    tracking_fps: f32,
    #[serde(default)]
    source_drops: u64,
    #[serde(default)]
    tracking_queue_drops: u64,
    #[serde(default)]
    mapping_drops: u64,
    #[serde(default)]
    overlap: f32,
    #[serde(default)]
    depth_rmse_mm: Option<f32>,
    #[serde(default)]
    tracking_state: String,
    #[serde(default)]
    tracking_confidence: f32,
    #[serde(default)]
    pose_uncertainty_mm: Option<f32>,
    #[serde(default)]
    pose_uncertainty_degrees: Option<f32>,
    #[serde(default)]
    pose_latency_ms: Option<f32>,
    #[serde(default)]
    map_update_latency_ms: Option<f32>,
    #[serde(default)]
    map_update_hz: f32,
    #[serde(default)]
    allocated_live_map_bytes: u64,
    #[serde(default)]
    active_voxel_count: u64,
    #[serde(default)]
    active_surfel_count: u64,
    #[serde(default)]
    resident_submap_count: u32,
    #[serde(default)]
    host_cached_submap_count: u32,
    #[serde(default)]
    dropped_preview_jobs: u64,
    #[serde(default)]
    tracking_queue_depth: u32,
    #[serde(default)]
    mapping_queue_depth: u32,
    #[serde(default)]
    degradation_level: u8,
    #[serde(default)]
    loop_closure_count: u32,
    #[serde(default)]
    loop_correction_active: bool,
    #[serde(default)]
    scale_status: String,
    #[serde(default)]
    integration_frozen: bool,
}

#[derive(Clone)]
pub struct AppState {
    pub project: Arc<Mutex<ProjectSummary>>,
    pub active_capture: Arc<Mutex<Option<ActiveCapture>>>,
    pub active_preview: Arc<Mutex<Option<ActiveCapture>>>,
    pub active_photo_localization: Arc<Mutex<bool>>,
    pub jobs: crate::jobs::JobManager,
}

struct PhotoLocalizationGuard {
    active: Arc<Mutex<bool>>,
}

impl Drop for PhotoLocalizationGuard {
    fn drop(&mut self) {
        if let Ok(mut active) = self.active.lock() {
            *active = false;
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            project: Arc::new(Mutex::new(ProjectSummary::placeholder())),
            active_capture: Arc::new(Mutex::new(None)),
            active_preview: Arc::new(Mutex::new(None)),
            active_photo_localization: Arc::new(Mutex::new(false)),
            jobs: crate::jobs::JobManager::default(),
        }
    }
}

pub fn terminate_active_capture(state: &AppState) {
    state.jobs.cancel_all();
    if let Ok(mut active) = state.active_capture.lock() {
        if let Some(mut capture) = active.take() {
            File::create(capture.phase_root.join("stop.flag")).ok();
            let deadline = Instant::now() + Duration::from_secs(3);
            loop {
                match capture.child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) if Instant::now() < deadline => {
                        thread::sleep(Duration::from_millis(80));
                    }
                    _ => {
                        capture.child.kill().ok();
                        capture.child.wait().ok();
                        break;
                    }
                }
            }
            drain_sensor_relay(&mut capture, Duration::from_secs(1));
            stop_live_reconstruction(&mut capture, Duration::from_secs(2));
            drain_sensor_relay(&mut capture, Duration::from_secs(1));
        }
    }
    if let Ok(mut active) = state.active_preview.lock() {
        if let Some(mut preview) = active.take() {
            shutdown_sensor_session(&mut preview, Duration::from_secs(3));
            discard_preview_files(&preview);
        }
    }
}

fn shutdown_sensor_session(capture: &mut ActiveCapture, timeout: Duration) {
    File::create(capture.phase_root.join("stop.flag")).ok();
    let deadline = Instant::now() + timeout;
    loop {
        match capture.child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(50)),
            _ => {
                capture.child.kill().ok();
                capture.child.wait().ok();
                break;
            }
        }
    }
    drain_sensor_relay(capture, Duration::from_secs(1));
    stop_live_reconstruction(capture, Duration::from_secs(2));
    drain_sensor_relay(capture, Duration::from_secs(1));
}

fn discard_preview_files(preview: &ActiveCapture) {
    let phases_root = preview.project_root.join("phases");
    if preview.phase_root.parent() == Some(phases_root.as_path()) {
        fs::remove_dir_all(&preview.phase_root).ok();
    }
}

fn drain_sensor_relay(capture: &mut ActiveCapture, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while capture
        .sensor_relay
        .as_ref()
        .is_some_and(|relay| !relay.is_finished())
        && Instant::now() < deadline
    {
        thread::sleep(Duration::from_millis(25));
    }
    if capture
        .sensor_relay
        .as_ref()
        .is_some_and(|relay| relay.is_finished())
    {
        if let Some(relay) = capture.sensor_relay.take() {
            relay.join().ok();
        }
    }
}

fn stop_live_reconstruction(capture: &mut ActiveCapture, timeout: Duration) {
    stop_live_reconstruction_child(
        &capture.phase_root,
        &mut capture.live_reconstruction,
        timeout,
    );
}

fn stop_live_reconstruction_child(
    _phase_root: &Path,
    live_reconstruction: &mut Option<Child>,
    timeout: Duration,
) {
    let Some(mut child) = live_reconstruction.take() else {
        return;
    };
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(50)),
            _ => {
                child.kill().ok();
                child.wait().ok();
                break;
            }
        }
    }
}

fn drain_live_reconstruction(capture: &mut ActiveCapture, timeout: Duration) {
    if capture.live_reconstruction.is_none() {
        return;
    }
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if capture
            .live_reconstruction
            .as_mut()
            .and_then(|child| child.try_wait().ok())
            .flatten()
            .is_some()
        {
            break;
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn project_base(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_data_dir()
        .map(|path| path.join("projects"))
        .map_err(|error| error.to_string())
}

fn validated_project_name(name: &str) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("Enter a project name".to_string());
    }
    if name.chars().count() > 80 {
        return Err("Project names cannot be longer than 80 characters".to_string());
    }
    if name.chars().any(char::is_control) {
        return Err("Project names cannot contain control characters".to_string());
    }
    Ok(name.to_string())
}

fn project_catalog_entry(project: ProjectSummary) -> ProjectCatalogEntry {
    let modified_at = fs::metadata(Path::new(&project.path).join("project.json"))
        .and_then(|metadata| metadata.modified())
        .map(|modified| DateTime::<Utc>::from(modified).to_rfc3339())
        .unwrap_or_else(|_| project.created_at.clone());
    let artifact_count = [
        project.artifacts.point_cloud.as_ref(),
        project.artifacts.textured_mesh.as_ref(),
        project.artifacts.gaussian_splat.as_ref(),
    ]
    .into_iter()
    .flatten()
    .filter(|artifact| artifact.status == "ready" && !artifact.stale)
    .count();
    ProjectCatalogEntry {
        id: project.id,
        name: project.name,
        path: project.path,
        created_at: project.created_at,
        modified_at,
        capture_count: project
            .phases
            .iter()
            .filter(|phase| phase.status == "complete")
            .count(),
        media_source_count: project.media_sources.len(),
        frame_count: project
            .phases
            .iter()
            .map(|phase| u64::from(phase.frame_count))
            .sum(),
        artifact_count,
        processing_status: project.processing_status,
    }
}

fn ensure_project_management_idle(state: &AppState) -> Result<(), String> {
    if state
        .active_capture
        .lock()
        .map_err(|_| "Capture state is unavailable".to_string())?
        .is_some()
    {
        return Err("Stop recording before changing projects".to_string());
    }
    if *state
        .active_photo_localization
        .lock()
        .map_err(|_| "Photo localization state is unavailable".to_string())?
    {
        return Err("Wait for photo localization to finish before changing projects".to_string());
    }
    let active_job = state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())?
        .active_job
        .clone();
    if active_job
        .as_deref()
        .is_some_and(|job_id| state.jobs.is_running(job_id))
    {
        return Err("Wait for reconstruction to finish before changing projects".to_string());
    }
    Ok(())
}

fn stop_active_preview(state: &AppState) -> Result<(), String> {
    if let Some(mut preview) = state
        .active_preview
        .lock()
        .map_err(|_| "Preview state is unavailable".to_string())?
        .take()
    {
        shutdown_sensor_session(&mut preview, Duration::from_secs(3));
        discard_preview_files(&preview);
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SensorPreference {
    #[serde(default)]
    sensor_id: String,
    #[serde(default = "default_preferred_sensor_kind")]
    sensor_kind: String,
    #[serde(default = "default_preferred_sensor_connection")]
    sensor_connection: String,
    #[serde(default)]
    sensor_address: String,
    #[serde(default = "default_preferred_use_imu")]
    use_imu: bool,
    #[serde(default = "default_preferred_depth_field_of_view")]
    depth_field_of_view: String,
    #[serde(default)]
    depth_binned: bool,
}

fn default_preferred_sensor_kind() -> String {
    "kinect_v2".to_string()
}

fn default_preferred_sensor_connection() -> String {
    "usb".to_string()
}

fn default_preferred_use_imu() -> bool {
    true
}

fn default_preferred_depth_field_of_view() -> String {
    "narrow".to_string()
}

fn sensor_preference_path(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_data_dir()
        .map(|path| path.join("sensor-preference.json"))
        .map_err(|error| error.to_string())
}

fn read_sensor_preference(app: &AppHandle) -> Option<SensorPreference> {
    let path = sensor_preference_path(app).ok()?;
    serde_json::from_reader(File::open(path).ok()?).ok()
}

fn write_sensor_preference(app: &AppHandle, settings: &CaptureSettings) -> Result<(), String> {
    storage::write_json(
        &sensor_preference_path(app)?,
        &SensorPreference {
            sensor_id: settings.sensor_id.clone(),
            sensor_kind: settings.sensor_kind.clone(),
            sensor_connection: settings.sensor_connection.clone(),
            sensor_address: settings.sensor_address.clone(),
            use_imu: settings.use_imu,
            depth_field_of_view: settings.depth_field_of_view.clone(),
            depth_binned: settings.depth_binned,
        },
    )
}

fn restore_sensor_preference(app: &AppHandle, project: &mut ProjectSummary) -> bool {
    let Some(preference) = read_sensor_preference(app) else {
        return false;
    };
    let changed = project.settings.sensor_id != preference.sensor_id
        || project.settings.sensor_kind != preference.sensor_kind
        || project.settings.sensor_connection != preference.sensor_connection
        || project.settings.sensor_address != preference.sensor_address
        || project.settings.use_imu != preference.use_imu
        || project.settings.depth_field_of_view != preference.depth_field_of_view
        || project.settings.depth_binned != preference.depth_binned;
    if changed {
        project.settings.sensor_id = preference.sensor_id;
        project.settings.sensor_kind = preference.sensor_kind;
        project.settings.sensor_connection = preference.sensor_connection;
        project.settings.sensor_address = preference.sensor_address;
        project.settings.use_imu = preference.use_imu;
        project.settings.depth_field_of_view = preference.depth_field_of_view;
        project.settings.depth_binned = preference.depth_binned;
    }
    changed
}

fn normalize_project(project: &mut ProjectSummary) -> bool {
    let mut changed = false;
    if project.settings.depth_refinement_backend == "off"
        && project.settings.lingbot_depth_refinement
    {
        project.settings.depth_refinement_backend = "lingbot".to_string();
        project.settings.lingbot_depth_refinement = false;
        changed = true;
    }
    if !matches!(
        project.settings.depth_refinement_backend.as_str(),
        "off" | "auto" | "lingbot" | "mapanything" | "da3"
    ) {
        project.settings.depth_refinement_backend = "off".to_string();
        changed = true;
    }
    let safe_voxel_size = project.settings.voxel_size_mm.clamp(1, 40);
    if safe_voxel_size != project.settings.voxel_size_mm {
        project.settings.voxel_size_mm = safe_voxel_size;
        changed = true;
    }
    let safe_live_memory = project.settings.live_map_memory_mib.clamp(256, 4096);
    if safe_live_memory != project.settings.live_map_memory_mib {
        project.settings.live_map_memory_mib = safe_live_memory;
        changed = true;
    }
    if !matches!(
        project.settings.mesh_repair_profile.as_str(),
        "faithful" | "architectural" | "natural"
    ) {
        project.settings.mesh_repair_profile = "faithful".to_string();
        changed = true;
    }
    if project.settings.repair_mesh && project.mesh_repair_report_path.is_none() {
        if let Some(artifact) = project.artifacts.textured_mesh.as_mut() {
            if artifact.status == "ready" && !artifact.stale {
                artifact.status = "stale".to_string();
                artifact.stale = true;
                changed = true;
            }
        }
    }
    if project.processing_status == "failed" {
        if project.processing_error.is_none() {
            project.processing_error = Some(
                "The last reconstruction did not finish. Captured phases can be rebuilt safely."
                    .to_string(),
            );
            changed = true;
        }
        if project.point_count.take().is_some() {
            changed = true;
        }
        if project.output_path.take().is_some() {
            changed = true;
        }
    }
    changed
}

fn ensure_project(app: &AppHandle, state: &AppState) -> Result<ProjectSummary, String> {
    let mut existing = state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())?
        .clone();
    if !existing.path.is_empty() && Path::new(&existing.path).join("project.json").exists() {
        if normalize_project(&mut existing)
            | restore_sensor_preference(app, &mut existing)
            | crate::jobs::recover_interrupted_job(&mut existing, &state.jobs)
        {
            storage::write_project(&existing)?;
            *state
                .project
                .lock()
                .map_err(|_| "Project state is unavailable".to_string())? = existing.clone();
        }
        return Ok(existing);
    }

    let base = project_base(app)?;
    let mut project = if let Some(mut project) = storage::latest_project(&base)? {
        storage::recover_interrupted_phases(&mut project)?;
        project
    } else {
        let folder = format!("scan-{}", Utc::now().format("%Y%m%d-%H%M%S"));
        storage::create_project(&base.join(folder))?
    };
    if normalize_project(&mut project)
        | restore_sensor_preference(app, &mut project)
        | crate::jobs::recover_interrupted_job(&mut project, &state.jobs)
    {
        storage::write_project(&project)?;
    }
    *state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
    Ok(project)
}

pub(crate) fn resource_root(app: &AppHandle) -> Option<PathBuf> {
    app.path().resource_dir().ok()
}

pub(crate) fn first_existing(paths: Vec<PathBuf>) -> Option<PathBuf> {
    paths.into_iter().find(|path| path.is_file())
}

pub(crate) fn worker_command(worker: &Path) -> Command {
    let mut command = Command::new(worker);

    // The capture and reconstruction helpers are console executables so they
    // remain convenient to diagnose directly. When the desktop app owns them,
    // keep their stdio pipes but do not create a visible console window.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    command
}

#[tauri::command]
pub async fn localize_supplemental_photos(
    app: AppHandle,
    project_path: String,
    photo_paths: Vec<String>,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    if photo_paths.is_empty() {
        return Err("Choose at least one high-resolution scene photo".to_string());
    }
    if state
        .active_capture
        .lock()
        .map_err(|_| "Capture state is unavailable".to_string())?
        .is_some()
        || state
            .active_preview
            .lock()
            .map_err(|_| "Preview state is unavailable".to_string())?
            .is_some()
    {
        return Err("Stop RGB-D capture before localizing texture photos".to_string());
    }
    let project_root = PathBuf::from(&project_path);
    let project = storage::read_project(&project_root)?;
    if project.active_job.is_some() {
        return Err("Wait for reconstruction to finish before adding texture photos".to_string());
    }
    let worker = first_existing(storage::candidate_reconstruction_worker_paths(
        resource_root(&app).as_deref(),
    ))
    .ok_or_else(|| "Reconstruction support is missing from this app build".to_string())?;
    {
        let mut active = state
            .active_photo_localization
            .lock()
            .map_err(|_| "Photo localization state is unavailable".to_string())?;
        if *active {
            return Err("Texture-photo localization is already running".to_string());
        }
        *active = true;
    }
    let _localization_guard = PhotoLocalizationGuard {
        active: Arc::clone(&state.active_photo_localization),
    };
    fs::remove_file(
        project_root
            .join("outputs")
            .join("photo-localization-progress.json"),
    )
    .ok();
    let worker_project = project_root.clone();
    let output_result = tauri::async_runtime::spawn_blocking(move || {
        let mut command = worker_command(&worker);
        command
            .arg("localize-photos")
            .arg(&worker_project)
            .args(photo_paths)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command.output().map_err(|error| error.to_string())
    })
    .await;
    let output = output_result.map_err(|error| error.to_string())??;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if detail.is_empty() {
            "The texture photos could not be localized".to_string()
        } else {
            detail
        });
    }
    let result: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(|error| error.to_string())?;
    if result
        .get("localizedPhotoCount")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
        > 0
    {
        let mut updated = storage::read_project(&project_root)?;
        if let Some(mesh) = updated.artifacts.textured_mesh.as_mut() {
            mesh.stale = true;
        }
        storage::write_project(&updated)?;
        *state
            .project
            .lock()
            .map_err(|_| "Project state is unavailable".to_string())? = updated;
    }
    Ok(result)
}

fn read_supplemental_photo_manifest(project_root: &Path) -> Result<serde_json::Value, String> {
    storage::read_project(project_root)?;
    let manifest_path = project_root.join("supplemental-photos.json");
    if !manifest_path.is_file() {
        return Ok(serde_json::json!({ "schemaVersion": 1, "photos": [], "attempts": [] }));
    }
    let file = File::open(&manifest_path).map_err(|error| error.to_string())?;
    let mut manifest: serde_json::Value =
        serde_json::from_reader(file).map_err(|error| error.to_string())?;
    if manifest
        .get("schemaVersion")
        .and_then(serde_json::Value::as_u64)
        != Some(1)
    {
        return Err("The supplemental-photo manifest uses an unsupported schema".to_string());
    }
    let photos = manifest
        .get("photos")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut attempts = manifest
        .get("attempts")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for photo in &photos {
        let Some(photo_id) = photo.get("id").and_then(serde_json::Value::as_str) else {
            continue;
        };
        if attempts
            .iter()
            .any(|attempt| attempt.get("id").and_then(serde_json::Value::as_str) == Some(photo_id))
        {
            continue;
        }
        let mut attempt = photo.clone();
        if let Some(object) = attempt.as_object_mut() {
            object.insert(
                "status".to_string(),
                serde_json::Value::String("localized".to_string()),
            );
        }
        attempts.push(attempt);
    }
    manifest["photos"] = serde_json::Value::Array(photos);
    manifest["attempts"] = serde_json::Value::Array(attempts);
    Ok(manifest)
}

#[tauri::command]
pub fn supplemental_photos(project_path: String) -> Result<serde_json::Value, String> {
    read_supplemental_photo_manifest(Path::new(&project_path))
}

#[tauri::command]
pub fn supplemental_photo_progress(
    project_path: String,
    state: State<'_, AppState>,
) -> Result<Option<serde_json::Value>, String> {
    let project_root = PathBuf::from(project_path);
    storage::read_project(&project_root)?;
    let progress_path = project_root
        .join("outputs")
        .join("photo-localization-progress.json");
    if !progress_path.is_file() {
        return Ok(None);
    }
    let file = File::open(progress_path).map_err(|error| error.to_string())?;
    let mut progress: serde_json::Value =
        serde_json::from_reader(file).map_err(|error| error.to_string())?;
    if progress
        .get("schemaVersion")
        .and_then(serde_json::Value::as_u64)
        != Some(1)
    {
        return Err("The texture-photo progress file uses an unsupported schema".to_string());
    }
    let active = *state
        .active_photo_localization
        .lock()
        .map_err(|_| "Photo localization state is unavailable".to_string())?;
    if progress.get("status").and_then(serde_json::Value::as_str) == Some("running") && !active {
        progress["status"] = serde_json::Value::String("failed".to_string());
        progress["stage"] = serde_json::Value::String("interrupted".to_string());
        progress["detail"] = serde_json::Value::String(
            "The previous photo-localization worker stopped before finishing. Completed photos were kept."
                .to_string(),
        );
    }
    Ok(Some(progress))
}

#[tauri::command]
pub fn remove_supplemental_photo(
    project_path: String,
    photo_id: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    if photo_id.is_empty()
        || photo_id.len() > 128
        || !photo_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err("The supplemental photo id is invalid".to_string());
    }
    if *state
        .active_photo_localization
        .lock()
        .map_err(|_| "Photo localization state is unavailable".to_string())?
    {
        return Err("Wait for photo localization to finish before removing a photo".to_string());
    }
    let project_root = PathBuf::from(&project_path);
    let mut manifest = read_supplemental_photo_manifest(&project_root)?;
    let photos = manifest["photos"]
        .as_array_mut()
        .ok_or_else(|| "The supplemental-photo manifest is invalid".to_string())?;
    let stored_path = photos
        .iter()
        .find(|photo| {
            photo.get("id").and_then(serde_json::Value::as_str) == Some(photo_id.as_str())
        })
        .and_then(|photo| photo.get("path"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let removed_localized = stored_path.is_some();
    photos.retain(|photo| {
        photo.get("id").and_then(serde_json::Value::as_str) != Some(photo_id.as_str())
    });
    manifest["attempts"]
        .as_array_mut()
        .ok_or_else(|| "The supplemental-photo manifest is invalid".to_string())?
        .retain(|photo| {
            photo.get("id").and_then(serde_json::Value::as_str) != Some(photo_id.as_str())
        });
    storage::write_json(&project_root.join("supplemental-photos.json"), &manifest)?;

    if let Some(relative_path) = stored_path {
        let relative = Path::new(&relative_path);
        let is_safe = !relative.is_absolute()
            && relative
                .components()
                .all(|component| matches!(component, std::path::Component::Normal(_)));
        if is_safe {
            fs::remove_file(project_root.join(relative)).ok();
        }
    }
    if removed_localized {
        let mut project = storage::read_project(&project_root)?;
        if let Some(mesh) = project.artifacts.textured_mesh.as_mut() {
            mesh.stale = true;
        }
        storage::write_project(&project)?;
        *state
            .project
            .lock()
            .map_err(|_| "Project state is unavailable".to_string())? = project;
    }
    Ok(manifest)
}

#[tauri::command]
pub fn import_media_sources(
    project_path: String,
    media_paths: Vec<String>,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    if media_paths.is_empty() {
        return Err("Choose at least three photos or one video".to_string());
    }
    if media_paths.len() > 5_000 {
        return Err("Import at most 5,000 media files at once".to_string());
    }
    if state
        .active_capture
        .lock()
        .map_err(|_| "Capture state is unavailable".to_string())?
        .is_some()
        || state
            .active_preview
            .lock()
            .map_err(|_| "Preview state is unavailable".to_string())?
            .is_some()
    {
        return Err("Stop the camera before importing photos or video".to_string());
    }
    let root = PathBuf::from(&project_path);
    let mut project = storage::read_project(&root)?;
    if project.active_job.is_some() || project.processing_status == "processing" {
        return Err("Wait for reconstruction to finish before importing media".to_string());
    }
    let active_path = state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())?
        .path
        .clone();
    if active_path != project.path {
        return Err("The selected project is no longer active".to_string());
    }
    let photo_extensions = ["jpg", "jpeg", "png", "tif", "tiff", "webp", "bmp"];
    let video_extensions = ["mp4", "mov", "m4v", "avi", "mkv", "webm", "mts", "m2ts"];
    let mut validated = Vec::with_capacity(media_paths.len());
    for value in media_paths {
        let source = PathBuf::from(&value);
        if !source.is_file() {
            return Err(format!("Media source is missing: {}", source.display()));
        }
        let extension = source
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let kind = if photo_extensions.contains(&extension.as_str()) {
            "photo"
        } else if video_extensions.contains(&extension.as_str()) {
            "video"
        } else {
            return Err(format!("Unsupported photo or video: {}", source.display()));
        };
        validated.push((source, extension, kind));
    }
    let media_root = root.join("media");
    fs::create_dir_all(&media_root).map_err(|error| error.to_string())?;
    let mut imported = Vec::new();
    let mut copied_paths = Vec::new();
    for (source, extension, kind) in validated {
        let id = Uuid::new_v4().to_string();
        let relative = PathBuf::from("media").join(format!("{id}.{extension}"));
        let destination = root.join(&relative);
        let temporary = media_root.join(format!(".{id}.importing"));
        let copied = (|| {
            fs::copy(&source, &temporary).map_err(|error| error.to_string())?;
            fs::rename(&temporary, &destination).map_err(|error| error.to_string())
        })();
        if let Err(error) = copied {
            fs::remove_file(&temporary).ok();
            for copied in copied_paths {
                fs::remove_file(copied).ok();
            }
            return Err(error);
        }
        copied_paths.push(destination.clone());
        let byte_size = destination
            .metadata()
            .map_err(|error| error.to_string())?
            .len();
        imported.push(MediaSourceSummary {
            id,
            name: source
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("media")
                .to_string(),
            path: relative.to_string_lossy().into_owned(),
            kind: kind.to_string(),
            byte_size,
            created_at: Utc::now().to_rfc3339(),
        });
    }
    project.media_sources.extend(imported);
    mark_all_artifacts_stale(&mut project);
    project.processing_status = "idle".to_string();
    project.processing_error = None;
    if let Err(error) = storage::write_project(&project) {
        for copied in copied_paths {
            fs::remove_file(copied).ok();
        }
        return Err(error);
    }
    *state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
    Ok(project)
}

fn mark_all_artifacts_stale(project: &mut ProjectSummary) {
    for artifact in [
        &mut project.artifacts.point_cloud,
        &mut project.artifacts.textured_mesh,
        &mut project.artifacts.gaussian_splat,
    ]
    .into_iter()
    .flatten()
    {
        artifact.stale = true;
        artifact.status = "stale".to_string();
    }
}

fn managed_media_source_path(project_root: &Path, relative_path: &str) -> Result<PathBuf, String> {
    let relative = Path::new(relative_path);
    let components = relative.components().collect::<Vec<_>>();
    let is_managed_media_file = matches!(
        components.as_slice(),
        [std::path::Component::Normal(directory), std::path::Component::Normal(_file)]
            if *directory == std::ffi::OsStr::new("media")
    );
    if relative.is_absolute() || !is_managed_media_file {
        return Err("The imported media source has an unsafe managed path".to_string());
    }
    Ok(project_root.join(relative))
}

#[tauri::command]
pub fn remove_media_source(
    project_path: String,
    media_source_id: String,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    if state
        .active_capture
        .lock()
        .map_err(|_| "Capture state is unavailable".to_string())?
        .is_some()
    {
        return Err("Stop the active capture before removing imported media".to_string());
    }

    let project_root = PathBuf::from(&project_path);
    let mut project = storage::read_project(&project_root)?;
    if project.active_job.is_some() || project.processing_status == "processing" {
        return Err("Wait for reconstruction to finish before removing imported media".to_string());
    }
    let active_path = state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())?
        .path
        .clone();
    if active_path != project.path {
        return Err("The selected project is no longer active".to_string());
    }

    let source_index = project
        .media_sources
        .iter()
        .position(|source| source.id == media_source_id)
        .ok_or_else(|| "The imported media source no longer exists".to_string())?;
    let source_path =
        managed_media_source_path(&project_root, &project.media_sources[source_index].path)?;
    let temporary = project_root
        .join("media")
        .join(format!(".{}.removing", Uuid::new_v4()));
    let moved_source = if source_path.is_file() {
        fs::rename(&source_path, &temporary).map_err(|error| {
            format!("Could not prepare the imported media for removal: {error}")
        })?;
        true
    } else if source_path.exists() {
        return Err("The imported media path is not a file".to_string());
    } else {
        false
    };

    project.media_sources.remove(source_index);
    mark_all_artifacts_stale(&mut project);
    project.processing_status = "idle".to_string();
    project.processing_error = None;
    if let Err(error) = storage::write_project(&project) {
        if moved_source {
            fs::rename(&temporary, &source_path).ok();
        }
        return Err(error);
    }
    if moved_source {
        fs::remove_file(&temporary).ok();
    }
    *state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
    Ok(project)
}

fn worker_capabilities(worker: &Path) -> Vec<String> {
    let Ok(output) = worker_command(worker).arg("--capabilities").output() else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    serde_json::from_slice(&output.stdout).unwrap_or_default()
}

fn first_sensor_worker(paths: Vec<PathBuf>, sensor_kind: &str) -> Option<PathBuf> {
    paths.into_iter().find(|path| {
        path.is_file()
            && worker_capabilities(path)
                .iter()
                .any(|capability| capability == sensor_kind)
    })
}

fn first_modern_sensor_worker(paths: Vec<PathBuf>) -> Option<PathBuf> {
    paths.into_iter().find(|path| {
        if !path.is_file() {
            return false;
        }
        worker_capabilities(path)
            .iter()
            .any(|capability| matches!(capability.as_str(), "azure_kinect" | "femto_mega"))
    })
}

fn installed_sensor_capabilities(resource_root: Option<&Path>) -> Vec<String> {
    let mut capabilities = Vec::new();
    let workers = storage::candidate_kinect_worker_paths(resource_root)
        .into_iter()
        .chain(storage::candidate_modern_sensor_worker_paths(resource_root));
    for worker in workers.filter(|path| path.is_file()) {
        for capability in worker_capabilities(&worker) {
            if matches!(
                capability.as_str(),
                "kinect_v2" | "azure_kinect" | "femto_mega"
            ) && !capabilities.contains(&capability)
            {
                capabilities.push(capability);
            }
        }
    }
    capabilities
}

fn sensor_name(settings: &CaptureSettings) -> &'static str {
    match settings.sensor_kind.as_str() {
        "azure_kinect" => "Azure Kinect DK",
        "femto_mega" => "Orbbec Femto Mega",
        _ => "Kinect v2",
    }
}

fn validate_sensor_settings(settings: &mut CaptureSettings) -> Result<(), String> {
    settings.rgb_jpeg_quality = settings.rgb_jpeg_quality.clamp(60, 100);
    settings.max_rgb_dimension = if settings.max_rgb_dimension == 0 {
        0
    } else {
        settings.max_rgb_dimension.clamp(640, 8192)
    };
    settings.sensor_id = settings.sensor_id.trim().to_string();
    if !matches!(
        settings.rgb_resolution.as_str(),
        "auto" | "720p" | "1080p" | "1440p" | "1536p" | "2160p" | "3072p"
    ) {
        return Err("Unknown RGB resolution".to_string());
    }
    settings.rgb_exposure_us = settings.rgb_exposure_us.clamp(100, 200_000);
    settings.rgb_gain = settings.rgb_gain.clamp(0, 255);
    settings.rgb_white_balance_k =
        ((settings.rgb_white_balance_k.clamp(2_000, 12_500) + 5) / 10) * 10;
    settings.rgb_brightness = settings.rgb_brightness.clamp(0, 255);
    settings.rgb_contrast = settings.rgb_contrast.clamp(0, 60);
    settings.rgb_saturation = settings.rgb_saturation.clamp(0, 80);
    settings.rgb_sharpness = settings.rgb_sharpness.clamp(0, 15);
    if !matches!(settings.rgb_powerline_hz, 0 | 50 | 60) {
        return Err(
            "RGB anti-flicker frequency must be camera default, 50 Hz, or 60 Hz".to_string(),
        );
    }
    if !matches!(
        settings.imu_accel_rate_hz,
        0 | 50 | 100 | 200 | 500 | 1_000 | 2_000
    ) {
        return Err("Unsupported accelerometer sample rate".to_string());
    }
    if !matches!(settings.imu_accel_range_g, 0 | 2 | 4 | 8 | 16) {
        return Err("Unsupported accelerometer range".to_string());
    }
    if !matches!(
        settings.imu_gyro_rate_hz,
        0 | 50 | 100 | 200 | 500 | 1_000 | 2_000
    ) {
        return Err("Unsupported gyroscope sample rate".to_string());
    }
    if !matches!(
        settings.imu_gyro_range_dps,
        0 | 125 | 250 | 500 | 1_000 | 2_000
    ) {
        return Err("Unsupported gyroscope range".to_string());
    }
    if !matches!(
        settings.sensor_kind.as_str(),
        "kinect_v2" | "azure_kinect" | "femto_mega"
    ) {
        return Err("Unknown depth sensor".to_string());
    }
    if !matches!(settings.sensor_connection.as_str(), "usb" | "network") {
        return Err("Unknown sensor connection type".to_string());
    }
    if !matches!(settings.depth_field_of_view.as_str(), "narrow" | "wide") {
        return Err("Depth field of view must be narrow or wide".to_string());
    }
    if !matches!(settings.sensor_fps, 0 | 5 | 15 | 25 | 30) {
        return Err("Sensor rate must be automatic, 5, 15, 25, or 30 fps".to_string());
    }
    if settings.sensor_kind == "azure_kinect" && settings.sensor_fps == 25 {
        return Err("Azure Kinect supports 5, 15, or 30 fps sensor rates".to_string());
    }
    if settings.depth_field_of_view == "wide" && !settings.depth_binned && settings.sensor_fps > 15
    {
        return Err("Wide full-resolution depth supports at most 15 fps".to_string());
    }
    let effective_sensor_fps = if settings.sensor_fps > 0 {
        settings.sensor_fps
    } else if settings.depth_field_of_view == "wide" && !settings.depth_binned {
        15
    } else {
        30
    };
    let exposure_limit_us = match effective_sensor_fps {
        0..=5 => 190_000,
        6..=15 => 60_000,
        _ => 30_000,
    };
    settings.rgb_exposure_us = settings.rgb_exposure_us.min(exposure_limit_us);
    if settings.sensor_kind != "femto_mega" && settings.sensor_connection == "network" {
        return Err("Network capture is currently supported only by Orbbec Femto Mega".to_string());
    }
    settings.sensor_address = settings.sensor_address.trim().to_string();
    if settings.sensor_kind == "femto_mega"
        && settings.sensor_connection == "network"
        && settings.sensor_address.is_empty()
    {
        return Err(
            "Enter the Femto Mega IP address before connecting over the network".to_string(),
        );
    }
    if settings.sensor_kind == "kinect_v2" {
        settings.sensor_connection = "usb".to_string();
        settings.sensor_address.clear();
        settings.use_imu = false;
    }
    if settings.sensor_kind == "azure_kinect"
        && matches!(settings.rgb_resolution.as_str(), "2160p" | "3072p")
        && effective_sensor_fps > 15
    {
        return Err("Azure Kinect 2160p/3072p RGB requires a 5 or 15 fps sensor rate".to_string());
    }
    if settings.sensor_kind == "femto_mega" && settings.rgb_resolution == "3072p" {
        return Err("Femto Mega does not expose a 4096x3072 RGB mode".to_string());
    }
    if settings.sensor_fps > 0 {
        settings.capture_fps = settings.capture_fps.min(settings.sensor_fps);
    }
    if !matches!(settings.live_reconstruction.as_str(), "points" | "mesh") {
        return Err("Unknown live reconstruction mode".to_string());
    }
    settings.live_map_memory_mib = settings.live_map_memory_mib.clamp(256, 4096);
    if !matches!(
        settings.mesh_repair_profile.as_str(),
        "faithful" | "architectural" | "natural"
    ) {
        return Err("Unknown mesh repair profile".to_string());
    }
    if !matches!(
        settings.depth_refinement_backend.as_str(),
        "off" | "auto" | "lingbot" | "mapanything" | "da3"
    ) {
        return Err("Unknown depth refinement backend".to_string());
    }
    Ok(())
}

fn sensor_worker(app: &AppHandle, settings: &CaptureSettings) -> Result<PathBuf, String> {
    let resources = resource_root(app);
    let candidates = if settings.sensor_kind == "kinect_v2" {
        storage::candidate_kinect_worker_paths(resources.as_deref())
    } else {
        storage::candidate_modern_sensor_worker_paths(resources.as_deref())
    };
    first_sensor_worker(candidates, &settings.sensor_kind).ok_or_else(|| {
        format!(
            "{} capture support is missing from this app build",
            sensor_name(settings)
        )
    })
}

fn start_realtime_engine(
    app: &AppHandle,
    phase_root: &Path,
    settings: &CaptureSettings,
) -> Result<
    (
        Child,
        std::process::ChildStdin,
        Arc<Mutex<RealtimeEngineSnapshot>>,
    ),
    String,
> {
    let worker = first_existing(storage::candidate_reconstruction_worker_paths(
        resource_root(app).as_deref(),
    ))
    .ok_or_else(|| "Realtime reconstruction support is missing from this app build".to_string())?;
    let stderr = File::create(phase_root.join("live-reconstruction.log"))
        .map_err(|error| error.to_string())?;
    let live_voxel_size_m = (settings.voxel_size_mm.clamp(5, 40) as f32) / 1000.0;
    let mut command = worker_command(&worker);
    command
        .arg("realtime")
        .arg("--mode")
        .arg(&settings.live_reconstruction)
        .arg("--voxel-size")
        .arg(live_voxel_size_m.to_string())
        .arg("--live-map-mib")
        .arg(settings.live_map_memory_mib.to_string())
        .arg("--session")
        .arg(phase_root)
        .arg("--sensor-kind")
        .arg(&settings.sensor_kind)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::from(stderr));
    let mut child = command
        .spawn()
        .map_err(|error| format!("Could not start realtime reconstruction: {error}"))?;
    let input = match child.stdin.take() {
        Some(input) => input,
        None => {
            child.kill().ok();
            child.wait().ok();
            return Err("Could not open the realtime engine input".to_string());
        }
    };
    let output = match child.stdout.take() {
        Some(output) => output,
        None => {
            drop(input);
            child.kill().ok();
            child.wait().ok();
            return Err("Could not open the realtime engine output".to_string());
        }
    };
    let snapshot = Arc::new(Mutex::new(RealtimeEngineSnapshot::default()));
    let reader_snapshot = Arc::clone(&snapshot);
    let mode = settings.live_reconstruction.clone();
    thread::spawn(move || read_realtime_engine_stream(output, reader_snapshot, mode));
    Ok((child, input, snapshot))
}

fn wait_for_realtime_engine_ready(
    child: &mut Child,
    snapshot: &Arc<Mutex<RealtimeEngineSnapshot>>,
    timeout: Duration,
) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(latest) = snapshot.lock() {
            if latest.status.active && latest.status.tracking_status.contains("ready") {
                return Ok(());
            }
            if let Some(error) = &latest.error {
                return Err(error.clone());
            }
        }
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            return Err(format!(
                "Realtime reconstruction stopped while warming up ({status})"
            ));
        }
        if Instant::now() >= deadline {
            child.kill().ok();
            child.wait().ok();
            return Err(
                "Realtime reconstruction did not become ready within 45 seconds; inspect live-reconstruction.log"
                    .to_string(),
            );
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn append_sensor_args(command: &mut Command, settings: &CaptureSettings) {
    if settings.sensor_kind == "kinect_v2" {
        return;
    }
    command
        .arg("--rgb-quality")
        .arg(settings.rgb_jpeg_quality.to_string());
    if settings.max_rgb_dimension > 0 {
        command
            .arg("--max-rgb-dimension")
            .arg(settings.max_rgb_dimension.to_string());
    }
    command
        .arg("--sensor")
        .arg(&settings.sensor_kind)
        .arg("--connection")
        .arg(&settings.sensor_connection)
        .arg("--depth-fov")
        .arg(&settings.depth_field_of_view)
        .arg("--sensor-fps")
        .arg(settings.sensor_fps.to_string())
        .arg("--rgb-resolution")
        .arg(&settings.rgb_resolution)
        .arg("--rgb-auto-exposure")
        .arg(settings.rgb_auto_exposure.to_string())
        .arg("--rgb-exposure-us")
        .arg(settings.rgb_exposure_us.to_string())
        .arg("--rgb-gain")
        .arg(settings.rgb_gain.to_string())
        .arg("--rgb-auto-white-balance")
        .arg(settings.rgb_auto_white_balance.to_string())
        .arg("--rgb-white-balance-k")
        .arg(settings.rgb_white_balance_k.to_string())
        .arg("--rgb-color-adjustments")
        .arg(settings.rgb_color_adjustments_enabled.to_string())
        .arg("--rgb-brightness")
        .arg(settings.rgb_brightness.to_string())
        .arg("--rgb-contrast")
        .arg(settings.rgb_contrast.to_string())
        .arg("--rgb-saturation")
        .arg(settings.rgb_saturation.to_string())
        .arg("--rgb-sharpness")
        .arg(settings.rgb_sharpness.to_string())
        .arg("--rgb-backlight-compensation")
        .arg(settings.rgb_backlight_compensation.to_string())
        .arg("--rgb-powerline-hz")
        .arg(settings.rgb_powerline_hz.to_string());
    if settings.depth_binned {
        command.arg("--depth-binned");
    }
    if !settings.sensor_id.is_empty() {
        command.arg("--device").arg(&settings.sensor_id);
    }
    if !settings.sensor_address.is_empty() {
        command.arg("--address").arg(&settings.sensor_address);
    }
    if settings.use_imu {
        command
            .arg("--imu")
            .arg("--imu-accel-rate")
            .arg(settings.imu_accel_rate_hz.to_string())
            .arg("--imu-accel-range")
            .arg(settings.imu_accel_range_g.to_string())
            .arg("--imu-gyro-rate")
            .arg(settings.imu_gyro_rate_hz.to_string())
            .arg("--imu-gyro-range")
            .arg(settings.imu_gyro_range_dps.to_string());
    }
}

fn output_message(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !stderr.is_empty() {
        return stderr;
    }
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn parse_available_sensors(output: &std::process::Output) -> Vec<AvailableSensor> {
    if !output.status.success() {
        return Vec::new();
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with("[{\"id\":"))
        .and_then(|line| serde_json::from_str(line.trim()).ok())
        .unwrap_or_default()
}

fn read_sensor_log(phase_root: &Path) -> String {
    fs::read_to_string(phase_root.join("sensor.log"))
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn indexed_frame_count(phase_root: &Path) -> u32 {
    fs::read_to_string(phase_root.join("frames.csv"))
        .map(|index| {
            index
                .lines()
                .skip(1)
                .filter(|line| !line.trim().is_empty())
                .count() as u32
        })
        .unwrap_or(0)
}

fn reconstruction_progress(project_root: &Path) -> Option<ReconstructionProgress> {
    let path = project_root.join("outputs").join("progress.json");
    serde_json::from_reader(File::open(path).ok()?).ok()
}

fn live_worker_status(root: &Path) -> Option<LiveWorkerStatus> {
    let path = root.join("live.json");
    let modified = fs::metadata(&path).ok()?.modified().ok()?;
    if modified.elapsed().ok()? > Duration::from_secs(3) {
        return None;
    }
    serde_json::from_reader(File::open(path).ok()?).ok()
}

fn read_realtime_engine_stream(
    stdout: std::process::ChildStdout,
    latest: Arc<Mutex<RealtimeEngineSnapshot>>,
    mode: String,
) {
    const HEADER_SIZE: usize = 24;
    const MAX_PAYLOAD_SIZE: usize = 128 * 1024 * 1024;
    let result = (|| -> Result<(), String> {
        let mut reader = BufReader::new(stdout);
        loop {
            let mut header = [0_u8; HEADER_SIZE];
            match reader.read_exact(&mut header) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(error) => {
                    return Err(format!("Could not read realtime engine output: {error}"))
                }
            }
            if &header[0..8] != b"SCANENG1" {
                return Err("Realtime engine emitted an unknown protocol".to_string());
            }
            let version = u16::from_le_bytes(header[8..10].try_into().unwrap());
            let kind = u16::from_le_bytes(header[10..12].try_into().unwrap());
            let payload_size = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
            if version != 1 || payload_size > MAX_PAYLOAD_SIZE {
                return Err("Realtime engine emitted an unsupported message".to_string());
            }
            let mut payload = vec![0_u8; payload_size];
            reader
                .read_exact(&mut payload)
                .map_err(|error| format!("Realtime engine message was truncated: {error}"))?;

            let mut snapshot = latest
                .lock()
                .map_err(|_| "Realtime engine state is unavailable".to_string())?;
            snapshot.updated = Some(Instant::now());
            match kind {
                1 => {
                    let message: RealtimeEngineStatusMessage = serde_json::from_slice(&payload)
                        .map_err(|error| format!("Realtime engine status is invalid: {error}"))?;
                    let tracking_state = if message.tracking_state.is_empty() {
                        message.state.clone()
                    } else {
                        message.tracking_state.clone()
                    };
                    snapshot.status = LiveReconstructionStatus {
                        contract_version: message.contract_version,
                        active: message.active,
                        mode: mode.clone(),
                        tracking: matches!(
                            tracking_state.as_str(),
                            "tracking" | "relocalized" | "frozen"
                        ),
                        tracking_status: if message.detail.is_empty() {
                            tracking_state.clone()
                        } else {
                            message.detail
                        },
                        tracking_state,
                        tracking_confidence: message.tracking_confidence,
                        processed_frames: message.processed_frames,
                        accepted_frames: message.accepted_frames,
                        integrated_frames: message.integrated_frames,
                        rejected_frames: message.rejected_frames,
                        point_count: message.point_count,
                        triangle_count: message.triangle_count,
                        backend: message.backend,
                        tracking_fps: message.tracking_fps,
                        source_drops: message.source_drops,
                        tracking_queue_drops: message.tracking_queue_drops,
                        mapping_drops: message.mapping_drops,
                        overlap: message.overlap,
                        depth_rmse_mm: message.depth_rmse_mm,
                        pose_uncertainty_mm: message.pose_uncertainty_mm,
                        pose_uncertainty_degrees: message.pose_uncertainty_degrees,
                        pose_latency_ms: message.pose_latency_ms,
                        map_update_latency_ms: message.map_update_latency_ms,
                        map_update_hz: message.map_update_hz,
                        allocated_live_map_bytes: message.allocated_live_map_bytes,
                        active_voxel_count: message.active_voxel_count,
                        active_surfel_count: message.active_surfel_count,
                        resident_submap_count: message.resident_submap_count,
                        host_cached_submap_count: message.host_cached_submap_count,
                        dropped_preview_jobs: message.dropped_preview_jobs,
                        tracking_queue_depth: message.tracking_queue_depth,
                        mapping_queue_depth: message.mapping_queue_depth,
                        degradation_level: message.degradation_level,
                        loop_closure_count: message.loop_closure_count,
                        loop_correction_active: message.loop_correction_active,
                        scale_status: message.scale_status,
                        integration_frozen: message.integration_frozen,
                    };
                }
                2 | 4 | 7 | 8 => {
                    if payload.len() < 24 || &payload[0..4] != b"K2P1" {
                        return Err("Realtime point packet has an invalid header".to_string());
                    }
                    let frame_count = u32::from_le_bytes(payload[4..8].try_into().unwrap());
                    let point_count =
                        u32::from_le_bytes(payload[20..24].try_into().unwrap()) as usize;
                    let expected = 24_usize
                        .checked_add(point_count.saturating_mul(15))
                        .ok_or_else(|| "Realtime point packet is too large".to_string())?;
                    if point_count > 150_000 || payload.len() != expected {
                        return Err("Realtime point packet is incomplete".to_string());
                    }
                    let frame = LiveGeometryFrame {
                        frame_count,
                        packet: Arc::new(payload),
                    };
                    if kind == 2 {
                        snapshot.status.point_count = point_count as u64;
                        snapshot.points = Some(frame);
                    } else if kind == 4 {
                        snapshot.camera_points = Some(frame);
                    } else if kind == 7 {
                        snapshot.coverage_points = Some(frame);
                    } else {
                        snapshot.tracking_points = Some(frame);
                    }
                }
                3 => {
                    if payload.len() < 16 || &payload[0..4] != b"K2M2" {
                        return Err("Realtime mesh packet has an invalid header".to_string());
                    }
                    let frame_count = u32::from_le_bytes(payload[4..8].try_into().unwrap());
                    let vertex_count =
                        u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
                    let index_count =
                        u32::from_le_bytes(payload[12..16].try_into().unwrap()) as usize;
                    let expected = 16_usize
                        .checked_add(vertex_count.saturating_mul(15))
                        .and_then(|size| size.checked_add(index_count.saturating_mul(4)))
                        .ok_or_else(|| "Realtime mesh packet is too large".to_string())?;
                    if vertex_count > 500_000
                        || index_count > 450_000
                        || index_count % 3 != 0
                        || payload.len() != expected
                    {
                        return Err("Realtime mesh packet is incomplete".to_string());
                    }
                    snapshot.status.triangle_count = (index_count / 3) as u64;
                    snapshot.mesh = Some(LiveGeometryFrame {
                        frame_count,
                        packet: Arc::new(payload),
                    });
                }
                5 | 6 => {
                    let message: serde_json::Value =
                        serde_json::from_slice(&payload).map_err(|error| {
                            format!("Realtime contract message is invalid: {error}")
                        })?;
                    if message
                        .get("contractVersion")
                        .and_then(|value| value.as_u64())
                        != Some(2)
                    {
                        return Err(
                            "Realtime contract message has an unsupported version".to_string()
                        );
                    }
                    if kind == 5 {
                        snapshot.coverage = Some(message);
                    } else {
                        snapshot.submaps = Some(message);
                    }
                }
                _ => return Err("Realtime engine emitted an unknown message kind".to_string()),
            }
        }
    })();

    if let Ok(mut snapshot) = latest.lock() {
        snapshot.status.active = false;
        if let Err(error) = result {
            snapshot.status.tracking = false;
            snapshot.status.tracking_status = error.clone();
            snapshot.error = Some(error);
        }
    }
}

fn realtime_packet(
    snapshot: &Arc<Mutex<RealtimeEngineSnapshot>>,
    geometry: RealtimeGeometry,
    after_frame: u32,
) -> Vec<u8> {
    let Ok(snapshot) = snapshot.lock() else {
        return Vec::new();
    };
    let geometry = match geometry {
        RealtimeGeometry::CameraPoints => snapshot.camera_points.as_ref(),
        RealtimeGeometry::FusedPoints => snapshot.points.as_ref(),
        RealtimeGeometry::CoveragePoints => snapshot.coverage_points.as_ref(),
        RealtimeGeometry::TrackingPoints => snapshot.tracking_points.as_ref(),
        RealtimeGeometry::Mesh => snapshot.mesh.as_ref(),
    };
    match geometry {
        Some(frame) if frame.frame_count != after_frame => frame.packet.as_ref().clone(),
        _ => Vec::new(),
    }
}

const LIVE_CAPTURE_PREVIEW_FILE: &str = "live-reconstruction.preview.bin";
const LIVE_ARTIFACT_DIRECTORY: &str = "live";

fn valid_packed_point_preview(packet: &[u8]) -> bool {
    if packet.len() < 24 || &packet[0..4] != b"K2P1" {
        return false;
    }
    let Ok(count_bytes) = packet[20..24].try_into() else {
        return false;
    };
    let point_count = u32::from_le_bytes(count_bytes) as usize;
    point_count <= 150_000 && packet.len() == 24 + point_count * 15
}

fn packed_point_preview_to_ply(packet: &[u8]) -> Result<Vec<u8>, String> {
    if !valid_packed_point_preview(packet) {
        return Err("Realtime point preview is incomplete".to_string());
    }
    let point_count = u32::from_le_bytes(packet[20..24].try_into().unwrap());
    let header = format!(
        concat!(
            "ply\n",
            "format binary_little_endian 1.0\n",
            "comment ScanLan provisional live reconstruction\n",
            "element vertex {}\n",
            "property float x\nproperty float y\nproperty float z\n",
            "property uchar red\nproperty uchar green\nproperty uchar blue\n",
            "end_header\n"
        ),
        point_count
    );
    let mut output = Vec::with_capacity(header.len() + packet.len() - 24);
    output.extend_from_slice(header.as_bytes());
    output.extend_from_slice(&packet[24..]);
    Ok(output)
}

fn packed_point_preview_to_glb(packet: &[u8]) -> Result<Vec<u8>, String> {
    if !valid_packed_point_preview(packet) {
        return Err("Realtime point preview is incomplete".to_string());
    }
    let point_count = u32::from_le_bytes(packet[20..24].try_into().unwrap()) as usize;
    let mut positions = Vec::with_capacity(point_count * 12);
    let mut colors = Vec::with_capacity(point_count * 3 + 3);
    let mut minimum = [f32::INFINITY; 3];
    let mut maximum = [f32::NEG_INFINITY; 3];
    for record in packet[24..].chunks_exact(15) {
        positions.extend_from_slice(&record[..12]);
        colors.extend_from_slice(&record[12..15]);
        for axis in 0..3 {
            let start = axis * 4;
            let value = f32::from_le_bytes(record[start..start + 4].try_into().unwrap());
            minimum[axis] = minimum[axis].min(value);
            maximum[axis] = maximum[axis].max(value);
        }
    }
    while colors.len() % 4 != 0 {
        colors.push(0);
    }
    if point_count == 0 {
        minimum = [0.0; 3];
        maximum = [0.0; 3];
    }
    let position_bytes = positions.len();
    let binary_bytes = position_bytes + colors.len();
    let document = serde_json::json!({
        "asset": {"version": "2.0", "generator": "ScanLan Reconstruction 2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"name": "Provisional live map", "primitives": [{
            "attributes": {"POSITION": 0, "COLOR_0": 1},
            "mode": 0
        }]}],
        "buffers": [{"byteLength": binary_bytes}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": position_bytes, "target": 34962},
            {"buffer": 0, "byteOffset": position_bytes, "byteLength": point_count * 3, "target": 34962}
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": point_count,
                "type": "VEC3",
                "min": minimum,
                "max": maximum
            },
            {
                "bufferView": 1,
                "componentType": 5121,
                "normalized": true,
                "count": point_count,
                "type": "VEC3"
            }
        ]
    });
    let mut json = serde_json::to_vec(&document).map_err(|error| error.to_string())?;
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let mut binary = positions;
    binary.extend_from_slice(&colors);
    while binary.len() % 4 != 0 {
        binary.push(0);
    }
    let total_length = 12 + 8 + json.len() + 8 + binary.len();
    let mut output = Vec::with_capacity(total_length);
    output.extend_from_slice(b"glTF");
    output.extend_from_slice(&2_u32.to_le_bytes());
    output.extend_from_slice(&(total_length as u32).to_le_bytes());
    output.extend_from_slice(&(json.len() as u32).to_le_bytes());
    output.extend_from_slice(&0x4E4F534A_u32.to_le_bytes());
    output.extend_from_slice(&json);
    output.extend_from_slice(&(binary.len() as u32).to_le_bytes());
    output.extend_from_slice(&0x004E4942_u32.to_le_bytes());
    output.extend_from_slice(&binary);
    Ok(output)
}

fn live_map_fingerprint(packet: &[u8]) -> String {
    let hash = packet.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    format!("fnv1a64:{hash:016x}")
}

fn save_live_reconstruction_preview(
    project_root: &Path,
    phase_root: &Path,
    realtime: &Arc<Mutex<RealtimeEngineSnapshot>>,
) -> Result<bool, String> {
    let snapshot = realtime
        .lock()
        .map_err(|_| "Realtime preview state is unavailable".to_string())?
        .clone();
    let packet = snapshot
        .points
        .as_ref()
        .map(|frame| Arc::clone(&frame.packet));
    let Some(packet) = packet else {
        return Ok(false);
    };
    if !valid_packed_point_preview(packet.as_ref()) {
        return Err("Realtime point preview is incomplete".to_string());
    }
    fs::write(phase_root.join(LIVE_CAPTURE_PREVIEW_FILE), packet.as_ref())
        .map_err(|error| format!("Could not preserve the live reconstruction: {error}"))?;
    let live_root = project_root.join("outputs").join(LIVE_ARTIFACT_DIRECTORY);
    fs::create_dir_all(live_root.join("submaps")).map_err(|error| error.to_string())?;
    fs::create_dir_all(live_root.join("coverage")).map_err(|error| error.to_string())?;
    write_export(&live_root.join("latest-preview.bin"), packet.as_ref())?;
    write_export(
        &live_root.join("latest-preview.ply"),
        &packed_point_preview_to_ply(packet.as_ref())?,
    )?;
    write_export(
        &live_root.join("latest-preview.glb"),
        &packed_point_preview_to_glb(packet.as_ref())?,
    )?;
    let journal = phase_root.join("tracking.jsonl");
    if let Ok(bytes) = fs::read(&journal) {
        write_export(&live_root.join("poses.jsonl"), &bytes)?;
    }
    let loop_journal = phase_root.join("live_loops.jsonl");
    let loop_decisions = fs::read_to_string(&loop_journal)
        .ok()
        .map(|value| {
            value
                .lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if let Ok(bytes) = fs::read(&loop_journal) {
        write_export(&live_root.join("loops.jsonl"), &bytes)?;
    }
    if let Some(coverage) = &snapshot.coverage {
        storage::write_json(&live_root.join("coverage").join("latest.json"), coverage)?;
    }
    if let Some(submaps) = &snapshot.submaps {
        storage::write_json(&live_root.join("submaps").join("descriptors.json"), submaps)?;
    }
    let phase_manifest = File::open(phase_root.join("phase.json"))
        .ok()
        .and_then(|file| serde_json::from_reader::<_, serde_json::Value>(file).ok());
    let session = serde_json::json!({
        "schemaVersion": 1,
        "contractVersion": 2,
        "sourceType": "rgbd",
        "liveEngineRevision": env!("CARGO_PKG_VERSION"),
        "phaseId": phase_root.file_name().map(|value| value.to_string_lossy()),
        "calibration": phase_manifest.as_ref().and_then(|value| value.get("camera")),
        "sensor": phase_manifest.as_ref().and_then(|value| value.get("sensor")),
        "submaps": snapshot.submaps.as_ref().and_then(|value| value.get("submaps")).cloned().unwrap_or_else(|| serde_json::json!([])),
        "poseGraph": snapshot.submaps.as_ref().and_then(|value| value.get("poseGraph")).cloned(),
        "coverage": snapshot.coverage.clone(),
        "provisionalScaleStatus": if snapshot.status.scale_status.is_empty() { "SENSOR_METRIC" } else { snapshot.status.scale_status.as_str() },
        "trackingStatistics": {
            "processedFrames": snapshot.status.processed_frames,
            "acceptedFrames": snapshot.status.accepted_frames,
            "rejectedFrames": snapshot.status.rejected_frames,
            "integratedFrames": snapshot.status.integrated_frames,
            "trackingConfidence": snapshot.status.tracking_confidence
        },
        "acceptedLoops": loop_decisions.iter().filter(|value| value.get("accepted").and_then(|accepted| accepted.as_bool()).unwrap_or(false)).cloned().collect::<Vec<_>>(),
        "rejectedLoops": loop_decisions.iter().filter(|value| !value.get("accepted").and_then(|accepted| accepted.as_bool()).unwrap_or(false)).cloned().collect::<Vec<_>>(),
        "queueDrops": {
            "source": snapshot.status.source_drops,
            "tracking": snapshot.status.tracking_queue_drops,
            "mapping": snapshot.status.mapping_drops,
            "preview": snapshot.status.dropped_preview_jobs
        },
        "peakMemory": {
            "allocatedLiveMapBytes": snapshot.status.allocated_live_map_bytes
        },
        "loopClosureCount": snapshot.status.loop_closure_count,
        "finalLiveMapFingerprint": live_map_fingerprint(packet.as_ref()),
        "preview": {
            "frameSequence": snapshot.points.as_ref().map(|frame| frame.frame_count),
            "pointCount": snapshot.status.point_count,
            "ply": "latest-preview.ply",
            "glb": "latest-preview.glb"
        },
        "publishedAt": Utc::now().to_rfc3339()
    });
    storage::write_json(&live_root.join("session.json"), &session)?;
    storage::write_json(
        &live_root.join("tracking-summary.json"),
        &serde_json::json!({
            "schemaVersion": 1,
            "state": snapshot.status.tracking_state,
            "confidence": snapshot.status.tracking_confidence,
            "processedFrames": snapshot.status.processed_frames,
            "acceptedFrames": snapshot.status.accepted_frames,
            "rejectedFrames": snapshot.status.rejected_frames,
            "integratedFrames": snapshot.status.integrated_frames,
            "integrationFrozen": snapshot.status.integration_frozen
        }),
    )?;
    Ok(true)
}

#[tauri::command]
pub async fn live_preview_frame(
    after_frame: u32,
    state: State<'_, AppState>,
) -> Result<tauri::ipc::Response, String> {
    let capture_root = state
        .active_capture
        .lock()
        .ok()
        .and_then(|active| active.as_ref().map(|capture| Arc::clone(&capture.realtime)));
    let body = if let Some(snapshot) = capture_root {
        realtime_packet(&snapshot, RealtimeGeometry::FusedPoints, after_frame)
    } else {
        state
            .active_preview
            .lock()
            .ok()
            .and_then(|active| active.as_ref().map(|preview| Arc::clone(&preview.realtime)))
            .map(|snapshot| realtime_packet(&snapshot, RealtimeGeometry::CameraPoints, after_frame))
            .unwrap_or_default()
    };
    Ok(tauri::ipc::Response::new(body))
}

#[tauri::command]
pub async fn live_reconstruction_mesh(
    after_frame: u32,
    state: State<'_, AppState>,
) -> Result<tauri::ipc::Response, String> {
    let capture_root = state
        .active_capture
        .lock()
        .ok()
        .and_then(|active| active.as_ref().map(|capture| Arc::clone(&capture.realtime)));
    let body = if let Some(snapshot) = capture_root {
        realtime_packet(&snapshot, RealtimeGeometry::Mesh, after_frame)
    } else {
        Vec::new()
    };
    Ok(tauri::ipc::Response::new(body))
}

#[tauri::command]
pub async fn live_reconstruction_overlay(
    mode: String,
    after_frame: u32,
    state: State<'_, AppState>,
) -> Result<tauri::ipc::Response, String> {
    let geometry = match mode.as_str() {
        "coverage" => RealtimeGeometry::CoveragePoints,
        "tracking" | "confidence" => RealtimeGeometry::TrackingPoints,
        _ => return Err("Live overlay mode must be coverage, tracking, or confidence".to_string()),
    };
    let realtime = state
        .active_capture
        .lock()
        .ok()
        .and_then(|active| active.as_ref().map(|capture| Arc::clone(&capture.realtime)));
    let body = realtime
        .map(|snapshot| realtime_packet(&snapshot, geometry, after_frame))
        .unwrap_or_default();
    Ok(tauri::ipc::Response::new(body))
}

#[tauri::command]
pub fn live_reconstruction_guidance(
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let realtime = state
        .active_capture
        .lock()
        .map_err(|_| "Capture state is unavailable".to_string())?
        .as_ref()
        .map(|capture| Arc::clone(&capture.realtime));
    let Some(realtime) = realtime else {
        return Ok(serde_json::json!({
            "contractVersion": 2,
            "coverage": null,
            "submaps": null
        }));
    };
    let snapshot = realtime
        .lock()
        .map_err(|_| "Realtime reconstruction state is unavailable".to_string())?;
    Ok(serde_json::json!({
        "contractVersion": 2,
        "coverage": snapshot.coverage,
        "submaps": snapshot.submaps
    }))
}

#[tauri::command]
pub async fn load_capture_draft(project_path: String) -> Result<tauri::ipc::Response, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let root = PathBuf::from(project_path);
        let project = storage::read_project(&root)?;
        let packet = fs::read(
            root.join("outputs")
                .join(LIVE_ARTIFACT_DIRECTORY)
                .join("latest-preview.bin"),
        )
        .ok()
        .or_else(|| {
            project
                .phases
                .iter()
                .rev()
                .filter(|phase| phase.status == "complete")
                .find_map(|phase| {
                    fs::read(
                        root.join("phases")
                            .join(&phase.id)
                            .join(LIVE_CAPTURE_PREVIEW_FILE),
                    )
                    .ok()
                })
        })
        .filter(|packet| valid_packed_point_preview(packet))
        .unwrap_or_default();
        Ok(tauri::ipc::Response::new(packet))
    })
    .await
    .map_err(|error| error.to_string())?
}

fn load_project_preview(project_root: &Path) -> Result<Vec<PreviewPoint>, String> {
    let project = storage::read_project(project_root)?;
    let outputs = project_root.join("outputs");
    let path = if project.processing_status == "processing" {
        outputs.join("build-preview.json")
    } else {
        outputs.join("preview.json")
    };
    if path.is_file() {
        return serde_json::from_reader(File::open(path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string());
    }
    Ok(Vec::new())
}

#[tauri::command]
pub async fn available_sensors(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<Vec<AvailableSensor>, String> {
    let resources = resource_root(&app);
    let saved = state
        .project
        .lock()
        .map(|project| project.settings.clone())
        .unwrap_or_default();
    tauri::async_runtime::spawn_blocking(move || {
        let mut sensors = Vec::new();

        // The Kinect v2 SDK has no passive enumeration API: probing opens the
        // camera, turns on its light, and starts its streams.
        // Advertise installed capture support without touching the hardware;
        // the camera is opened only when recording starts.
        if first_sensor_worker(
            storage::candidate_kinect_worker_paths(resources.as_deref()),
            "kinect_v2",
        )
        .is_some()
        {
            sensors.push(AvailableSensor {
                id: "kinect_v2:default".to_string(),
                kind: "kinect_v2".to_string(),
                name: "Kinect v2".to_string(),
                connection: "usb".to_string(),
                address: String::new(),
                serial: String::new(),
                connected: false,
                supports_imu: false,
            });
        }

        if let Some(worker) = first_modern_sensor_worker(
            storage::candidate_modern_sensor_worker_paths(resources.as_deref()),
        ) {
            let supports_femto = worker_capabilities(&worker)
                .iter()
                .any(|capability| capability == "femto_mega");
            if let Ok(output) = worker_command(&worker).arg("--list").output() {
                sensors.extend(parse_available_sensors(&output));
            }

            if supports_femto
                && saved.sensor_kind == "femto_mega"
                && saved.sensor_connection == "network"
                && !saved.sensor_address.is_empty()
                && !sensors.iter().any(|sensor| {
                    sensor.kind == "femto_mega"
                        && sensor.connection == "network"
                        && sensor.address == saved.sensor_address
                })
            {
                sensors.push(AvailableSensor {
                    id: if saved.sensor_id.is_empty() {
                        format!("femto_mega:network:{}", saved.sensor_address)
                    } else {
                        saved.sensor_id.clone()
                    },
                    kind: "femto_mega".to_string(),
                    name: "Orbbec Femto Mega (configured)".to_string(),
                    connection: "network".to_string(),
                    address: saved.sensor_address.clone(),
                    serial: String::new(),
                    connected: false,
                    supports_imu: true,
                });
            }
        }

        sensors.sort_by_key(|sensor| {
            if !saved.sensor_id.is_empty() && sensor.id == saved.sensor_id {
                0
            } else if sensor.kind == saved.sensor_kind
                && sensor.connection == saved.sensor_connection
                && (sensor.connection != "network" || sensor.address == saved.sensor_address)
            {
                1
            } else {
                2
            }
        });
        let mut seen = std::collections::HashSet::new();
        sensors.retain(|sensor| seen.insert(sensor.id.clone()));
        sensors
    })
    .await
    .map_err(|error| format!("Sensor discovery failed: {error}"))
}

#[tauri::command]
pub async fn runtime_info(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<RuntimeInfo, String> {
    let resources = resource_root(&app);
    let settings = state
        .project
        .lock()
        .map(|project| project.settings.clone())
        .unwrap_or_default();
    let runtime = tauri::async_runtime::spawn_blocking(move || {
        let sensor_capabilities = installed_sensor_capabilities(resources.as_deref());
        let sensor_worker_available = sensor_capabilities
            .iter()
            .any(|capability| capability == &settings.sensor_kind);
        let reconstruction_worker_available = first_existing(
            storage::candidate_reconstruction_worker_paths(resources.as_deref()),
        )
        .is_some();
        let splat_worker =
            first_existing(storage::candidate_splat_worker_paths(resources.as_deref()));
        let (splat_worker_available, splat_status) = match splat_worker {
            Some(worker) => {
                let mut command = worker_command(&worker);
                command.args([
                    "diagnostics",
                    "--require-cuda",
                    "--require-learned-features",
                    "--require-adaptive-frames",
                ]);
                match command.output() {
                    Ok(output) => {
                        let diagnostics =
                            serde_json::from_slice::<serde_json::Value>(&output.stdout)
                                .unwrap_or_default();
                        let cuda = diagnostics
                            .get("cuda")
                            .and_then(serde_json::Value::as_bool)
                            .unwrap_or(false);
                        let device = diagnostics
                            .get("device")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("CUDA device");
                        if output.status.success() && cuda {
                            (
                                true,
                                format!("Gaussian-splat CUDA runtime ready on {device}"),
                            )
                        } else if output.status.success() {
                            (
                                false,
                                "Splat runtime is installed, but CUDA is unavailable".to_string(),
                            )
                        } else {
                            let detail = output_message(&output);
                            (
                                false,
                                if detail.is_empty() {
                                    "Splat runtime diagnostics failed".to_string()
                                } else {
                                    detail
                                },
                            )
                        }
                    }
                    Err(error) => (
                        false,
                        format!("Could not start splat runtime diagnostics: {error}"),
                    ),
                }
            }
            None => (
                false,
                "Not installed; run npm run prepare:splat".to_string(),
            ),
        };
        let geometry_worker = first_existing(storage::candidate_geometry_worker_paths(
            resources.as_deref(),
        ));
        let (geometry_worker_available, geometry_status) = match geometry_worker {
            Some(worker) => {
                let output = worker_command(&worker)
                    .args([
                        "diagnostics",
                        "--require-lingbot",
                        "--require-lingbot-depth",
                        "--require-mapanything",
                        "--require-da3",
                        "--require-flashinfer",
                    ])
                    .output();
                match output {
                    Ok(output) if output.status.success() => (
                        true,
                        "Isolated LingBot, MapAnything, and DA3 Max runtime ready".to_string(),
                    ),
                    Ok(output) => {
                        let detail = output_message(&output);
                        (
                            false,
                            if detail.is_empty() {
                                "Learned geometry runtime diagnostics failed".to_string()
                            } else {
                                detail
                            },
                        )
                    }
                    Err(error) => (
                        false,
                        format!("Could not start learned geometry diagnostics: {error}"),
                    ),
                }
            }
            None => (
                false,
                "Not installed; run npm run prepare:splat".to_string(),
            ),
        };
        let sensor_status = if sensor_worker_available {
            format!(
                "{} capture support ready; the camera opens when recording starts",
                sensor_name(&settings)
            )
        } else {
            format!(
                "{} capture support is missing from this app build",
                sensor_name(&settings)
            )
        };

        RuntimeInfo {
            platform: std::env::consts::OS.to_string(),
            sensor_capabilities,
            sensor_worker_available,
            sensor_status,
            reconstruction_worker_available,
            splat_worker_available,
            splat_status,
            geometry_worker_available,
            geometry_status,
        }
    })
    .await
    .unwrap_or_else(|error| RuntimeInfo {
        platform: std::env::consts::OS.to_string(),
        sensor_capabilities: Vec::new(),
        sensor_worker_available: false,
        sensor_status: format!("Sensor connection check failed: {error}"),
        reconstruction_worker_available: false,
        splat_worker_available: false,
        splat_status: "Splat runtime detection failed".to_string(),
        geometry_worker_available: false,
        geometry_status: "Learned geometry runtime detection failed".to_string(),
    });
    Ok(runtime)
}

#[tauri::command]
pub fn current_project(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    ensure_project(&app, state.inner())
}

#[tauri::command]
pub fn list_projects(app: AppHandle) -> Result<Vec<ProjectCatalogEntry>, String> {
    storage::list_projects(&project_base(&app)?)
        .map(|projects| projects.into_iter().map(project_catalog_entry).collect())
}

#[tauri::command]
pub fn open_project(
    app: AppHandle,
    project_path: String,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    ensure_project_management_idle(state.inner())?;
    let base = project_base(&app)?;
    let requested = PathBuf::from(&project_path);
    storage::managed_project_path(&base, &requested)?;
    let mut project = storage::read_project(&requested)?;
    storage::recover_interrupted_phases(&mut project)?;
    if normalize_project(&mut project)
        | restore_sensor_preference(&app, &mut project)
        | crate::jobs::recover_interrupted_job(&mut project, &state.jobs)
    {
        storage::write_project(&project)?;
    }
    stop_active_preview(state.inner())?;
    *state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
    Ok(project)
}

#[tauri::command]
pub fn save_project(
    project_path: String,
    name: String,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    ensure_project_management_idle(state.inner())?;
    let mut project = state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())?
        .clone();
    if project.path != project_path {
        return Err("The selected project is no longer active".to_string());
    }
    project.name = validated_project_name(&name)?;
    storage::write_project(&project)?;
    *state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
    Ok(project)
}

#[tauri::command]
pub fn create_project(
    app: AppHandle,
    name: Option<String>,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    ensure_project_management_idle(state.inner())?;
    let name = name.as_deref().map(validated_project_name).transpose()?;
    stop_active_preview(state.inner())?;
    let previous_settings = state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())?
        .settings
        .clone();
    let folder = format!("scan-{}", Utc::now().format("%Y%m%d-%H%M%S-%3f"));
    let mut project = storage::create_project(&project_base(&app)?.join(folder))?;
    project.settings = previous_settings;
    if let Some(name) = name {
        project.name = name;
    }
    storage::write_project(&project)?;
    *state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
    Ok(project)
}

#[tauri::command]
pub fn delete_project(
    app: AppHandle,
    project_path: String,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    ensure_project_management_idle(state.inner())?;
    let base = project_base(&app)?;
    let requested = PathBuf::from(&project_path);
    let managed_path = storage::managed_project_path(&base, &requested)?;
    let selected = storage::read_project(&requested)?;
    let current = state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())?
        .clone();
    let deleting_current = current.id == selected.id;

    let mut replacement = if deleting_current {
        storage::list_projects(&base)?
            .into_iter()
            .find(|candidate| candidate.id != selected.id)
    } else {
        None
    };
    if deleting_current && replacement.is_none() {
        let folder = format!("scan-{}", Utc::now().format("%Y%m%d-%H%M%S-%3f"));
        let mut created = storage::create_project(&base.join(folder))?;
        created.settings = current.settings.clone();
        storage::write_project(&created)?;
        replacement = Some(created);
    }
    if deleting_current {
        stop_active_preview(state.inner())?;
    }
    fs::remove_dir_all(&managed_path)
        .map_err(|error| format!("Could not delete the project: {error}"))?;

    if let Some(mut project) = replacement {
        storage::recover_interrupted_phases(&mut project)?;
        if normalize_project(&mut project)
            | restore_sensor_preference(&app, &mut project)
            | crate::jobs::recover_interrupted_job(&mut project, &state.jobs)
        {
            storage::write_project(&project)?;
        }
        *state
            .project
            .lock()
            .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
        Ok(project)
    } else {
        Ok(current)
    }
}

#[tauri::command]
pub fn update_project_settings(
    app: AppHandle,
    project_path: String,
    mut settings: CaptureSettings,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    if state
        .active_capture
        .lock()
        .map_err(|_| "Capture state is unavailable".to_string())?
        .is_some()
    {
        return Err("Capture settings cannot change while recording".to_string());
    }
    if let Some(mut preview) = state
        .active_preview
        .lock()
        .map_err(|_| "Preview state is unavailable".to_string())?
        .take()
    {
        shutdown_sensor_session(&mut preview, Duration::from_secs(3));
        discard_preview_files(&preview);
    }
    let mut project = state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())?
        .clone();
    if project.path != project_path {
        return Err("The selected project is no longer active".to_string());
    }
    if project.processing_status == "processing" || project.active_job.is_some() {
        return Err("Capture settings cannot change during reconstruction".to_string());
    }
    settings.capture_fps = settings.capture_fps.clamp(1, 30);
    settings.max_depth_m = settings.max_depth_m.clamp(0.5, 8.0);
    settings.voxel_size_mm = settings.voxel_size_mm.clamp(1, 40);
    validate_sensor_settings(&mut settings)?;
    let mesh_repair_changed = project.settings.repair_mesh != settings.repair_mesh
        || project.settings.mesh_repair_profile != settings.mesh_repair_profile
        || project.settings.fill_inferred_mesh_holes != settings.fill_inferred_mesh_holes
        || project.settings.produce_watertight_mesh != settings.produce_watertight_mesh
        || project.settings.neural_sdf_refinement != settings.neural_sdf_refinement;
    let depth_refinement_changed = project.settings.depth_refinement_backend
        != settings.depth_refinement_backend
        || project.settings.lingbot_depth_refinement != settings.lingbot_depth_refinement;
    project.settings = settings;
    if depth_refinement_changed {
        for artifact in [
            project.artifacts.point_cloud.as_mut(),
            project.artifacts.textured_mesh.as_mut(),
            project.artifacts.gaussian_splat.as_mut(),
        ]
        .into_iter()
        .flatten()
        {
            artifact.status = "stale".to_string();
            artifact.stale = true;
        }
        project.depth_refinement = None;
    }
    if mesh_repair_changed {
        if let Some(artifact) = project.artifacts.textured_mesh.as_mut() {
            artifact.status = "stale".to_string();
            artifact.stale = true;
        }
        project.mesh_repair_profile = None;
        project.mesh_repair_status = None;
        project.mesh_repair_report_path = None;
        project.mesh_repair_fallback = None;
        project.mesh_repair_defects_fixed = None;
        project.mesh_repair_holes_filled = None;
        project.mesh_repair_openings_preserved = None;
        project.mesh_repair_unknown_preserved = None;
        project.watertight_mesh_output_path = None;
        project.neural_sdf = None;
    }
    storage::write_project(&project)?;
    write_sensor_preference(&app, &project.settings)?;
    *state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
    Ok(project)
}

fn start_sensor_session(
    app: &AppHandle,
    project_root: &Path,
    settings: &CaptureSettings,
    phase_id: String,
    phase_name: String,
    preview: bool,
) -> Result<ActiveCapture, String> {
    let worker = sensor_worker(app, settings)?;
    let phase_root = project_root.join("phases").join(&phase_id);
    fs::create_dir_all(&phase_root).map_err(|error| error.to_string())?;

    let (live_child, mut live_input, realtime) = start_realtime_engine(app, &phase_root, settings)?;
    let mut live_reconstruction = Some(live_child);
    if let Err(error) = wait_for_realtime_engine_ready(
        live_reconstruction.as_mut().unwrap(),
        &realtime,
        Duration::from_secs(45),
    ) {
        drop(live_input);
        stop_live_reconstruction_child(
            &phase_root,
            &mut live_reconstruction,
            Duration::from_secs(2),
        );
        return Err(error);
    }

    let mut command = worker_command(&worker);
    command
        .arg("--phase")
        .arg(&phase_root)
        .arg("--id")
        .arg(&phase_id)
        .arg("--name")
        .arg(&phase_name)
        .arg("--fps")
        .arg(settings.capture_fps.to_string())
        .arg("--max-depth")
        .arg(settings.max_depth_m.to_string())
        .arg("--stream-rgbd");
    if preview {
        command.arg("--preview");
    }
    append_sensor_args(&mut command, settings);
    let sensor_log = match File::create(phase_root.join("sensor.log")) {
        Ok(log) => log,
        Err(error) => {
            drop(live_input);
            stop_live_reconstruction_child(
                &phase_root,
                &mut live_reconstruction,
                Duration::from_secs(2),
            );
            return Err(format!("Could not create the sensor log: {error}"));
        }
    };
    let mut child = match command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::from(sensor_log))
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            drop(live_input);
            stop_live_reconstruction_child(
                &phase_root,
                &mut live_reconstruction,
                Duration::from_secs(2),
            );
            return Err(format!(
                "Could not start {} capture: {error}",
                sensor_name(settings)
            ));
        }
    };

    let manifest_path = phase_root.join("phase.json");
    let startup_deadline = Instant::now() + Duration::from_secs(12);
    while !manifest_path.exists() {
        match child.try_wait() {
            Ok(Some(status)) => {
                let detail = read_sensor_log(&phase_root);
                drop(live_input);
                stop_live_reconstruction_child(
                    &phase_root,
                    &mut live_reconstruction,
                    Duration::from_secs(2),
                );
                return Err(if detail.is_empty() {
                    format!("Sensor capture stopped during startup ({status})")
                } else {
                    detail
                });
            }
            Ok(None) => {}
            Err(error) => {
                child.kill().ok();
                child.wait().ok();
                drop(live_input);
                stop_live_reconstruction_child(
                    &phase_root,
                    &mut live_reconstruction,
                    Duration::from_secs(2),
                );
                return Err(format!("Could not inspect sensor startup: {error}"));
            }
        }
        if Instant::now() >= startup_deadline {
            child.kill().ok();
            child.wait().ok();
            let detail = read_sensor_log(&phase_root);
            drop(live_input);
            stop_live_reconstruction_child(
                &phase_root,
                &mut live_reconstruction,
                Duration::from_secs(2),
            );
            return Err(if detail.is_empty() {
                format!(
                    "{} did not begin streaming within 12 seconds",
                    sensor_name(settings)
                )
            } else {
                detail
            });
        }
        thread::sleep(Duration::from_millis(50));
    }

    let sensor_output = match child.stdout.take() {
        Some(output) => output,
        None => {
            child.kill().ok();
            child.wait().ok();
            drop(live_input);
            stop_live_reconstruction_child(
                &phase_root,
                &mut live_reconstruction,
                Duration::from_secs(2),
            );
            return Err("Could not open the RGB-D sensor stream".to_string());
        }
    };
    let sensor_relay = thread::spawn(move || {
        let mut source = BufReader::new(sensor_output);
        std::io::copy(&mut source, &mut live_input).ok();
        live_input.flush().ok();
    });

    Ok(ActiveCapture {
        child,
        sensor_relay: Some(sensor_relay),
        live_reconstruction,
        realtime,
        project_root: project_root.to_path_buf(),
        phase_root,
        phase_id,
        phase_name,
        settings: settings.clone(),
    })
}

#[tauri::command]
pub async fn start_sensor_preview(
    app: AppHandle,
    project_path: String,
    settings: CaptureSettings,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    let state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        start_sensor_preview_blocking(app, project_path, settings, &state)
    })
    .await
    .map_err(|error| error.to_string())?
}

fn start_sensor_preview_blocking(
    app: AppHandle,
    project_path: String,
    mut settings: CaptureSettings,
    state: &AppState,
) -> Result<ProjectSummary, String> {
    validate_sensor_settings(&mut settings)?;
    if settings.sensor_kind == "kinect_v2" {
        return Err("Kinect v2 preview opens when capture starts".to_string());
    }
    if state
        .active_capture
        .lock()
        .map_err(|_| "Capture state is unavailable".to_string())?
        .is_some()
    {
        return Err("A sensor capture is already running".to_string());
    }
    let project_root = PathBuf::from(project_path);
    let project = storage::read_project(&project_root)?;
    if project.processing_status == "processing" || project.active_job.is_some() {
        return Err("Camera preview is paused during reconstruction".to_string());
    }
    let mut active = state
        .active_preview
        .lock()
        .map_err(|_| "Preview state is unavailable".to_string())?;
    if active
        .as_ref()
        .is_some_and(|preview| preview.project_root == project_root && preview.settings == settings)
    {
        return Ok(project);
    }
    if let Some(mut previous) = active.take() {
        shutdown_sensor_session(&mut previous, Duration::from_secs(3));
        discard_preview_files(&previous);
    }
    let phase_id = Uuid::new_v4().to_string();
    let phase_name = format!(
        "{} phase {}",
        sensor_name(&settings),
        project.phases.len() + 1
    );
    let preview = start_sensor_session(&app, &project_root, &settings, phase_id, phase_name, true)?;
    *active = Some(preview);
    Ok(project)
}

#[tauri::command]
pub async fn stop_sensor_preview(state: State<'_, AppState>) -> Result<(), String> {
    let state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || stop_sensor_preview_blocking(&state))
        .await
        .map_err(|error| error.to_string())?
}

fn stop_sensor_preview_blocking(state: &AppState) -> Result<(), String> {
    let preview = state
        .active_preview
        .lock()
        .map_err(|_| "Preview state is unavailable".to_string())?
        .take();
    if let Some(mut preview) = preview {
        shutdown_sensor_session(&mut preview, Duration::from_secs(4));
        discard_preview_files(&preview);
    }
    Ok(())
}

#[tauri::command]
pub async fn start_sensor_phase(
    app: AppHandle,
    project_path: String,
    settings: CaptureSettings,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    let state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        start_sensor_phase_blocking(app, project_path, settings, &state)
    })
    .await
    .map_err(|error| error.to_string())?
}

fn start_sensor_phase_blocking(
    app: AppHandle,
    project_path: String,
    mut settings: CaptureSettings,
    state: &AppState,
) -> Result<ProjectSummary, String> {
    validate_sensor_settings(&mut settings)?;
    let mut active = state
        .active_capture
        .lock()
        .map_err(|_| "Capture state is unavailable".to_string())?;
    if active.is_some() {
        return Err("A sensor capture is already running".to_string());
    }

    let project_root = PathBuf::from(project_path);
    let worker = sensor_worker(&app, &settings)?;
    let mut project = storage::read_project(&project_root)?;
    if project.active_job.is_some() {
        return Err("Cancel the active artifact job before capturing another phase".to_string());
    }
    if !project.media_sources.is_empty() {
        return Err(
            "Start a new project for RGB-D capture; this project uses photos/video".to_string(),
        );
    }
    project.settings = settings;
    write_sensor_preference(&app, &project.settings)?;
    project.settings.voxel_size_mm = project.settings.voxel_size_mm.clamp(1, 40);
    project.processing_status = "idle".to_string();
    project.processing_error = None;
    project.point_count = None;
    project.output_path = None;
    project.mesh_triangle_count = None;
    project.mesh_output_path = None;
    project.camera_frame_count = None;
    project.confidence_score = None;
    project.confidence_label = None;
    project.confidence_detail = None;
    project.frames_used = None;
    project.processing_backend = None;
    project.processing_duration_seconds = None;
    project.artifacts = crate::models::ArtifactCatalog::default();

    let preview = state
        .active_preview
        .lock()
        .map_err(|_| "Preview state is unavailable".to_string())?
        .take();
    if let Some(mut preview) = preview {
        let reusable = preview.project_root == project_root
            && preview.settings == project.settings
            && matches!(preview.child.try_wait(), Ok(None));
        if reusable {
            if let Ok(mut snapshot) = preview.realtime.lock() {
                snapshot.points = None;
                snapshot.mesh = None;
                snapshot.status.processed_frames = 0;
                snapshot.status.integrated_frames = 0;
                snapshot.status.rejected_frames = 0;
                snapshot.status.point_count = 0;
                snapshot.status.triangle_count = 0;
            }
            File::create(preview.phase_root.join("record.flag"))
                .map_err(|error| format!("Could not begin recording: {error}"))?;
            let is_reference = project.phases.is_empty();
            project.phases.push(crate::models::PhaseSummary {
                id: preview.phase_id.clone(),
                name: preview.phase_name.clone(),
                created_at: Utc::now().to_rfc3339(),
                duration_seconds: 0,
                frame_count: 0,
                status: "capturing".to_string(),
                overlap_hint: if is_reference {
                    "Reference phase".to_string()
                } else {
                    "Capture overlapping geometry".to_string()
                },
            });
            storage::write_project(&project)?;
            *state
                .project
                .lock()
                .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
            *active = Some(preview);
            return Ok(project);
        }
        shutdown_sensor_session(&mut preview, Duration::from_secs(3));
        discard_preview_files(&preview);
    }
    let phase_id = Uuid::new_v4().to_string();
    let phase_name = format!(
        "{} phase {}",
        sensor_name(&project.settings),
        project.phases.len() + 1
    );
    let phase_root = project_root.join("phases").join(&phase_id);
    fs::create_dir_all(&phase_root).map_err(|error| error.to_string())?;

    // Make the CUDA engine ready before opening the sensor so the first
    // captured frames can already contribute to the visible map. The packaged
    // runtime is kept extracted, so this is normally sub-second initialization.
    let (live_child, mut live_input, realtime) =
        start_realtime_engine(&app, &phase_root, &project.settings)?;
    let mut live_reconstruction = Some(live_child);
    if let Err(error) = wait_for_realtime_engine_ready(
        live_reconstruction.as_mut().unwrap(),
        &realtime,
        Duration::from_secs(45),
    ) {
        drop(live_input);
        stop_live_reconstruction_child(
            &phase_root,
            &mut live_reconstruction,
            Duration::from_secs(2),
        );
        return Err(error);
    }

    let mut command = worker_command(&worker);
    command
        .arg("--phase")
        .arg(&phase_root)
        .arg("--id")
        .arg(&phase_id)
        .arg("--name")
        .arg(&phase_name)
        .arg("--fps")
        .arg(project.settings.capture_fps.to_string())
        .arg("--max-depth")
        .arg(project.settings.max_depth_m.to_string())
        .arg("--stream-rgbd");
    append_sensor_args(&mut command, &project.settings);
    let sensor_log = match File::create(phase_root.join("sensor.log")) {
        Ok(log) => log,
        Err(error) => {
            drop(live_input);
            stop_live_reconstruction_child(
                &phase_root,
                &mut live_reconstruction,
                Duration::from_secs(2),
            );
            return Err(format!("Could not create the sensor log: {error}"));
        }
    };
    let mut child = match command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::from(sensor_log))
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            drop(live_input);
            stop_live_reconstruction_child(
                &phase_root,
                &mut live_reconstruction,
                Duration::from_secs(2),
            );
            return Err(format!(
                "Could not start {} capture: {error}",
                sensor_name(&project.settings)
            ));
        }
    };

    let manifest_path = phase_root.join("phase.json");
    let startup_deadline = Instant::now() + Duration::from_secs(12);
    while !manifest_path.exists() {
        match child.try_wait() {
            Ok(Some(status)) => {
                let detail = read_sensor_log(&phase_root);
                drop(live_input);
                stop_live_reconstruction_child(
                    &phase_root,
                    &mut live_reconstruction,
                    Duration::from_secs(2),
                );
                return Err(if detail.is_empty() {
                    format!("Sensor capture stopped during startup ({status})")
                } else {
                    detail
                });
            }
            Ok(None) => {}
            Err(error) => {
                child.kill().ok();
                child.wait().ok();
                drop(live_input);
                stop_live_reconstruction_child(
                    &phase_root,
                    &mut live_reconstruction,
                    Duration::from_secs(2),
                );
                return Err(format!("Could not inspect sensor startup: {error}"));
            }
        }
        if Instant::now() >= startup_deadline {
            child.kill().ok();
            child.wait().ok();
            let detail = read_sensor_log(&phase_root);
            drop(live_input);
            stop_live_reconstruction_child(
                &phase_root,
                &mut live_reconstruction,
                Duration::from_secs(2),
            );
            return Err(if detail.is_empty() {
                format!(
                    "{} did not begin streaming within 12 seconds",
                    sensor_name(&project.settings)
                )
            } else {
                detail
            });
        }
        thread::sleep(Duration::from_millis(50));
    }

    let sensor_output = match child.stdout.take() {
        Some(output) => output,
        None => {
            child.kill().ok();
            child.wait().ok();
            drop(live_input);
            stop_live_reconstruction_child(
                &phase_root,
                &mut live_reconstruction,
                Duration::from_secs(2),
            );
            return Err("Could not open the RGB-D sensor stream".to_string());
        }
    };
    let sensor_relay = thread::spawn(move || {
        let mut source = BufReader::new(sensor_output);
        std::io::copy(&mut source, &mut live_input).ok();
        live_input.flush().ok();
    });

    let is_reference = project.phases.is_empty();
    project.phases.push(crate::models::PhaseSummary {
        id: phase_id.clone(),
        name: phase_name.clone(),
        created_at: Utc::now().to_rfc3339(),
        duration_seconds: 0,
        frame_count: 0,
        status: "capturing".to_string(),
        overlap_hint: if is_reference {
            "Reference phase".to_string()
        } else {
            "Capture overlapping geometry".to_string()
        },
    });
    let pending_capture = ActiveCapture {
        child,
        sensor_relay: Some(sensor_relay),
        live_reconstruction,
        realtime,
        project_root,
        phase_root,
        phase_id,
        phase_name,
        settings: project.settings.clone(),
    };
    let mut project_state = state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())?;
    storage::write_project(&project)?;
    *project_state = project.clone();
    drop(project_state);
    *active = Some(pending_capture);
    Ok(project)
}

#[tauri::command]
pub fn capture_status(state: State<'_, AppState>) -> Result<CaptureStatus, String> {
    let mut active_snapshot = None;
    let ended_capture = {
        let mut active = state
            .active_capture
            .lock()
            .map_err(|_| "Capture state is unavailable".to_string())?;
        if let Some(capture) = active.as_mut() {
            match capture
                .child
                .try_wait()
                .map_err(|error| error.to_string())?
            {
                Some(status) => active.take().map(|capture| (capture, status)),
                None => {
                    active_snapshot = Some((
                        capture.phase_root.clone(),
                        capture.phase_id.clone(),
                        Arc::clone(&capture.realtime),
                        true,
                    ));
                    None
                }
            }
        } else {
            None
        }
    };
    let mut project = state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())?
        .clone();

    if let Some((mut capture, status)) = ended_capture {
        drain_sensor_relay(&mut capture, Duration::from_secs(1));
        drain_live_reconstruction(&mut capture, Duration::from_secs(1));
        stop_live_reconstruction(&mut capture, Duration::from_secs(2));
        save_live_reconstruction_preview(
            &capture.project_root,
            &capture.phase_root,
            &capture.realtime,
        )
        .ok();
        drain_sensor_relay(&mut capture, Duration::from_secs(1));
        let detail = read_sensor_log(&capture.phase_root);
        let frame_count = indexed_frame_count(&capture.phase_root);
        let clean_stop = status.success();
        let completed = frame_count > 0;
        if let Some(phase) = project
            .phases
            .iter_mut()
            .find(|phase| phase.id == capture.phase_id)
        {
            phase.frame_count = frame_count;
            phase.duration_seconds =
                (frame_count / project.settings.capture_fps.max(1)).max(u32::from(frame_count > 0));
            phase.status = if completed { "complete" } else { "failed" }.to_string();
            phase.overlap_hint = if completed && clean_stop {
                "Capture ended; ready for alignment".to_string()
            } else if completed {
                "Sensor stream ended unexpectedly; indexed frames were retained".to_string()
            } else {
                "Sensor stream ended unexpectedly".to_string()
            };
        }
        storage::write_project(&project)?;
        *state
            .project
            .lock()
            .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
        let total_frame_count = project.phases.iter().map(|phase| phase.frame_count).sum();
        let reconstruction = reconstruction_progress(Path::new(&project.path));
        let selected_sensor_name = sensor_name(&project.settings).to_string();
        return Ok(CaptureStatus {
            project: project.clone(),
            live_contract_version: 2,
            preview: Vec::new(),
            capturing: false,
            previewing: false,
            sensor_connected: false,
            sensor_paused: false,
            sensor_status: if detail.is_empty() {
                "Sensor capture stream stopped".to_string()
            } else {
                detail.clone()
            },
            sensor_name: selected_sensor_name,
            frame_count,
            total_frame_count,
            preview_point_count: 0,
            stream_fps: 0.0,
            tracking: false,
            tracking_status: "Tracking stopped".to_string(),
            tracking_state: "complete".to_string(),
            tracking_confidence: 0.0,
            imu_active: false,
            imu_rate_hz: 0.0,
            live_reconstruction_active: false,
            live_reconstruction_mode: project.settings.live_reconstruction.clone(),
            live_processed_frame_count: 0,
            live_integrated_frame_count: 0,
            live_rejected_frame_count: 0,
            live_triangle_count: 0,
            tracking_fps: 0.0,
            source_drop_count: 0,
            tracking_queue_drop_count: 0,
            mapping_drop_count: 0,
            tracking_queue_depth: 0,
            mapping_queue_depth: 0,
            tracking_overlap: 0.0,
            pose_uncertainty_mm: None,
            pose_uncertainty_degrees: None,
            pose_latency_ms: None,
            map_update_latency_ms: None,
            map_update_hz: 0.0,
            allocated_live_map_bytes: 0,
            active_voxel_count: 0,
            active_surfel_count: 0,
            resident_submap_count: 0,
            host_cached_submap_count: 0,
            dropped_preview_job_count: 0,
            degradation_level: 0,
            loop_closure_count: 0,
            loop_correction_active: false,
            live_scale_status: "SENSOR_METRIC".to_string(),
            integration_frozen: false,
            depth_rmse_mm: None,
            live_reconstruction_backend: None,
            reconstruction,
            error: (!clean_stop || !completed).then_some(if completed {
                if detail.is_empty() {
                    format!(
                        "The sensor stream ended unexpectedly; {frame_count} indexed frames were retained"
                    )
                } else {
                    format!("{detail} · {frame_count} indexed frames were retained")
                }
            } else if detail.is_empty() {
                "The sensor capture stopped before a usable phase was completed".to_string()
            } else {
                detail
            }),
        });
    }

    let mut preview_error = None;
    if active_snapshot.is_none() {
        let ended_preview = {
            let mut active = state
                .active_preview
                .lock()
                .map_err(|_| "Preview state is unavailable".to_string())?;
            if let Some(preview) = active.as_mut() {
                match preview
                    .child
                    .try_wait()
                    .map_err(|error| error.to_string())?
                {
                    Some(status) => active.take().map(|preview| (preview, status)),
                    None => {
                        active_snapshot = Some((
                            preview.phase_root.clone(),
                            preview.phase_id.clone(),
                            Arc::clone(&preview.realtime),
                            false,
                        ));
                        None
                    }
                }
            } else {
                None
            }
        };
        if let Some((mut preview, status)) = ended_preview {
            drain_sensor_relay(&mut preview, Duration::from_secs(1));
            stop_live_reconstruction(&mut preview, Duration::from_secs(2));
            let detail = read_sensor_log(&preview.phase_root);
            preview_error = Some(if detail.is_empty() {
                format!("Camera preview stopped unexpectedly ({status})")
            } else {
                detail
            });
            discard_preview_files(&preview);
        }
    }

    if let Some((phase_root, phase_id, realtime, recording)) = active_snapshot {
        let live = live_worker_status(&phase_root);
        let frame_count = live
            .as_ref()
            .map(|status| status.frame_count)
            .or_else(|| {
                project
                    .phases
                    .iter()
                    .find(|phase| phase.id == phase_id)
                    .map(|phase| phase.frame_count)
            })
            .unwrap_or(0);
        if recording {
            if let Some(phase) = project.phases.iter_mut().find(|phase| phase.id == phase_id) {
                phase.frame_count = frame_count;
                phase.duration_seconds = frame_count / project.settings.capture_fps.max(1);
            }
        }
        *state
            .project
            .lock()
            .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
        let (live_reconstruction, live_error) = realtime
            .lock()
            .map(|snapshot| (Some(snapshot.status.clone()), snapshot.error.clone()))
            .unwrap_or_else(|_| {
                (
                    None,
                    Some("Realtime reconstruction state is unavailable".to_string()),
                )
            });
        let preview = Vec::new();
        let total_frame_count = project.phases.iter().map(|phase| phase.frame_count).sum();
        let reconstruction = reconstruction_progress(Path::new(&project.path));
        let selected_sensor_name = live
            .as_ref()
            .map(|status| status.sensor_name.trim())
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| sensor_name(&project.settings).to_string());
        return Ok(CaptureStatus {
            project: project.clone(),
            live_contract_version: live_reconstruction
                .as_ref()
                .map(|status| status.contract_version)
                .filter(|version| *version > 0)
                .unwrap_or(1),
            preview_point_count: live_reconstruction
                .as_ref()
                .map(|status| status.point_count)
                .unwrap_or(preview.len() as u64),
            preview,
            capturing: recording,
            previewing: !recording,
            sensor_connected: live.is_some(),
            sensor_paused: false,
            sensor_status: live
                .as_ref()
                .map(|status| {
                    format!(
                        "{} {} at {:.1} fps",
                        selected_sensor_name,
                        if recording { "recording" } else { "previewing" },
                        status.stream_fps
                    )
                })
                .unwrap_or_else(|| "Waiting for the next sensor frame".to_string()),
            sensor_name: selected_sensor_name,
            frame_count,
            total_frame_count,
            stream_fps: live.as_ref().map(|status| status.stream_fps).unwrap_or(0.0),
            tracking: live_reconstruction
                .as_ref()
                .map(|status| status.tracking)
                .or_else(|| live.as_ref().map(|status| status.tracking))
                .unwrap_or(false),
            tracking_status: live_reconstruction
                .as_ref()
                .map(|status| status.tracking_status.clone())
                .or_else(|| live.as_ref().map(|status| status.tracking_status.clone()))
                .unwrap_or_else(|| "Initializing camera tracking".to_string()),
            tracking_state: live_reconstruction
                .as_ref()
                .map(|status| status.tracking_state.clone())
                .filter(|state| !state.is_empty())
                .unwrap_or_else(|| if recording { "searching" } else { "preview" }.to_string()),
            tracking_confidence: live_reconstruction
                .as_ref()
                .map(|status| status.tracking_confidence)
                .unwrap_or(0.0),
            imu_active: live
                .as_ref()
                .map(|status| status.imu_active)
                .unwrap_or(false),
            imu_rate_hz: live
                .as_ref()
                .map(|status| status.imu_rate_hz)
                .unwrap_or(0.0),
            live_reconstruction_active: live_reconstruction
                .as_ref()
                .map(|status| status.active)
                .unwrap_or(false),
            live_reconstruction_mode: live_reconstruction
                .as_ref()
                .map(|status| status.mode.clone())
                .unwrap_or_else(|| project.settings.live_reconstruction.clone()),
            live_processed_frame_count: live_reconstruction
                .as_ref()
                .map(|status| status.processed_frames)
                .unwrap_or(0),
            live_integrated_frame_count: live_reconstruction
                .as_ref()
                .map(|status| status.integrated_frames)
                .unwrap_or(0),
            live_rejected_frame_count: live_reconstruction
                .as_ref()
                .map(|status| status.rejected_frames)
                .unwrap_or(0),
            live_triangle_count: live_reconstruction
                .as_ref()
                .map(|status| status.triangle_count)
                .unwrap_or(0),
            tracking_fps: live_reconstruction
                .as_ref()
                .map(|status| status.tracking_fps)
                .unwrap_or(0.0),
            source_drop_count: live_reconstruction
                .as_ref()
                .map(|status| status.source_drops)
                .unwrap_or(0),
            tracking_queue_drop_count: live_reconstruction
                .as_ref()
                .map(|status| status.tracking_queue_drops)
                .unwrap_or(0),
            mapping_drop_count: live_reconstruction
                .as_ref()
                .map(|status| status.mapping_drops)
                .unwrap_or(0),
            tracking_queue_depth: live_reconstruction
                .as_ref()
                .map(|status| status.tracking_queue_depth)
                .unwrap_or(0),
            mapping_queue_depth: live_reconstruction
                .as_ref()
                .map(|status| status.mapping_queue_depth)
                .unwrap_or(0),
            tracking_overlap: live_reconstruction
                .as_ref()
                .map(|status| status.overlap)
                .unwrap_or(0.0),
            pose_uncertainty_mm: live_reconstruction
                .as_ref()
                .and_then(|status| status.pose_uncertainty_mm),
            pose_uncertainty_degrees: live_reconstruction
                .as_ref()
                .and_then(|status| status.pose_uncertainty_degrees),
            pose_latency_ms: live_reconstruction
                .as_ref()
                .and_then(|status| status.pose_latency_ms),
            map_update_latency_ms: live_reconstruction
                .as_ref()
                .and_then(|status| status.map_update_latency_ms),
            map_update_hz: live_reconstruction
                .as_ref()
                .map(|status| status.map_update_hz)
                .unwrap_or(0.0),
            allocated_live_map_bytes: live_reconstruction
                .as_ref()
                .map(|status| status.allocated_live_map_bytes)
                .unwrap_or(0),
            active_voxel_count: live_reconstruction
                .as_ref()
                .map(|status| status.active_voxel_count)
                .unwrap_or(0),
            active_surfel_count: live_reconstruction
                .as_ref()
                .map(|status| status.active_surfel_count)
                .unwrap_or(0),
            resident_submap_count: live_reconstruction
                .as_ref()
                .map(|status| status.resident_submap_count)
                .unwrap_or(0),
            host_cached_submap_count: live_reconstruction
                .as_ref()
                .map(|status| status.host_cached_submap_count)
                .unwrap_or(0),
            dropped_preview_job_count: live_reconstruction
                .as_ref()
                .map(|status| status.dropped_preview_jobs)
                .unwrap_or(0),
            degradation_level: live_reconstruction
                .as_ref()
                .map(|status| status.degradation_level)
                .unwrap_or(0),
            loop_closure_count: live_reconstruction
                .as_ref()
                .map(|status| status.loop_closure_count)
                .unwrap_or(0),
            loop_correction_active: live_reconstruction
                .as_ref()
                .map(|status| status.loop_correction_active)
                .unwrap_or(false),
            live_scale_status: live_reconstruction
                .as_ref()
                .map(|status| status.scale_status.clone())
                .filter(|status| !status.is_empty())
                .unwrap_or_else(|| "SENSOR_METRIC".to_string()),
            integration_frozen: live_reconstruction
                .as_ref()
                .map(|status| status.integration_frozen)
                .unwrap_or(false),
            depth_rmse_mm: live_reconstruction
                .as_ref()
                .and_then(|status| status.depth_rmse_mm),
            live_reconstruction_backend: live_reconstruction
                .as_ref()
                .map(|status| status.backend.clone()),
            reconstruction,
            error: live_error,
        });
    }

    let project_root = PathBuf::from(&project.path);
    if project.processing_status == "processing" || project.active_job.is_some() {
        let preview_path = project_root.join("outputs").join("build-preview.json");
        // Close the Windows file handle immediately after the raw read, before
        // spending time parsing the multi-megabyte JSON payload. This leaves a
        // reliable gap for the reconstruction worker's atomic replacement.
        let preview: Vec<PreviewPoint> = fs::read(preview_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        let total_frame_count = project.phases.iter().map(|phase| phase.frame_count).sum();
        let selected_sensor_name = sensor_name(&project.settings).to_string();
        return Ok(CaptureStatus {
            project: project.clone(),
            live_contract_version: 2,
            preview_point_count: preview.len() as u64,
            preview,
            capturing: false,
            previewing: false,
            sensor_connected: false,
            sensor_paused: true,
            sensor_status: "Sensor preview paused while reconstructing".to_string(),
            sensor_name: selected_sensor_name,
            frame_count: 0,
            total_frame_count,
            stream_fps: 0.0,
            tracking: false,
            tracking_status: "Reconstruction preview".to_string(),
            tracking_state: "complete".to_string(),
            tracking_confidence: 0.0,
            imu_active: false,
            imu_rate_hz: 0.0,
            live_reconstruction_active: false,
            live_reconstruction_mode: project.settings.live_reconstruction.clone(),
            live_processed_frame_count: 0,
            live_integrated_frame_count: 0,
            live_rejected_frame_count: 0,
            live_triangle_count: 0,
            tracking_fps: 0.0,
            source_drop_count: 0,
            tracking_queue_drop_count: 0,
            mapping_drop_count: 0,
            tracking_queue_depth: 0,
            mapping_queue_depth: 0,
            tracking_overlap: 0.0,
            pose_uncertainty_mm: None,
            pose_uncertainty_degrees: None,
            pose_latency_ms: None,
            map_update_latency_ms: None,
            map_update_hz: 0.0,
            allocated_live_map_bytes: 0,
            active_voxel_count: 0,
            active_surfel_count: 0,
            resident_submap_count: 0,
            host_cached_submap_count: 0,
            dropped_preview_job_count: 0,
            degradation_level: 0,
            loop_closure_count: 0,
            loop_correction_active: false,
            live_scale_status: "SENSOR_METRIC".to_string(),
            integration_frozen: false,
            depth_rmse_mm: None,
            live_reconstruction_backend: None,
            reconstruction: reconstruction_progress(&project_root),
            error: None,
        });
    }
    let preview = load_project_preview(&project_root).unwrap_or_default();
    let total_frame_count = project.phases.iter().map(|phase| phase.frame_count).sum();
    let selected_sensor_name = sensor_name(&project.settings).to_string();
    Ok(CaptureStatus {
        project: project.clone(),
        live_contract_version: 2,
        preview_point_count: preview.len() as u64,
        preview,
        capturing: false,
        previewing: false,
        sensor_connected: false,
        sensor_paused: false,
        sensor_status: format!("{} opens when capture starts", selected_sensor_name),
        sensor_name: selected_sensor_name,
        frame_count: 0,
        total_frame_count,
        stream_fps: 0.0,
        tracking: false,
        tracking_status: "Ready to capture".to_string(),
        tracking_state: "ready".to_string(),
        tracking_confidence: 0.0,
        imu_active: false,
        imu_rate_hz: 0.0,
        live_reconstruction_active: false,
        live_reconstruction_mode: project.settings.live_reconstruction.clone(),
        live_processed_frame_count: 0,
        live_integrated_frame_count: 0,
        live_rejected_frame_count: 0,
        live_triangle_count: 0,
        tracking_fps: 0.0,
        source_drop_count: 0,
        tracking_queue_drop_count: 0,
        mapping_drop_count: 0,
        tracking_queue_depth: 0,
        mapping_queue_depth: 0,
        tracking_overlap: 0.0,
        pose_uncertainty_mm: None,
        pose_uncertainty_degrees: None,
        pose_latency_ms: None,
        map_update_latency_ms: None,
        map_update_hz: 0.0,
        allocated_live_map_bytes: 0,
        active_voxel_count: 0,
        active_surfel_count: 0,
        resident_submap_count: 0,
        host_cached_submap_count: 0,
        dropped_preview_job_count: 0,
        degradation_level: 0,
        loop_closure_count: 0,
        loop_correction_active: false,
        live_scale_status: "SENSOR_METRIC".to_string(),
        integration_frozen: false,
        depth_rmse_mm: None,
        live_reconstruction_backend: None,
        reconstruction: reconstruction_progress(&project_root),
        error: preview_error,
    })
}

#[tauri::command]
pub fn remove_capture(
    phase_id: String,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    if state
        .active_capture
        .lock()
        .map_err(|_| "Capture state is unavailable".to_string())?
        .is_some()
    {
        return Err("Stop the active capture before removing a phase".to_string());
    }

    let mut project = state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())?
        .clone();
    if project.processing_status == "processing" {
        return Err("Wait for reconstruction to finish before removing a phase".to_string());
    }
    let phase_index = project
        .phases
        .iter()
        .position(|phase| phase.id == phase_id)
        .ok_or_else(|| "The selected capture no longer exists".to_string())?;
    let project_root = PathBuf::from(&project.path);
    let phases_root = project_root.join("phases");
    let phase_root = phases_root.join(&phase_id);
    if phase_root.parent() != Some(phases_root.as_path()) {
        return Err("Refusing an invalid capture path".to_string());
    }

    if phase_root.exists() {
        fs::remove_dir_all(&phase_root)
            .map_err(|error| format!("Could not remove the capture files: {error}"))?;
    }
    project.phases.remove(phase_index);

    // Any reconstruction output depends on every selected phase and becomes
    // stale as soon as one is removed.
    let outputs = project_root.join("outputs");
    if outputs.exists() {
        fs::remove_dir_all(&outputs)
            .map_err(|error| format!("Could not clear the previous reconstruction: {error}"))?;
    }
    fs::create_dir_all(&outputs).map_err(|error| error.to_string())?;
    project.processing_status = "idle".to_string();
    project.processing_error = None;
    project.point_count = None;
    project.output_path = None;
    project.mesh_triangle_count = None;
    project.mesh_output_path = None;
    project.camera_frame_count = None;
    project.confidence_score = None;
    project.confidence_label = None;
    project.confidence_detail = None;
    project.frames_used = None;
    project.processing_backend = None;
    project.processing_duration_seconds = None;
    project.artifacts = crate::models::ArtifactCatalog::default();
    storage::write_project(&project)?;
    *state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
    Ok(project)
}

#[tauri::command]
pub async fn stop_sensor_phase(state: State<'_, AppState>) -> Result<ProjectSummary, String> {
    let capture = state
        .active_capture
        .lock()
        .map_err(|_| "Capture state is unavailable".to_string())?
        .take()
        .ok_or_else(|| "No sensor capture is running".to_string())?;
    let project_state = Arc::clone(&state.project);
    tauri::async_runtime::spawn_blocking(move || {
        let mut capture = capture;
        let mut stop_error = File::create(capture.phase_root.join("stop.flag"))
            .err()
            .map(|error| format!("Could not request a clean sensor stop: {error}"));
        let deadline = Instant::now() + Duration::from_secs(15);
        let status = if stop_error.is_some() {
            capture.child.kill().ok();
            capture.child.wait().ok()
        } else {
            loop {
                match capture.child.try_wait() {
                    Ok(Some(status)) => break Some(status),
                    Ok(None) if Instant::now() < deadline => {
                        thread::sleep(Duration::from_millis(80));
                    }
                    Ok(None) => {
                        stop_error = Some(
                            "Sensor capture did not flush within 15 seconds and was terminated"
                                .to_string(),
                        );
                        capture.child.kill().ok();
                        break capture.child.wait().ok();
                    }
                    Err(error) => {
                        stop_error = Some(format!("Could not inspect the sensor process: {error}"));
                        capture.child.kill().ok();
                        break capture.child.wait().ok();
                    }
                }
            }
        };
        drain_sensor_relay(&mut capture, Duration::from_secs(3));
        drain_live_reconstruction(&mut capture, Duration::from_secs(3));
        stop_live_reconstruction(&mut capture, Duration::from_secs(5));
        let live_preview_error = save_live_reconstruction_preview(
            &capture.project_root,
            &capture.phase_root,
            &capture.realtime,
        )
        .err();
        drain_sensor_relay(&mut capture, Duration::from_secs(1));

        let manifest_path = capture.phase_root.join("phase.json");
        let capture_summary = File::open(&manifest_path)
            .ok()
            .and_then(|file| serde_json::from_reader::<_, crate::models::PhaseManifest>(file).ok())
            .map(|manifest| (manifest.frame_count, manifest.duration_seconds))
            .unwrap_or((0, 0));
        let mut project = storage::read_project(&capture.project_root)?;
        let frame_count = capture_summary
            .0
            .max(indexed_frame_count(&capture.phase_root));
        let duration_seconds = capture_summary.1.max(
            (frame_count / project.settings.capture_fps.max(1)).max(u32::from(frame_count > 0)),
        );
        let clean_stop =
            stop_error.is_none() && status.as_ref().is_some_and(|status| status.success());
        // A CSV row is flushed only after both frame payloads are durable. Keep
        // those indexed frames reconstructable even if process shutdown itself
        // was forced or reported an error.
        let completed = frame_count > 0;
        let phase_count = project.phases.len();
        if let Some(phase) = project
            .phases
            .iter_mut()
            .find(|phase| phase.id == capture.phase_id)
        {
            phase.frame_count = frame_count;
            phase.duration_seconds = duration_seconds;
            phase.status = if completed {
                "complete".to_string()
            } else {
                "failed".to_string()
            };
            phase.overlap_hint = if frame_count == 0 {
                "No usable sensor frames were saved".to_string()
            } else if let Some(error) = &live_preview_error {
                format!("Capture saved; {error}")
            } else if let Some(error) = &stop_error {
                format!("Recovered {frame_count} indexed frames · {error}")
            } else if !clean_stop {
                "Sensor capture ended unexpectedly; archived frames were retained".to_string()
            } else if phase_count == 1 {
                "Reference phase".to_string()
            } else {
                "Ready for offline alignment".to_string()
            };
        }
        fs::remove_file(capture.phase_root.join("stop.flag")).ok();
        storage::write_project(&project)?;
        *project_state
            .lock()
            .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
        Ok(project)
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
pub async fn load_preview(project_path: String) -> Result<Vec<PreviewPoint>, String> {
    tauri::async_runtime::spawn_blocking(move || load_project_preview(&PathBuf::from(project_path)))
        .await
        .map_err(|error| error.to_string())?
}

fn load_preview_mesh_file(project_path: &str, filename: &str) -> Result<Response, String> {
    let path = PathBuf::from(project_path).join("outputs").join(filename);
    fs::read(&path)
        .map(Response::new)
        .map_err(|error| format!("Could not read {}: {error}", path.display()))
}

fn obj_index(value: &str, item_count: usize) -> Option<usize> {
    let index = value.parse::<isize>().ok()?;
    let resolved = if index > 0 {
        index - 1
    } else if index < 0 {
        item_count as isize + index
    } else {
        return None;
    };
    (resolved >= 0 && resolved < item_count as isize).then_some(resolved as usize)
}

fn pack_preview_mesh(path: &Path, _triangle_count: usize) -> Result<Vec<u8>, String> {
    // Keep every face in the fused indexed surface: striding by face disconnects
    // neighboring triangles and produces "confetti" along otherwise shared edges.
    let source =
        fs::read(path).map_err(|error| format!("Could not read {}: {error}", path.display()))?;
    let mut source_positions: Vec<[f32; 3]> = Vec::new();
    let mut source_uvs: Vec<[f32; 2]> = Vec::new();
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut remapped: HashMap<(usize, usize), u32> = HashMap::new();
    let mut requires_remapping = false;

    // Reading the whole OBJ and splitting borrowed lines avoids millions of
    // String allocations in debug builds. ScanLan meshes also use matching
    // position/UV indices, so their common path needs no hash-map lookup.
    for raw_line in source.split(|byte| *byte == b'\n') {
        let line = std::str::from_utf8(raw_line)
            .map_err(|error| format!("Could not parse {}: {error}", path.display()))?;
        let mut values = line.split_ascii_whitespace();
        match values.next() {
            Some("v") => {
                let x = values
                    .next()
                    .ok_or_else(|| "Invalid mesh vertex".to_string())?
                    .parse::<f32>()
                    .map_err(|error| format!("Invalid mesh vertex: {error}"))?;
                let y = values
                    .next()
                    .ok_or_else(|| "Invalid mesh vertex".to_string())?
                    .parse::<f32>()
                    .map_err(|error| format!("Invalid mesh vertex: {error}"))?;
                let z = values
                    .next()
                    .ok_or_else(|| "Invalid mesh vertex".to_string())?
                    .parse::<f32>()
                    .map_err(|error| format!("Invalid mesh vertex: {error}"))?;
                source_positions.push([x, y, z]);
            }
            Some("vt") => {
                let u = values
                    .next()
                    .ok_or_else(|| "Invalid mesh UV".to_string())?
                    .parse::<f32>()
                    .map_err(|error| format!("Invalid mesh UV: {error}"))?;
                let v = values
                    .next()
                    .ok_or_else(|| "Invalid mesh UV".to_string())?
                    .parse::<f32>()
                    .map_err(|error| format!("Invalid mesh UV: {error}"))?;
                source_uvs.push([u, v]);
            }
            Some("f") => {
                let Some(corners) = values
                    .next()
                    .zip(values.next())
                    .zip(values.next())
                    .map(|((first, second), third)| [first, second, third])
                else {
                    continue;
                };
                let mut source_triangle = [(0usize, 0usize); 3];
                let mut valid = true;
                for (corner_index, corner) in corners.iter().enumerate() {
                    let mut components = corner.split('/');
                    let position_index = components
                        .next()
                        .and_then(|value| obj_index(value, source_positions.len()));
                    let uv_index = components
                        .next()
                        .and_then(|value| obj_index(value, source_uvs.len()));
                    let (Some(position_index), Some(uv_index)) = (position_index, uv_index) else {
                        valid = false;
                        break;
                    };
                    source_triangle[corner_index] = (position_index, uv_index);
                }
                if valid {
                    if !requires_remapping
                        && source_triangle
                            .iter()
                            .all(|(position_index, uv_index)| position_index == uv_index)
                    {
                        indices.extend(
                            source_triangle
                                .iter()
                                .map(|(position_index, _)| *position_index as u32),
                        );
                        continue;
                    }
                    if !requires_remapping {
                        requires_remapping = true;
                        let aligned_indices = std::mem::take(&mut indices);
                        for source_index in aligned_indices {
                            let source_index = source_index as usize;
                            let preview_index = *remapped
                                .entry((source_index, source_index))
                                .or_insert_with(|| {
                                    let index = positions.len() as u32;
                                    positions.push(source_positions[source_index]);
                                    uvs.push(source_uvs[source_index]);
                                    index
                                });
                            indices.push(preview_index);
                        }
                    }
                    for key in source_triangle {
                        let preview_index = *remapped.entry(key).or_insert_with(|| {
                            let index = positions.len() as u32;
                            positions.push(source_positions[key.0]);
                            uvs.push(source_uvs[key.1]);
                            index
                        });
                        indices.push(preview_index);
                    }
                }
            }
            _ => {}
        }
    }

    if !requires_remapping {
        positions = source_positions;
        uvs = source_uvs;
    }

    if positions.is_empty() || positions.len() != uvs.len() || indices.is_empty() {
        return Err("The reconstructed mesh does not contain previewable triangles".to_string());
    }
    let vertex_count = u32::try_from(positions.len())
        .map_err(|_| "The reconstructed mesh has too many vertices to preview".to_string())?;
    let index_count = u32::try_from(indices.len())
        .map_err(|_| "The reconstructed mesh has too many indices to preview".to_string())?;
    let mut bytes = Vec::with_capacity(12 + positions.len() * 20 + indices.len() * 4);
    bytes.extend_from_slice(b"K2M1");
    bytes.extend_from_slice(&vertex_count.to_le_bytes());
    bytes.extend_from_slice(&index_count.to_le_bytes());
    for position in positions {
        for value in position {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for uv in uvs {
        for value in uv {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for index in indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    Ok(bytes)
}

fn valid_packed_preview_mesh(bytes: &[u8]) -> bool {
    if bytes.len() < 12 || &bytes[..4] != b"K2M1" {
        return false;
    }
    let vertex_count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let index_count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    12usize
        .checked_add(vertex_count.checked_mul(20).unwrap_or(usize::MAX))
        .and_then(|length| length.checked_add(index_count.checked_mul(4)?))
        == Some(bytes.len())
}

fn preview_cache_is_current(source: &Path, cache: &Path) -> bool {
    let Ok(source_modified) = source.metadata().and_then(|metadata| metadata.modified()) else {
        return false;
    };
    let Ok(cache_modified) = cache.metadata().and_then(|metadata| metadata.modified()) else {
        return false;
    };
    cache_modified >= source_modified
}

fn load_or_pack_preview_mesh(project_root: &Path) -> Result<Vec<u8>, String> {
    let output_root = project_root.join("outputs");
    let source = output_root.join("room-mesh.obj");
    let cache = output_root.join("room-mesh.preview.bin");
    if preview_cache_is_current(&source, &cache) {
        if let Ok(bytes) = fs::read(&cache) {
            if valid_packed_preview_mesh(&bytes) {
                return Ok(bytes);
            }
        }
    }

    let bytes = pack_preview_mesh(&source, 0)?;
    let temporary = output_root.join(format!(".room-mesh.preview.{}.tmp", Uuid::new_v4()));
    if fs::write(&temporary, &bytes).is_ok() {
        if cache.exists() {
            fs::remove_file(&cache).ok();
        }
        if fs::rename(&temporary, &cache).is_err() {
            fs::remove_file(&temporary).ok();
        }
    }
    Ok(bytes)
}

#[tauri::command]
pub async fn load_preview_mesh_geometry(project_path: String) -> Result<Response, String> {
    tauri::async_runtime::spawn_blocking(move || {
        load_or_pack_preview_mesh(&PathBuf::from(project_path)).map(Response::new)
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
pub async fn load_preview_mesh_texture(project_path: String) -> Result<Response, String> {
    tauri::async_runtime::spawn_blocking(move || {
        load_preview_mesh_file(&project_path, "room-texture.png")
    })
    .await
    .map_err(|error| error.to_string())?
}

const PLY_VERTEX_STRIDE: usize = 15;

fn validate_cloud_transform(transform: &CloudTransform) -> Result<(), String> {
    if transform
        .position
        .iter()
        .chain(transform.rotation.iter())
        .chain(transform.scale.iter())
        .any(|value| !value.is_finite())
    {
        return Err("The model edit pose contains a non-finite value".to_string());
    }
    if transform.scale.iter().any(|value| value.abs() < 1e-6) {
        return Err("The model edit pose contains a zero scale".to_string());
    }
    Ok(())
}

fn validate_clip_bounds(bounds: &Option<BoundingBoxClip>) -> Result<(), String> {
    let Some(bounds) = bounds else {
        return Ok(());
    };
    if bounds
        .min
        .iter()
        .chain(bounds.max.iter())
        .any(|value| !value.is_finite())
    {
        return Err("The bounding box contains a non-finite value".to_string());
    }
    if (0..3).any(|axis| bounds.max[axis] - bounds.min[axis] < 1e-6) {
        return Err("Each bounding-box maximum must be greater than its minimum".to_string());
    }
    Ok(())
}

fn clip_contains(bounds: &BoundingBoxClip, position: [f32; 3]) -> bool {
    (0..3).all(|axis| {
        position[axis].is_finite()
            && position[axis] >= bounds.min[axis]
            && position[axis] <= bounds.max[axis]
    })
}

fn transformed_position(position: [f32; 3], transform: &CloudTransform) -> [f32; 3] {
    let [x_angle, y_angle, z_angle] = transform.rotation.map(f32::to_radians);
    let (sx, cx) = (x_angle * 0.5).sin_cos();
    let (sy, cy) = (y_angle * 0.5).sin_cos();
    let (sz, cz) = (z_angle * 0.5).sin_cos();
    let qx = sx * cy * cz + cx * sy * sz;
    let qy = cx * sy * cz - sx * cy * sz;
    let qz = cx * cy * sz + sx * sy * cz;
    let qw = cx * cy * cz - sx * sy * sz;
    let [x, y, z] = [
        position[0] * transform.scale[0],
        position[1] * transform.scale[1],
        position[2] * transform.scale[2],
    ];
    let tx = 2.0 * (qy * z - qz * y);
    let ty = 2.0 * (qz * x - qx * z);
    let tz = 2.0 * (qx * y - qy * x);
    [
        x + qw * tx + (qy * tz - qz * ty) + transform.position[0],
        y + qw * ty + (qz * tx - qx * tz) + transform.position[1],
        z + qw * tz + (qx * ty - qy * tx) + transform.position[2],
    ]
}

fn transformed_direction(direction: [f32; 3], transform: &CloudTransform) -> [f32; 3] {
    let origin = transformed_position([0.0, 0.0, 0.0], transform);
    let endpoint = transformed_position(direction, transform);
    [
        endpoint[0] - origin[0],
        endpoint[1] - origin[1],
        endpoint[2] - origin[2],
    ]
}

fn transformed_normal(normal: [f32; 3], transform: &CloudTransform) -> [f32; 3] {
    // Positions use R*S, so normals use R*inverse(S), followed by normalization.
    let adjusted =
        std::array::from_fn(|axis| normal[axis] / (transform.scale[axis] * transform.scale[axis]));
    let transformed = transformed_direction(adjusted, transform);
    let length = transformed
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if length > f32::EPSILON {
        transformed.map(|value| value / length)
    } else {
        transformed
    }
}

fn transformed_obj(source: &str, transform: &CloudTransform) -> Result<String, String> {
    let mut output = String::with_capacity(source.len());
    let reverse_winding = transform.scale.iter().product::<f32>() < 0.0;
    for line in source.lines() {
        if let Some(vertex) = line.strip_prefix("v ") {
            let values = vertex
                .split_whitespace()
                .take(3)
                .map(str::parse::<f32>)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| "The generated OBJ contains an invalid vertex".to_string())?;
            if values.len() != 3 {
                return Err("The generated OBJ contains an incomplete vertex".to_string());
            }
            let position = transformed_position([values[0], values[1], values[2]], transform);
            output.push_str(&format!(
                "v {:.7} {:.7} {:.7}\n",
                position[0], position[1], position[2]
            ));
        } else if let Some(normal) = line.strip_prefix("vn ") {
            let values = normal
                .split_whitespace()
                .take(3)
                .map(str::parse::<f32>)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| "The generated OBJ contains an invalid normal".to_string())?;
            if values.len() != 3 {
                return Err("The generated OBJ contains an incomplete normal".to_string());
            }
            let normal = transformed_normal([values[0], values[1], values[2]], transform);
            output.push_str(&format!(
                "vn {:.7} {:.7} {:.7}\n",
                normal[0], normal[1], normal[2]
            ));
        } else if reverse_winding && line.starts_with("f ") {
            let vertices = line[2..].split_whitespace().collect::<Vec<_>>();
            if vertices.len() == 3 {
                output.push_str(&format!(
                    "f {} {} {}\n",
                    vertices[0], vertices[2], vertices[1]
                ));
            } else {
                output.push_str(line);
                output.push('\n');
            }
        } else {
            output.push_str(line);
            output.push('\n');
        }
    }
    Ok(output)
}

#[derive(Clone)]
struct ObjClipVertex {
    position: [f32; 3],
    texcoord: Option<[f32; 2]>,
    normal: Option<[f32; 3]>,
    source_position: Option<usize>,
    source_texcoord: Option<usize>,
    source_normal: Option<usize>,
}

#[derive(Clone, Copy)]
struct ObjOutputRef {
    position: usize,
    texcoord: Option<usize>,
    normal: Option<usize>,
}

#[derive(Default)]
struct ClippedObjGeometry {
    positions: Vec<[f32; 3]>,
    texcoords: Vec<[f32; 2]>,
    normals: Vec<[f32; 3]>,
    faces: Vec<[ObjOutputRef; 3]>,
    position_map: HashMap<usize, usize>,
    texcoord_map: HashMap<usize, usize>,
    normal_map: HashMap<usize, usize>,
}

fn parse_obj_index(value: &str, length: usize) -> Result<usize, String> {
    let parsed = value
        .parse::<isize>()
        .map_err(|_| "The generated OBJ contains an invalid face index".to_string())?;
    let index = if parsed > 0 {
        parsed - 1
    } else if parsed < 0 {
        length as isize + parsed
    } else {
        -1
    };
    if index < 0 || index as usize >= length {
        return Err("The generated OBJ face index is out of range".to_string());
    }
    Ok(index as usize)
}

fn parse_obj_clip_vertex(
    value: &str,
    positions: &[[f32; 3]],
    texcoords: &[[f32; 2]],
    normals: &[[f32; 3]],
) -> Result<ObjClipVertex, String> {
    let fields = value.split('/').collect::<Vec<_>>();
    let position_index = parse_obj_index(
        fields
            .first()
            .filter(|field| !field.is_empty())
            .ok_or_else(|| "The generated OBJ face is missing a position".to_string())?,
        positions.len(),
    )?;
    let texcoord_index = fields
        .get(1)
        .filter(|field| !field.is_empty())
        .map(|field| parse_obj_index(field, texcoords.len()))
        .transpose()?;
    let normal_index = fields
        .get(2)
        .filter(|field| !field.is_empty())
        .map(|field| parse_obj_index(field, normals.len()))
        .transpose()?;
    Ok(ObjClipVertex {
        position: positions[position_index],
        texcoord: texcoord_index.map(|index| texcoords[index]),
        normal: normal_index.map(|index| normals[index]),
        source_position: Some(position_index),
        source_texcoord: texcoord_index,
        source_normal: normal_index,
    })
}

fn interpolate_obj_clip_vertex(
    left: &ObjClipVertex,
    right: &ObjClipVertex,
    t: f32,
) -> ObjClipVertex {
    let lerp3 = |left: [f32; 3], right: [f32; 3]| {
        std::array::from_fn(|axis| left[axis] + (right[axis] - left[axis]) * t)
    };
    let texcoord = left.texcoord.zip(right.texcoord).map(|(left, right)| {
        std::array::from_fn(|axis| left[axis] + (right[axis] - left[axis]) * t)
    });
    let normal = left.normal.zip(right.normal).map(|(left, right)| {
        let mut value = lerp3(left, right);
        let length = value
            .iter()
            .map(|component| component * component)
            .sum::<f32>()
            .sqrt();
        if length > f32::EPSILON {
            value.iter_mut().for_each(|component| *component /= length);
        }
        value
    });
    ObjClipVertex {
        position: lerp3(left.position, right.position),
        texcoord,
        normal,
        source_position: None,
        source_texcoord: None,
        source_normal: None,
    }
}

fn clip_obj_polygon_axis(
    polygon: &[ObjClipVertex],
    axis: usize,
    boundary: f32,
    keep_greater: bool,
) -> Vec<ObjClipVertex> {
    if polygon.is_empty() {
        return Vec::new();
    }
    let inside = |vertex: &ObjClipVertex| {
        if keep_greater {
            vertex.position[axis] >= boundary
        } else {
            vertex.position[axis] <= boundary
        }
    };
    let mut output = Vec::with_capacity(polygon.len() + 1);
    let mut previous = polygon.last().unwrap();
    let mut previous_inside = inside(previous);
    for current in polygon {
        let current_inside = inside(current);
        if previous_inside != current_inside {
            let denominator = current.position[axis] - previous.position[axis];
            if denominator.abs() > f32::EPSILON {
                let t = ((boundary - previous.position[axis]) / denominator).clamp(0.0, 1.0);
                output.push(interpolate_obj_clip_vertex(previous, current, t));
            }
        }
        if current_inside {
            output.push(current.clone());
        }
        previous = current;
        previous_inside = current_inside;
    }
    output
}

fn intern_obj_clip_vertex(
    vertex: &ObjClipVertex,
    geometry: &mut ClippedObjGeometry,
) -> ObjOutputRef {
    let position = if let Some(source) = vertex.source_position {
        if let Some(index) = geometry.position_map.get(&source) {
            *index
        } else {
            let index = geometry.positions.len();
            geometry.positions.push(vertex.position);
            geometry.position_map.insert(source, index);
            index
        }
    } else {
        let index = geometry.positions.len();
        geometry.positions.push(vertex.position);
        index
    };
    let texcoord = vertex.texcoord.map(|value| {
        if let Some(source) = vertex.source_texcoord {
            if let Some(index) = geometry.texcoord_map.get(&source) {
                return *index;
            }
            let index = geometry.texcoords.len();
            geometry.texcoords.push(value);
            geometry.texcoord_map.insert(source, index);
            index
        } else {
            let index = geometry.texcoords.len();
            geometry.texcoords.push(value);
            index
        }
    });
    let normal = vertex.normal.map(|value| {
        if let Some(source) = vertex.source_normal {
            if let Some(index) = geometry.normal_map.get(&source) {
                return *index;
            }
            let index = geometry.normals.len();
            geometry.normals.push(value);
            geometry.normal_map.insert(source, index);
            index
        } else {
            let index = geometry.normals.len();
            geometry.normals.push(value);
            index
        }
    });
    ObjOutputRef {
        position,
        texcoord,
        normal,
    }
}

fn clipped_obj(source: &str, bounds: &BoundingBoxClip) -> Result<String, String> {
    let mut metadata = Vec::new();
    let mut positions = Vec::new();
    let mut texcoords = Vec::new();
    let mut normals = Vec::new();
    let mut face_lines = Vec::new();
    for line in source.lines() {
        if let Some(value) = line.strip_prefix("v ") {
            let values = value
                .split_whitespace()
                .take(3)
                .map(str::parse::<f32>)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| "The generated OBJ contains an invalid vertex".to_string())?;
            if values.len() != 3 {
                return Err("The generated OBJ contains an incomplete vertex".to_string());
            }
            positions.push([values[0], values[1], values[2]]);
        } else if let Some(value) = line.strip_prefix("vt ") {
            let values = value
                .split_whitespace()
                .take(2)
                .map(str::parse::<f32>)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| {
                    "The generated OBJ contains an invalid texture coordinate".to_string()
                })?;
            if values.len() != 2 {
                return Err(
                    "The generated OBJ contains an incomplete texture coordinate".to_string(),
                );
            }
            texcoords.push([values[0], values[1]]);
        } else if let Some(value) = line.strip_prefix("vn ") {
            let values = value
                .split_whitespace()
                .take(3)
                .map(str::parse::<f32>)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| "The generated OBJ contains an invalid normal".to_string())?;
            if values.len() != 3 {
                return Err("The generated OBJ contains an incomplete normal".to_string());
            }
            normals.push([values[0], values[1], values[2]]);
        } else if let Some(value) = line.strip_prefix("f ") {
            face_lines.push(value.to_string());
        } else {
            metadata.push(line.to_string());
        }
    }

    let mut geometry = ClippedObjGeometry::default();
    for face in face_lines {
        let corners = face
            .split_whitespace()
            .map(|value| parse_obj_clip_vertex(value, &positions, &texcoords, &normals))
            .collect::<Result<Vec<_>, _>>()?;
        if corners.len() < 3 {
            return Err("The generated OBJ contains an incomplete face".to_string());
        }
        for triangle_index in 1..corners.len() - 1 {
            let mut polygon = vec![
                corners[0].clone(),
                corners[triangle_index].clone(),
                corners[triangle_index + 1].clone(),
            ];
            for axis in 0..3 {
                polygon = clip_obj_polygon_axis(&polygon, axis, bounds.min[axis], true);
                polygon = clip_obj_polygon_axis(&polygon, axis, bounds.max[axis], false);
                if polygon.len() < 3 {
                    break;
                }
            }
            if polygon.len() < 3 {
                continue;
            }
            let references = polygon
                .iter()
                .map(|vertex| intern_obj_clip_vertex(vertex, &mut geometry))
                .collect::<Vec<_>>();
            for index in 1..references.len() - 1 {
                geometry
                    .faces
                    .push([references[0], references[index], references[index + 1]]);
            }
        }
    }
    if geometry.faces.is_empty() {
        return Err("The bounding box does not contain any mesh triangles".to_string());
    }

    let mut output = String::with_capacity(source.len());
    for line in metadata {
        output.push_str(&line);
        output.push('\n');
    }
    for value in geometry.positions {
        output.push_str(&format!(
            "v {:.7} {:.7} {:.7}\n",
            value[0], value[1], value[2]
        ));
    }
    for value in geometry.normals {
        output.push_str(&format!(
            "vn {:.7} {:.7} {:.7}\n",
            value[0], value[1], value[2]
        ));
    }
    for value in geometry.texcoords {
        output.push_str(&format!("vt {:.7} {:.7}\n", value[0], value[1]));
    }
    for face in geometry.faces {
        output.push_str("f");
        for vertex in face {
            let position = vertex.position + 1;
            match (vertex.texcoord, vertex.normal) {
                (Some(texcoord), Some(normal)) => {
                    output.push_str(&format!(" {}/{}/{}", position, texcoord + 1, normal + 1))
                }
                (Some(texcoord), None) => {
                    output.push_str(&format!(" {}/{}", position, texcoord + 1))
                }
                (None, Some(normal)) => output.push_str(&format!(" {}//{}", position, normal + 1)),
                (None, None) => output.push_str(&format!(" {position}")),
            }
        }
        output.push('\n');
    }
    Ok(output)
}

fn transformed_cloud_ply(
    mut bytes: Vec<u8>,
    transform: &CloudTransform,
) -> Result<Vec<u8>, String> {
    let marker = b"end_header\n";
    let payload_start = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .map(|position| position + marker.len())
        .ok_or_else(|| "The generated PLY header is invalid".to_string())?;
    if (bytes.len() - payload_start) % PLY_VERTEX_STRIDE != 0 {
        return Err("The generated PLY vertex data has an unexpected size".to_string());
    }
    for vertex in bytes[payload_start..].chunks_exact_mut(PLY_VERTEX_STRIDE) {
        let position = [
            f32::from_le_bytes(vertex[0..4].try_into().unwrap()),
            f32::from_le_bytes(vertex[4..8].try_into().unwrap()),
            f32::from_le_bytes(vertex[8..12].try_into().unwrap()),
        ];
        let transformed = transformed_position(position, transform);
        vertex[0..4].copy_from_slice(&transformed[0].to_le_bytes());
        vertex[4..8].copy_from_slice(&transformed[1].to_le_bytes());
        vertex[8..12].copy_from_slice(&transformed[2].to_le_bytes());
    }
    Ok(bytes)
}

fn unity_compatible_ply(mut bytes: Vec<u8>) -> Result<Vec<u8>, String> {
    let marker = b"end_header\n";
    let header_end = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .ok_or_else(|| "The generated PLY header is invalid".to_string())?;
    let payload_start = header_end + marker.len();
    let header = std::str::from_utf8(&bytes[..header_end])
        .map_err(|_| "The generated PLY header is not valid ASCII".to_string())?;
    if !header
        .lines()
        .any(|line| line == "format binary_little_endian 1.0")
    {
        return Err("Only binary little-endian PLY exports are supported".to_string());
    }
    let vertex_count = header
        .lines()
        .find_map(|line| line.strip_prefix("element vertex "))
        .ok_or_else(|| "The generated PLY does not declare its vertex count".to_string())?
        .parse::<usize>()
        .map_err(|_| "The generated PLY vertex count is invalid".to_string())?;
    let payload_size = vertex_count
        .checked_mul(PLY_VERTEX_STRIDE)
        .ok_or_else(|| "The generated PLY is too large to export".to_string())?;
    let expected_size = payload_start
        .checked_add(payload_size)
        .ok_or_else(|| "The generated PLY is too large to export".to_string())?;
    if bytes.len() != expected_size {
        return Err("The generated PLY vertex data has an unexpected size".to_string());
    }

    for vertex in bytes[payload_start..].chunks_exact_mut(PLY_VERTEX_STRIDE) {
        let z = f32::from_le_bytes(vertex[8..12].try_into().unwrap());
        vertex[8..12].copy_from_slice(&(-z).to_le_bytes());
    }

    let unity_comment = b"comment Unity-ready coordinates: Z axis flipped\n";
    let mut output = Vec::with_capacity(bytes.len() + unity_comment.len());
    output.extend_from_slice(&bytes[..header_end]);
    output.extend_from_slice(unity_comment);
    output.extend_from_slice(marker);
    output.extend_from_slice(&bytes[payload_start..]);
    Ok(output)
}

fn unity_compatible_obj(source: &str) -> Result<String, String> {
    // Unity's OBJ import convention leaves ScanLan OBJ bundles facing 180
    // degrees away from the equivalent Unity-ready PLY/splat exports. Bake the
    // same Y rotation users would otherwise need on the imported GameObject.
    transformed_obj(
        source,
        &CloudTransform {
            position: [0.0; 3],
            rotation: [0.0, 180.0, 0.0],
            scale: [1.0; 3],
        },
    )
}

fn unity_compatible_gaussian_ply(bytes: Vec<u8>) -> Result<Vec<u8>, String> {
    // A Gaussian's orientation is part of its covariance, so the handedness
    // conversion must transform more than its centre position.
    transformed_gaussian_ply(
        bytes,
        &CloudTransform {
            position: [0.0; 3],
            rotation: [0.0; 3],
            scale: [1.0, 1.0, -1.0],
        },
    )
}

fn write_export(destination: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = destination
        .parent()
        .ok_or_else(|| "Choose a folder for the export".to_string())?;
    if !parent.is_dir() {
        return Err("The selected export folder does not exist".to_string());
    }
    let file_name = destination
        .file_name()
        .ok_or_else(|| "Choose a file name for the exported PLY".to_string())?
        .to_string_lossy();
    let token = Uuid::new_v4();
    let temporary = parent.join(format!(".{file_name}.{token}.tmp"));
    let backup = parent.join(format!(".{file_name}.{token}.bak"));

    File::create(&temporary)
        .and_then(|mut file| {
            file.write_all(bytes)?;
            file.sync_all()
        })
        .map_err(|error| format!("Could not write the export: {error}"))?;

    let had_destination = destination.exists();
    if had_destination {
        if let Err(error) = fs::rename(destination, &backup) {
            fs::remove_file(&temporary).ok();
            return Err(format!("Could not replace the selected export: {error}"));
        }
    }
    if let Err(error) = fs::rename(&temporary, destination) {
        if had_destination {
            fs::rename(&backup, destination).ok();
        }
        fs::remove_file(&temporary).ok();
        return Err(format!("Could not finish the export: {error}"));
    }
    if had_destination {
        fs::remove_file(backup).ok();
    }
    Ok(())
}

#[tauri::command]
pub async fn export_ply(
    project_path: String,
    destination_path: String,
    transform: CloudTransform,
    clip_bounds: Option<BoundingBoxClip>,
) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        export_ply_blocking(project_path, destination_path, transform, clip_bounds)
    })
    .await
    .map_err(|error| error.to_string())?
}

fn export_ply_blocking(
    project_path: String,
    destination_path: String,
    transform: CloudTransform,
    clip_bounds: Option<BoundingBoxClip>,
) -> Result<String, String> {
    validate_cloud_transform(&transform)?;
    validate_clip_bounds(&clip_bounds)?;
    let root = PathBuf::from(project_path);
    let project = storage::read_project(&root)?;
    if project.processing_status != "complete" {
        return Err("Build the point cloud before exporting it".to_string());
    }

    let source = root.join("outputs").join("room-cloud.ply");
    if !source.is_file() {
        return Err("The reconstructed room-cloud.ply could not be found".to_string());
    }
    let mut destination = PathBuf::from(destination_path);
    if destination.as_os_str().is_empty() || !destination.is_absolute() {
        return Err("Choose a valid destination for the exported PLY".to_string());
    }
    if !destination
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("ply"))
    {
        destination.set_extension("ply");
    }
    if destination.exists() && fs::canonicalize(&source).ok() == fs::canonicalize(&destination).ok()
    {
        return Err(
            "Choose a different location so the project's source PLY stays unchanged".to_string(),
        );
    }

    let bytes = fs::read(&source)
        .map_err(|error| format!("Could not read the reconstructed PLY: {error}"))?;
    let transformed = transformed_cloud_ply(bytes, &transform)?;
    let clipped = match &clip_bounds {
        Some(bounds) => clipped_binary_ply(transformed, bounds, "point cloud")?,
        None => transformed,
    };
    let unity_bytes = unity_compatible_ply(clipped)?;
    write_export(&destination, &unity_bytes)?;
    Ok(destination.to_string_lossy().into_owned())
}

#[tauri::command]
pub async fn export_textured_mesh(
    project_path: String,
    destination_path: String,
    transform: CloudTransform,
    clip_bounds: Option<BoundingBoxClip>,
) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        export_textured_mesh_blocking(project_path, destination_path, transform, clip_bounds)
    })
    .await
    .map_err(|error| error.to_string())?
}

fn export_textured_mesh_blocking(
    project_path: String,
    destination_path: String,
    transform: CloudTransform,
    clip_bounds: Option<BoundingBoxClip>,
) -> Result<String, String> {
    validate_cloud_transform(&transform)?;
    validate_clip_bounds(&clip_bounds)?;
    let root = PathBuf::from(project_path);
    let project = storage::read_project(&root)?;
    if project.processing_status != "complete" {
        return Err("Build the 3D model before exporting its textured mesh".to_string());
    }
    let output_root = root.join("outputs");
    let source_obj = output_root.join("room-mesh.obj");
    let source_mtl = output_root.join("room-mesh.mtl");
    let source_texture = output_root.join("room-texture.png");
    if !source_obj.is_file() || !source_mtl.is_file() || !source_texture.is_file() {
        return Err(
            "The reconstructed textured-mesh bundle could not be found; rebuild the model"
                .to_string(),
        );
    }

    let mut destination_obj = PathBuf::from(destination_path);
    if destination_obj.as_os_str().is_empty() || !destination_obj.is_absolute() {
        return Err("Choose a valid destination for the textured OBJ".to_string());
    }
    if !destination_obj
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("obj"))
    {
        destination_obj.set_extension("obj");
    }
    let stem = destination_obj
        .file_stem()
        .ok_or_else(|| "Choose a file name for the textured OBJ".to_string())?
        .to_string_lossy()
        .into_owned();
    let destination_mtl = destination_obj.with_file_name(format!("{stem}.mtl"));
    let destination_texture = destination_obj.with_file_name(format!("{stem}-texture.png"));
    let mtl_name = destination_mtl
        .file_name()
        .ok_or_else(|| "Choose a valid OBJ destination".to_string())?
        .to_string_lossy();
    let texture_name = destination_texture
        .file_name()
        .ok_or_else(|| "Choose a valid texture destination".to_string())?
        .to_string_lossy();

    let source_obj = fs::read_to_string(source_obj)
        .map_err(|error| format!("Could not read the reconstructed OBJ: {error}"))?;
    let transformed = transformed_obj(&source_obj, &transform)?;
    let clipped = match &clip_bounds {
        Some(bounds) => clipped_obj(&transformed, bounds)?,
        None => transformed,
    };
    let unity_obj = unity_compatible_obj(&clipped)?;
    let obj = unity_obj.replace("mtllib room-mesh.mtl", &format!("mtllib {mtl_name}"));
    let mtl = fs::read_to_string(source_mtl)
        .map_err(|error| format!("Could not read the reconstructed material: {error}"))?
        .replace("map_Kd room-texture.png", &format!("map_Kd {texture_name}"));
    let texture = fs::read(source_texture)
        .map_err(|error| format!("Could not read the reconstructed texture: {error}"))?;

    write_export(&destination_texture, &texture)?;
    write_export(&destination_mtl, mtl.as_bytes())?;
    write_export(&destination_obj, obj.as_bytes())?;
    Ok(destination_obj.to_string_lossy().into_owned())
}

#[derive(Clone, Copy)]
enum PlyScalarType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl PlyScalarType {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "char" | "int8" => Some(Self::I8),
            "uchar" | "uint8" => Some(Self::U8),
            "short" | "int16" => Some(Self::I16),
            "ushort" | "uint16" => Some(Self::U16),
            "int" | "int32" => Some(Self::I32),
            "uint" | "uint32" => Some(Self::U32),
            "float" | "float32" => Some(Self::F32),
            "double" | "float64" => Some(Self::F64),
            _ => None,
        }
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    fn read(self, bytes: &[u8]) -> f32 {
        match self {
            Self::I8 => bytes[0] as i8 as f32,
            Self::U8 => bytes[0] as f32,
            Self::I16 => i16::from_le_bytes(bytes.try_into().unwrap()) as f32,
            Self::U16 => u16::from_le_bytes(bytes.try_into().unwrap()) as f32,
            Self::I32 => i32::from_le_bytes(bytes.try_into().unwrap()) as f32,
            Self::U32 => u32::from_le_bytes(bytes.try_into().unwrap()) as f32,
            Self::F32 => f32::from_le_bytes(bytes.try_into().unwrap()),
            Self::F64 => f64::from_le_bytes(bytes.try_into().unwrap()) as f32,
        }
    }

    fn write(self, bytes: &mut [u8], value: f32) -> Result<(), String> {
        match self {
            Self::F32 => bytes.copy_from_slice(&value.to_le_bytes()),
            Self::F64 => bytes.copy_from_slice(&(value as f64).to_le_bytes()),
            _ => {
                return Err(
                    "Gaussian geometry properties must use floating-point values".to_string(),
                )
            }
        }
        Ok(())
    }
}

type Matrix3 = [[f64; 3]; 3];

fn matrix_product(left: Matrix3, right: Matrix3) -> Matrix3 {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|axis| left[row][axis] * right[axis][column])
                .sum()
        })
    })
}

fn matrix_determinant(matrix: Matrix3) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn quaternion_matrix(mut quaternion: [f64; 4]) -> Matrix3 {
    let norm = quaternion
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm < 1e-12 {
        quaternion = [1.0, 0.0, 0.0, 0.0];
    } else {
        quaternion.iter_mut().for_each(|value| *value /= norm);
    }
    let [w, x, y, z] = quaternion;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn matrix_quaternion(matrix: Matrix3) -> [f32; 4] {
    let trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
    let quaternion = if trace > 0.0 {
        let scale = (trace + 1.0).sqrt() * 2.0;
        [
            0.25 * scale,
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
        ]
    } else if matrix[0][0] > matrix[1][1] && matrix[0][0] > matrix[2][2] {
        let scale = (1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2])
            .max(0.0)
            .sqrt()
            * 2.0;
        [
            (matrix[2][1] - matrix[1][2]) / scale,
            0.25 * scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
        ]
    } else if matrix[1][1] > matrix[2][2] {
        let scale = (1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2])
            .max(0.0)
            .sqrt()
            * 2.0;
        [
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            0.25 * scale,
            (matrix[1][2] + matrix[2][1]) / scale,
        ]
    } else {
        let scale = (1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1])
            .max(0.0)
            .sqrt()
            * 2.0;
        [
            (matrix[1][0] - matrix[0][1]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
            (matrix[1][2] + matrix[2][1]) / scale,
            0.25 * scale,
        ]
    };
    let norm = quaternion
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm < 1e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        quaternion.map(|value| (value / norm) as f32)
    }
}

fn symmetric_eigen(mut matrix: Matrix3) -> ([f64; 3], Matrix3) {
    let mut vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        let mut pair = (0, 1);
        let mut largest = matrix[0][1].abs();
        for candidate in [(0, 2), (1, 2)] {
            if matrix[candidate.0][candidate.1].abs() > largest {
                pair = candidate;
                largest = matrix[candidate.0][candidate.1].abs();
            }
        }
        let diagonal_scale = matrix[0][0].abs() + matrix[1][1].abs() + matrix[2][2].abs();
        if largest <= 1e-14 * diagonal_scale.max(1.0) {
            break;
        }
        let (p, q) = pair;
        let angle = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[q][q] - matrix[p][p]);
        let (sine, cosine) = angle.sin_cos();
        let app = matrix[p][p];
        let aqq = matrix[q][q];
        let apq = matrix[p][q];
        matrix[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        for axis in 0..3 {
            if axis == p || axis == q {
                continue;
            }
            let aip = matrix[axis][p];
            let aiq = matrix[axis][q];
            matrix[axis][p] = cosine * aip - sine * aiq;
            matrix[p][axis] = matrix[axis][p];
            matrix[axis][q] = sine * aip + cosine * aiq;
            matrix[q][axis] = matrix[axis][q];
        }
        for row in &mut vectors {
            let vip = row[p];
            let viq = row[q];
            row[p] = cosine * vip - sine * viq;
            row[q] = sine * vip + cosine * viq;
        }
    }

    let values = [matrix[0][0], matrix[1][1], matrix[2][2]];
    let mut order = [0, 1, 2];
    order.sort_by(|left, right| values[*right].total_cmp(&values[*left]));
    let sorted_values = order.map(|axis| values[axis]);
    let mut sorted_vectors = std::array::from_fn(|row| order.map(|axis| vectors[row][axis]));
    if matrix_determinant(sorted_vectors) < 0.0 {
        for row in &mut sorted_vectors {
            row[2] = -row[2];
        }
    }
    (sorted_values, sorted_vectors)
}

fn gaussian_edit_matrix(transform: &CloudTransform) -> Matrix3 {
    let columns = [
        transformed_direction([1.0, 0.0, 0.0], transform),
        transformed_direction([0.0, 1.0, 0.0], transform),
        transformed_direction([0.0, 0.0, 1.0], transform),
    ];
    std::array::from_fn(|row| std::array::from_fn(|column| columns[column][row] as f64))
}

fn read_ply_property(
    record: &[u8],
    properties: &HashMap<String, (usize, PlyScalarType)>,
    name: &str,
) -> f32 {
    let (offset, scalar_type) = properties[name];
    scalar_type.read(&record[offset..offset + scalar_type.size()])
}

fn write_ply_property(
    record: &mut [u8],
    properties: &HashMap<String, (usize, PlyScalarType)>,
    name: &str,
    value: f32,
) -> Result<(), String> {
    let (offset, scalar_type) = properties[name];
    scalar_type.write(&mut record[offset..offset + scalar_type.size()], value)
}

fn transformed_gaussian_ply(
    mut bytes: Vec<u8>,
    transform: &CloudTransform,
) -> Result<Vec<u8>, String> {
    let marker = b"end_header\n";
    let payload_start = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .map(|position| position + marker.len())
        .ok_or_else(|| "The Gaussian PLY header is invalid".to_string())?;
    let header = std::str::from_utf8(&bytes[..payload_start])
        .map_err(|_| "The Gaussian PLY header is not valid ASCII".to_string())?;
    let mut vertex_count = None;
    let mut vertex_stride = 0usize;
    let mut in_vertex_element = false;
    let mut properties: HashMap<String, (usize, PlyScalarType)> = HashMap::new();
    let mut binary_little_endian = false;
    for line in header.lines() {
        let fields = line
            .trim_end_matches('\r')
            .split_whitespace()
            .collect::<Vec<_>>();
        if fields.first() == Some(&"format") {
            binary_little_endian = fields.get(1) == Some(&"binary_little_endian");
        } else if fields.first() == Some(&"element") && fields.len() >= 3 {
            in_vertex_element = fields[1] == "vertex";
            if in_vertex_element {
                vertex_count = Some(
                    fields[2]
                        .parse::<usize>()
                        .map_err(|_| "The Gaussian PLY vertex count is invalid".to_string())?,
                );
            }
        } else if fields.first() == Some(&"property") && in_vertex_element {
            if fields.get(1) == Some(&"list") || fields.len() < 3 {
                return Err(
                    "List-valued Gaussian PLY vertex properties are unsupported".to_string()
                );
            }
            let scalar_type = PlyScalarType::parse(fields[1])
                .ok_or_else(|| format!("Unsupported Gaussian PLY property type: {}", fields[1]))?;
            properties.insert(fields[2].to_string(), (vertex_stride, scalar_type));
            vertex_stride = vertex_stride
                .checked_add(scalar_type.size())
                .ok_or_else(|| "The Gaussian PLY vertex layout is too large".to_string())?;
        }
    }
    if !binary_little_endian {
        return Err("Only binary little-endian Gaussian PLY files can be transformed".to_string());
    }
    let vertex_count =
        vertex_count.ok_or_else(|| "The Gaussian PLY has no vertices".to_string())?;
    for property in [
        "x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        if !properties.contains_key(property) {
            return Err(format!("The Gaussian PLY is missing property {property}"));
        }
    }
    let payload_end = payload_start
        .checked_add(
            vertex_count
                .checked_mul(vertex_stride)
                .ok_or_else(|| "The Gaussian PLY is too large".to_string())?,
        )
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| "The Gaussian PLY vertex payload is incomplete".to_string())?;
    let edit_matrix = gaussian_edit_matrix(transform);
    let uniform_edit = if transform.scale[0] > 0.0
        && transform
            .scale
            .iter()
            .all(|value| (*value - transform.scale[0]).abs() < 1e-6)
    {
        let scale = transform.scale[0] as f64;
        Some((scale, edit_matrix.map(|row| row.map(|value| value / scale))))
    } else {
        None
    };
    let scale_names = ["scale_0", "scale_1", "scale_2"];
    let rotation_names = ["rot_0", "rot_1", "rot_2", "rot_3"];
    let has_normals = ["nx", "ny", "nz"]
        .iter()
        .all(|name| properties.contains_key(*name));
    for record in bytes[payload_start..payload_end].chunks_exact_mut(vertex_stride) {
        let position = [
            read_ply_property(record, &properties, "x"),
            read_ply_property(record, &properties, "y"),
            read_ply_property(record, &properties, "z"),
        ];
        let transformed = transformed_position(position, transform);
        for (name, value) in ["x", "y", "z"].into_iter().zip(transformed) {
            write_ply_property(record, &properties, name, value)?;
        }

        if has_normals {
            let normal = transformed_normal(
                [
                    read_ply_property(record, &properties, "nx"),
                    read_ply_property(record, &properties, "ny"),
                    read_ply_property(record, &properties, "nz"),
                ],
                transform,
            );
            for (name, value) in ["nx", "ny", "nz"].into_iter().zip(normal) {
                write_ply_property(record, &properties, name, value)?;
            }
        }

        let gaussian_rotation = quaternion_matrix([
            read_ply_property(record, &properties, rotation_names[0]) as f64,
            read_ply_property(record, &properties, rotation_names[1]) as f64,
            read_ply_property(record, &properties, rotation_names[2]) as f64,
            read_ply_property(record, &properties, rotation_names[3]) as f64,
        ]);
        let log_scales = scale_names.map(|name| read_ply_property(record, &properties, name));
        if let Some((uniform_scale, edit_rotation)) = uniform_edit {
            for axis in 0..3 {
                write_ply_property(
                    record,
                    &properties,
                    scale_names[axis],
                    log_scales[axis] + uniform_scale.ln() as f32,
                )?;
            }
            let quaternion = matrix_quaternion(matrix_product(edit_rotation, gaussian_rotation));
            for (axis, value) in quaternion.into_iter().enumerate() {
                write_ply_property(record, &properties, rotation_names[axis], value)?;
            }
            continue;
        }
        let mut gaussian_basis = gaussian_rotation;
        for axis in 0..3 {
            let scale = (log_scales[axis] as f64).exp();
            if !scale.is_finite() || scale <= 0.0 {
                return Err("The Gaussian PLY contains an invalid scale".to_string());
            }
            for row in &mut gaussian_basis {
                row[axis] *= scale;
            }
        }
        let deformed_basis = matrix_product(edit_matrix, gaussian_basis);
        // Gaussian scale and rotation encode a covariance basis. Under a
        // general affine edit A, covariance becomes (A*B)*(A*B)^T; its
        // eigensystem converts that result back to 3DGS scale/quaternion form.
        let covariance = std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                (0..3)
                    .map(|axis| deformed_basis[row][axis] * deformed_basis[column][axis])
                    .sum()
            })
        });
        let (variances, orientation) = symmetric_eigen(covariance);
        if variances
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err("The transformed Gaussian covariance is invalid".to_string());
        }
        for (axis, variance) in variances.into_iter().enumerate() {
            write_ply_property(
                record,
                &properties,
                scale_names[axis],
                (variance.sqrt().ln()) as f32,
            )?;
        }
        let quaternion = matrix_quaternion(orientation);
        for (axis, value) in quaternion.into_iter().enumerate() {
            write_ply_property(record, &properties, rotation_names[axis], value)?;
        }
    }
    Ok(bytes)
}

fn clipped_binary_ply(
    bytes: Vec<u8>,
    bounds: &BoundingBoxClip,
    content_name: &str,
) -> Result<Vec<u8>, String> {
    let marker = b"end_header\n";
    let payload_start = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .map(|position| position + marker.len())
        .ok_or_else(|| format!("The {content_name} PLY header is invalid"))?;
    let header = std::str::from_utf8(&bytes[..payload_start])
        .map_err(|_| format!("The {content_name} PLY header is not valid ASCII"))?;
    let mut vertex_count = None;
    let mut vertex_stride = 0usize;
    let mut in_vertex_element = false;
    let mut properties: HashMap<String, (usize, PlyScalarType)> = HashMap::new();
    let mut binary_little_endian = false;
    for line in header.lines() {
        let fields = line
            .trim_end_matches('\r')
            .split_whitespace()
            .collect::<Vec<_>>();
        if fields.first() == Some(&"format") {
            binary_little_endian = fields.get(1) == Some(&"binary_little_endian");
        } else if fields.first() == Some(&"element") && fields.len() >= 3 {
            in_vertex_element = fields[1] == "vertex";
            if in_vertex_element {
                vertex_count = Some(
                    fields[2]
                        .parse::<usize>()
                        .map_err(|_| format!("The {content_name} PLY vertex count is invalid"))?,
                );
            }
        } else if fields.first() == Some(&"property") && in_vertex_element {
            if fields.get(1) == Some(&"list") || fields.len() < 3 {
                return Err(format!(
                    "List-valued {content_name} PLY vertex properties are unsupported"
                ));
            }
            let scalar_type = PlyScalarType::parse(fields[1]).ok_or_else(|| {
                format!(
                    "Unsupported {content_name} PLY property type: {}",
                    fields[1]
                )
            })?;
            properties.insert(fields[2].to_string(), (vertex_stride, scalar_type));
            vertex_stride = vertex_stride
                .checked_add(scalar_type.size())
                .ok_or_else(|| format!("The {content_name} PLY vertex layout is too large"))?;
        }
    }
    if !binary_little_endian {
        return Err(format!(
            "Only binary little-endian {content_name} PLY files can be clipped"
        ));
    }
    let vertex_count = vertex_count
        .ok_or_else(|| format!("The {content_name} PLY does not declare its vertex count"))?;
    if vertex_stride == 0 {
        return Err(format!("The {content_name} PLY has no vertex properties"));
    }
    for property in ["x", "y", "z"] {
        if !properties.contains_key(property) {
            return Err(format!(
                "The {content_name} PLY is missing property {property}"
            ));
        }
    }
    let vertex_payload_size = vertex_count
        .checked_mul(vertex_stride)
        .ok_or_else(|| format!("The {content_name} PLY is too large"))?;
    let vertex_payload_end = payload_start
        .checked_add(vertex_payload_size)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| format!("The {content_name} PLY vertex payload is incomplete"))?;
    let property = |record: &[u8], name: &str| -> f32 {
        let (offset, scalar_type) = properties[name];
        scalar_type.read(&record[offset..offset + scalar_type.size()])
    };
    let mut retained = Vec::with_capacity(vertex_payload_size);
    for record in bytes[payload_start..vertex_payload_end].chunks_exact(vertex_stride) {
        let position = [
            property(record, "x"),
            property(record, "y"),
            property(record, "z"),
        ];
        if clip_contains(bounds, position) {
            retained.extend_from_slice(record);
        }
    }
    let retained_count = retained.len() / vertex_stride;
    if retained_count == 0 {
        return Err(format!(
            "The bounding box does not contain any {content_name} vertices"
        ));
    }
    let declaration = format!("element vertex {vertex_count}");
    if !header.contains(&declaration) {
        return Err(format!(
            "The {content_name} PLY vertex declaration is invalid"
        ));
    }
    let clipped_header =
        header.replacen(&declaration, &format!("element vertex {retained_count}"), 1);
    let mut output = Vec::with_capacity(
        clipped_header.len() + retained.len() + bytes.len() - vertex_payload_end,
    );
    output.extend_from_slice(clipped_header.as_bytes());
    output.extend_from_slice(&retained);
    output.extend_from_slice(&bytes[vertex_payload_end..]);
    Ok(output)
}

fn gaussian_splat_preview_path(source: &Path) -> PathBuf {
    let stem = source
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("room-splat");
    source.with_file_name(format!("{stem}.preview.splat"))
}

fn compact_splat_preview(bytes: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    if bytes.is_empty() || bytes.len() % 32 != 0 {
        return Err("The Gaussian preview is incomplete".to_string());
    }
    if limit == 0 {
        return Err("The Gaussian preview limit must be positive".to_string());
    }
    let splat_count = bytes.len() / 32;
    if splat_count <= limit {
        return Ok(bytes.to_vec());
    }
    if limit == 1 {
        return Ok(bytes[..32].to_vec());
    }
    let mut compact = Vec::with_capacity(limit * 32);
    for target_index in 0..limit {
        let source_index = target_index * (splat_count - 1) / (limit - 1);
        compact.extend_from_slice(&bytes[source_index * 32..source_index * 32 + 32]);
    }
    Ok(compact)
}

fn convert_3dgs_ply_to_splat(source: &Path) -> Result<Vec<u8>, String> {
    const SH_C0: f32 = 0.282_094_8;
    let file = File::open(source)
        .map_err(|error| format!("Could not open the canonical Gaussian PLY: {error}"))?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut vertex_count = None;
    let mut in_vertex_element = false;
    let mut properties: HashMap<String, (usize, PlyScalarType)> = HashMap::new();
    let mut vertex_stride = 0usize;
    let mut saw_binary_little_endian = false;

    loop {
        line.clear();
        if reader
            .read_line(&mut line)
            .map_err(|error| format!("Could not read the Gaussian PLY header: {error}"))?
            == 0
        {
            return Err("The Gaussian PLY header is incomplete".to_string());
        }
        let header_line = line.trim_end_matches(|value| value == '\r' || value == '\n');
        let fields: Vec<_> = header_line.split_whitespace().collect();
        if fields.first() == Some(&"format") {
            saw_binary_little_endian = fields.get(1) == Some(&"binary_little_endian");
        } else if fields.first() == Some(&"element") && fields.len() >= 3 {
            in_vertex_element = fields[1] == "vertex";
            if in_vertex_element {
                vertex_count = Some(
                    fields[2]
                        .parse::<usize>()
                        .map_err(|_| "The Gaussian PLY has an invalid vertex count".to_string())?,
                );
            }
        } else if fields.first() == Some(&"property") && in_vertex_element {
            if fields.get(1) == Some(&"list") || fields.len() < 3 {
                return Err(
                    "List-valued Gaussian PLY vertex properties are unsupported".to_string()
                );
            }
            let scalar_type = PlyScalarType::parse(fields[1])
                .ok_or_else(|| format!("Unsupported Gaussian PLY property type: {}", fields[1]))?;
            properties.insert(fields[2].to_string(), (vertex_stride, scalar_type));
            vertex_stride = vertex_stride
                .checked_add(scalar_type.size())
                .ok_or_else(|| "The Gaussian PLY vertex layout is too large".to_string())?;
        }
        if header_line == "end_header" {
            break;
        }
    }
    if !saw_binary_little_endian {
        return Err("Only binary little-endian Gaussian PLY files can be previewed".to_string());
    }
    let vertex_count =
        vertex_count.ok_or_else(|| "The Gaussian PLY has no vertices".to_string())?;
    if vertex_count == 0 || vertex_stride == 0 {
        return Err("The Gaussian PLY contains no splats".to_string());
    }
    for required in [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        if !properties.contains_key(required) {
            return Err(format!("The Gaussian PLY is missing property {required}"));
        }
    }

    let property = |record: &[u8], name: &str| -> f32 {
        let (offset, scalar_type) = properties[name];
        scalar_type.read(&record[offset..offset + scalar_type.size()])
    };
    let output_size = vertex_count
        .checked_mul(32)
        .ok_or_else(|| "The Gaussian preview is too large".to_string())?;
    let mut output = Vec::with_capacity(output_size);
    let mut record = vec![0u8; vertex_stride];
    for _ in 0..vertex_count {
        reader
            .read_exact(&mut record)
            .map_err(|error| format!("The Gaussian PLY vertex payload is incomplete: {error}"))?;
        for name in ["x", "y", "z"] {
            output.extend_from_slice(&property(&record, name).to_le_bytes());
        }
        for name in ["scale_0", "scale_1", "scale_2"] {
            let scale = property(&record, name).exp();
            if !scale.is_finite() {
                return Err("The Gaussian PLY contains an invalid scale".to_string());
            }
            output.extend_from_slice(&scale.to_le_bytes());
        }
        for name in ["f_dc_0", "f_dc_1", "f_dc_2"] {
            let color = (0.5 + SH_C0 * property(&record, name)).clamp(0.0, 1.0);
            output.push((color * 255.0).round() as u8);
        }
        let opacity_logit = property(&record, "opacity");
        let opacity = if opacity_logit >= 0.0 {
            1.0 / (1.0 + (-opacity_logit).exp())
        } else {
            let exponent = opacity_logit.exp();
            exponent / (1.0 + exponent)
        };
        output.push((opacity.clamp(0.0, 1.0) * 255.0).round() as u8);

        let mut quaternion = [
            property(&record, "rot_0"),
            property(&record, "rot_1"),
            property(&record, "rot_2"),
            property(&record, "rot_3"),
        ];
        let norm = quaternion
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if norm > 1e-8 && norm.is_finite() {
            for value in &mut quaternion {
                *value /= norm;
            }
        } else {
            quaternion = [1.0, 0.0, 0.0, 0.0];
        }
        for value in quaternion {
            output.push((value.mul_add(128.0, 128.0).round().clamp(0.0, 255.0)) as u8);
        }
    }
    Ok(output)
}

fn ensure_gaussian_splat_preview(source: &Path) -> Result<PathBuf, String> {
    const MAX_PREVIEW_SPLATS: usize = 500_000;
    let preview = gaussian_splat_preview_path(source);
    let preview_is_current = fs::metadata(&preview).ok().is_some_and(|preview_metadata| {
        preview_metadata.len() > 0
            && preview_metadata.len() % 32 == 0
            && match (
                preview_metadata.modified(),
                fs::metadata(source).and_then(|value| value.modified()),
            ) {
                (Ok(preview_time), Ok(source_time)) => preview_time >= source_time,
                _ => true,
            }
    });
    if preview_is_current {
        let bytes = fs::read(&preview)
            .map_err(|error| format!("Could not read the Gaussian preview cache: {error}"))?;
        if bytes.len() / 32 > MAX_PREVIEW_SPLATS {
            let compact = compact_splat_preview(&bytes, MAX_PREVIEW_SPLATS)?;
            write_export(&preview, &compact).map_err(|error| {
                format!("Could not compact the realtime Gaussian preview: {error}")
            })?;
        }
    } else {
        let bytes = convert_3dgs_ply_to_splat(source)?;
        let compact = compact_splat_preview(&bytes, MAX_PREVIEW_SPLATS)?;
        write_export(&preview, &compact)
            .map_err(|error| format!("Could not cache the realtime Gaussian preview: {error}"))?;
    }
    Ok(preview)
}

#[tauri::command]
pub async fn load_gaussian_splat(project_path: String) -> Result<Response, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let root = PathBuf::from(project_path);
        let project = storage::read_project(&root)?;
        // The reconstruction worker can publish its artifacts before the
        // separately packaged splat trainer finishes. `active_job` is the
        // durable source of truth across that worker boundary.
        if gaussian_splat_preview_is_live(&project) {
            let live_preview = root.join("outputs").join("room-splat.preview.splat");
            let bytes = fs::read(&live_preview).map_err(|_| {
                "The trainer has not published its first live splat preview yet".to_string()
            })?;
            if bytes.is_empty() || bytes.len() % 32 != 0 {
                return Err("The live splat preview is still being published".to_string());
            }
            return Ok(Response::new(bytes));
        }
        let artifact = project
            .artifacts
            .gaussian_splat
            .as_ref()
            .filter(|artifact| !artifact.stale)
            .ok_or_else(|| "Build a Gaussian splat before previewing it".to_string())?;
        let source = root.join(&artifact.path);
        if !source.is_file() {
            return Err("The Gaussian splat artifact is missing".to_string());
        }
        let preview = ensure_gaussian_splat_preview(&source)?;
        let bytes = fs::read(preview)
            .map_err(|error| format!("Could not read the Gaussian splat preview: {error}"))?;
        Ok(Response::new(bytes))
    })
    .await
    .map_err(|error| error.to_string())?
}

fn gaussian_splat_preview_is_live(project: &ProjectSummary) -> bool {
    project.processing_status == "processing" || project.active_job.is_some()
}

#[tauri::command]
pub async fn export_gaussian_splat(
    project_path: String,
    destination_path: String,
    transform: CloudTransform,
    clip_bounds: Option<BoundingBoxClip>,
) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        export_gaussian_splat_blocking(project_path, destination_path, transform, clip_bounds)
    })
    .await
    .map_err(|error| error.to_string())?
}

fn export_gaussian_splat_blocking(
    project_path: String,
    destination_path: String,
    transform: CloudTransform,
    clip_bounds: Option<BoundingBoxClip>,
) -> Result<String, String> {
    validate_cloud_transform(&transform)?;
    validate_clip_bounds(&clip_bounds)?;
    let root = PathBuf::from(project_path);
    let project = storage::read_project(&root)?;
    let relative = project
        .artifacts
        .gaussian_splat
        .as_ref()
        .filter(|artifact| !artifact.stale)
        .map(|artifact| artifact.path.as_str())
        .unwrap_or("outputs/room-splat.ply");
    let source = root.join(relative);
    if !source.is_file() {
        return Err("Build a Gaussian splat before exporting it".to_string());
    }
    let mut destination = PathBuf::from(destination_path);
    if destination.as_os_str().is_empty() || !destination.is_absolute() {
        return Err("Choose a valid destination for the Gaussian PLY".to_string());
    }
    if !destination
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("ply"))
    {
        destination.set_extension("ply");
    }
    let bytes =
        fs::read(&source).map_err(|error| format!("Could not read Gaussian PLY: {error}"))?;
    let transformed = transformed_gaussian_ply(bytes, &transform)?;
    let clipped = match &clip_bounds {
        Some(bounds) => clipped_binary_ply(transformed, bounds, "Gaussian splat")?,
        None => transformed,
    };
    let unity_bytes = unity_compatible_gaussian_ply(clipped)?;
    write_export(&destination, &unity_bytes)?;
    let stem = destination
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("room-splat");
    let material_source = root.join("outputs").join("room-splat-material.npz");
    let material_destination = destination.with_file_name(format!("{stem}.material.npz"));
    let material_exported = clip_bounds.is_none() && material_source.is_file();
    if material_exported {
        let material_bytes = fs::read(&material_source)
            .map_err(|error| format!("Could not read Gaussian material sidecar: {error}"))?;
        write_export(&material_destination, &material_bytes)?;
    } else if material_destination.is_file() {
        fs::remove_file(&material_destination).map_err(|error| {
            format!("Could not remove stale Gaussian material sidecar: {error}")
        })?;
    }
    for (source_name, suffix) in [
        ("room-splat.transform.json", "transform.json"),
        ("splat-manifest.json", "manifest.json"),
    ] {
        let sidecar = root.join("outputs").join(source_name);
        if sidecar.is_file() {
            let destination_sidecar = destination.with_file_name(format!("{stem}.{suffix}"));
            let mut sidecar_bytes = fs::read(&sidecar).map_err(|error| error.to_string())?;
            if source_name == "room-splat.transform.json" {
                let mut metadata = serde_json::from_slice::<serde_json::Value>(&sidecar_bytes)
                    .unwrap_or_else(|_| serde_json::json!({ "schemaVersion": 1 }));
                if let Some(object) = metadata.as_object_mut() {
                    object.insert(
                        "applyAtGameObject".to_string(),
                        serde_json::Value::Bool(false),
                    );
                    object.insert(
                        "editPoseBakedIntoPly".to_string(),
                        serde_json::to_value(&transform).unwrap(),
                    );
                    object.insert(
                        "note".to_string(),
                        serde_json::Value::String(
                            "The ScanLan edit pose and Unity Z-axis handedness conversion are baked into Gaussian means, normals, scales, and rotations."
                                .to_string(),
                        ),
                    );
                    object.insert(
                        "unityCoordinateConversionBakedIntoPly".to_string(),
                        serde_json::Value::String("flip_z".to_string()),
                    );
                }
                sidecar_bytes = serde_json::to_vec_pretty(&metadata).map_err(|error| {
                    format!("Could not serialize Gaussian transform metadata: {error}")
                })?;
                sidecar_bytes.push(b'\n');
            } else if source_name == "splat-manifest.json" {
                let mut metadata = serde_json::from_slice::<serde_json::Value>(&sidecar_bytes)
                    .unwrap_or_else(|_| serde_json::json!({ "schemaVersion": 1 }));
                if let Some(object) = metadata.as_object_mut() {
                    let convention = object
                        .entry("coordinateConvention")
                        .or_insert_with(|| serde_json::json!({}));
                    if let Some(convention) = convention.as_object_mut() {
                        convention.insert(
                            "handedness".to_string(),
                            serde_json::Value::String("left".to_string()),
                        );
                        convention.insert(
                            "worldAxes".to_string(),
                            serde_json::Value::String("unity_x_right_y_up_z_forward".to_string()),
                        );
                        convention.insert(
                            "conversionBakedIntoPly".to_string(),
                            serde_json::Value::String("flip_z".to_string()),
                        );
                    }
                    if let Some(material) = object.get_mut("material") {
                        if material_exported {
                            if let Some(material) = material.as_object_mut() {
                                material.insert(
                                    "path".to_string(),
                                    serde_json::Value::String(format!("{stem}.material.npz")),
                                );
                                material.insert(
                                    "alignedWith".to_string(),
                                    serde_json::Value::String(format!(
                                        "{} vertex order",
                                        destination
                                            .file_name()
                                            .and_then(|value| value.to_str())
                                            .unwrap_or("Gaussian PLY")
                                    )),
                                );
                            }
                        } else {
                            object.remove("material");
                            object.insert(
                                "materialExport".to_string(),
                                serde_json::json!({
                                    "status": "omitted",
                                    "reason": if clip_bounds.is_some() {
                                        "bounding-box clipping changes Gaussian row alignment"
                                    } else {
                                        "the source material sidecar is unavailable"
                                    }
                                }),
                            );
                        }
                    }
                }
                sidecar_bytes = serde_json::to_vec_pretty(&metadata).map_err(|error| {
                    format!("Could not serialize Gaussian manifest metadata: {error}")
                })?;
                sidecar_bytes.push(b'\n');
            }
            write_export(&destination_sidecar, &sidecar_bytes)?;
        }
    }
    Ok(destination.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::{
        append_sensor_args, clipped_binary_ply, clipped_obj, compact_splat_preview,
        convert_3dgs_ply_to_splat, export_gaussian_splat_blocking, gaussian_edit_matrix,
        gaussian_splat_preview_is_live, managed_media_source_path, matrix_product,
        normalize_project, pack_preview_mesh, quaternion_matrix, read_supplemental_photo_manifest,
        save_live_reconstruction_preview, transformed_cloud_ply, transformed_gaussian_ply,
        transformed_normal, transformed_obj, transformed_position, unity_compatible_gaussian_ply,
        unity_compatible_obj, unity_compatible_ply, valid_packed_preview_mesh,
        validate_sensor_settings, LiveGeometryFrame, RealtimeEngineSnapshot,
    };
    use crate::models::{
        ArtifactSummary, BoundingBoxClip, CaptureSettings, CloudTransform, ProjectSummary,
    };
    use std::sync::{Arc, Mutex};
    use std::{fs, process::Command};

    #[test]
    fn managed_media_source_paths_cannot_escape_the_project_media_directory() {
        let root = std::env::temp_dir().join("scanlan-media-path-test");
        assert_eq!(
            managed_media_source_path(&root, "media/source.mp4").unwrap(),
            root.join("media").join("source.mp4")
        );
        for unsafe_path in [
            "../source.mp4",
            "media/../source.mp4",
            "other/source.mp4",
            "media/nested/source.mp4",
            "source.mp4",
        ] {
            assert!(managed_media_source_path(&root, unsafe_path).is_err());
        }
    }

    #[test]
    fn gaussian_export_preserves_material_alignment_or_omits_it_when_clipped() {
        let root = std::env::temp_dir().join(format!(
            "scanlan-material-splat-export-{}",
            uuid::Uuid::new_v4()
        ));
        let mut project = crate::storage::create_project(&root).unwrap();
        project.artifacts.gaussian_splat = Some(ArtifactSummary {
            path: "outputs/room-splat.ply".to_string(),
            material_path: Some("outputs/room-splat-material.npz".to_string()),
            refined_camera_path: None,
            status: "ready".to_string(),
            source_fingerprint: "fixture".to_string(),
            updated_at: "2026-08-11T00:00:00Z".to_string(),
            metric: true,
            stale: false,
        });
        crate::storage::write_project(&project).unwrap();
        let mut names = vec![
            "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
        ];
        let rest = (0..45)
            .map(|index| format!("f_rest_{index}"))
            .collect::<Vec<_>>();
        let trailing = [
            "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ];
        let properties = names
            .drain(..)
            .map(str::to_string)
            .chain(rest)
            .chain(trailing.into_iter().map(str::to_string))
            .collect::<Vec<_>>();
        let header = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex 1\n{}end_header\n",
            properties
                .iter()
                .map(|name| format!("property float {name}\n"))
                .collect::<String>()
        );
        let mut ply = header.into_bytes();
        for index in 0..properties.len() {
            let value = match properties[index].as_str() {
                "scale_0" | "scale_1" | "scale_2" => -2.0_f32,
                "rot_0" => 1.0_f32,
                _ => 0.0_f32,
            };
            ply.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(root.join("outputs/room-splat.ply"), ply).unwrap();
        fs::write(
            root.join("outputs/room-splat-material.npz"),
            b"material fixture",
        )
        .unwrap();
        fs::write(
            root.join("outputs/splat-manifest.json"),
            serde_json::to_vec(&serde_json::json!({
                "schemaVersion": 1,
                "coordinateConvention": {},
                "material": {"path": "room-splat-material.npz"}
            }))
            .unwrap(),
        )
        .unwrap();
        let destination = root.join("export.ply");
        let transform = CloudTransform {
            position: [0.0; 3],
            rotation: [0.0; 3],
            scale: [1.0; 3],
        };

        export_gaussian_splat_blocking(
            root.to_string_lossy().into_owned(),
            destination.to_string_lossy().into_owned(),
            transform.clone(),
            None,
        )
        .unwrap();
        assert_eq!(
            fs::read(root.join("export.material.npz")).unwrap(),
            b"material fixture"
        );
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("export.manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["material"]["path"], "export.material.npz");

        export_gaussian_splat_blocking(
            root.to_string_lossy().into_owned(),
            destination.to_string_lossy().into_owned(),
            transform,
            Some(BoundingBoxClip {
                min: [-1.0; 3],
                max: [1.0; 3],
            }),
        )
        .unwrap();
        assert!(!root.join("export.material.npz").exists());
        let clipped_manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("export.manifest.json")).unwrap()).unwrap();
        assert!(clipped_manifest.get("material").is_none());
        assert_eq!(clipped_manifest["materialExport"]["status"], "omitted");
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn legacy_supplemental_photos_are_visible_as_localized_attempts() {
        let root =
            std::env::temp_dir().join(format!("scanlan-supplemental-{}", uuid::Uuid::new_v4()));
        crate::storage::create_project(&root).unwrap();
        crate::storage::write_json(
            &root.join("supplemental-photos.json"),
            &serde_json::json!({
                "schemaVersion": 1,
                "photos": [{
                    "id": "legacy-photo",
                    "name": "legacy",
                    "path": "supplemental/legacy.png"
                }]
            }),
        )
        .unwrap();

        let manifest = read_supplemental_photo_manifest(&root).unwrap();
        let attempts = manifest["attempts"].as_array().unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0]["id"], "legacy-photo");
        assert_eq!(attempts[0]["status"], "localized");

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn final_live_point_packet_is_preserved_with_the_capture() {
        let root =
            std::env::temp_dir().join(format!("scanlan-live-preview-{}", uuid::Uuid::new_v4()));
        let phase_root = root.join("phases").join("phase-1");
        fs::create_dir_all(&phase_root).unwrap();
        fs::write(
            phase_root.join("live_loops.jsonl"),
            b"{\"schemaVersion\":1,\"accepted\":true,\"requiresProductionRevalidation\":true}\n",
        )
        .unwrap();
        let mut packet = Vec::new();
        packet.extend_from_slice(b"K2P1");
        packet.extend_from_slice(&7_u32.to_le_bytes());
        packet.extend_from_slice(&123_456_u64.to_le_bytes());
        packet.extend_from_slice(&12.0_f32.to_le_bytes());
        packet.extend_from_slice(&1_u32.to_le_bytes());
        packet.extend_from_slice(&1.0_f32.to_le_bytes());
        packet.extend_from_slice(&2.0_f32.to_le_bytes());
        packet.extend_from_slice(&3.0_f32.to_le_bytes());
        packet.extend_from_slice(&[10, 20, 30]);
        let mut snapshot = RealtimeEngineSnapshot::default();
        snapshot.points = Some(LiveGeometryFrame {
            frame_count: 7,
            packet: Arc::new(packet.clone()),
        });

        assert!(save_live_reconstruction_preview(
            &root,
            &phase_root,
            &Arc::new(Mutex::new(snapshot)),
        )
        .unwrap());
        assert_eq!(
            fs::read(phase_root.join("live-reconstruction.preview.bin")).unwrap(),
            packet
        );
        assert!(root.join("outputs/live/session.json").is_file());
        assert!(root.join("outputs/live/loops.jsonl").is_file());
        let session: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("outputs/live/session.json")).unwrap())
                .unwrap();
        assert_eq!(session["acceptedLoops"].as_array().unwrap().len(), 1);
        let ply = fs::read(root.join("outputs/live/latest-preview.ply")).unwrap();
        let glb = fs::read(root.join("outputs/live/latest-preview.glb")).unwrap();
        assert!(ply.starts_with(b"ply\nformat binary_little_endian 1.0\n"));
        assert_eq!(&glb[..4], b"glTF");
        assert_eq!(u32::from_le_bytes(glb[4..8].try_into().unwrap()), 2);
        assert_eq!(
            u32::from_le_bytes(glb[8..12].try_into().unwrap()) as usize,
            glb.len()
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn mesh_preview_is_packed_as_indexed_binary_geometry() {
        let path = std::env::temp_dir().join(format!("scanlan-mesh-{}.obj", uuid::Uuid::new_v4()));
        fs::write(
            &path,
            concat!(
                "v 1 2 3\n",
                "v 4 5 6\n",
                "v 7 8 9\n",
                "vt 0.1 0.2\n",
                "vt 0.3 0.4\n",
                "vt 0.5 0.6\n",
                "f 1/1 2/2 3/3\n"
            ),
        )
        .unwrap();
        let packed = pack_preview_mesh(&path, 1).unwrap();
        fs::remove_file(path).ok();

        assert_eq!(&packed[0..4], b"K2M1");
        assert_eq!(u32::from_le_bytes(packed[4..8].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(packed[8..12].try_into().unwrap()), 3);
        assert_eq!(f32::from_le_bytes(packed[12..16].try_into().unwrap()), 1.0);
        assert_eq!(packed.len(), 12 + 3 * 20 + 3 * 4);
        assert!(valid_packed_preview_mesh(&packed));
        assert!(!valid_packed_preview_mesh(&packed[..packed.len() - 1]));
    }

    #[test]
    fn canonical_gaussian_ply_converts_to_compact_preview() {
        let path = std::env::temp_dir().join(format!("scanlan-splat-{}.ply", uuid::Uuid::new_v4()));
        let names = [
            "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
            "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ];
        let mut source = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex 1\n{}end_header\n",
            names
                .iter()
                .map(|name| format!("property float {name}\n"))
                .collect::<String>()
        )
        .into_bytes();
        let dc = (0.25_f32 - 0.5) / 0.282_094_8;
        for value in [
            1.0_f32,
            2.0,
            3.0,
            dc,
            0.0,
            -dc,
            0.0,
            0.1_f32.ln(),
            0.2_f32.ln(),
            0.4_f32.ln(),
            1.0,
            0.0,
            0.0,
            0.0,
        ] {
            source.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&path, source).unwrap();
        let preview = convert_3dgs_ply_to_splat(&path).unwrap();
        fs::remove_file(path).ok();

        assert_eq!(preview.len(), 32);
        let floats: Vec<_> = preview[..24]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(&floats[..3], &[1.0, 2.0, 3.0]);
        assert!((floats[3] - 0.1).abs() < 1e-6);
        assert!((floats[4] - 0.2).abs() < 1e-6);
        assert!((floats[5] - 0.4).abs() < 1e-6);
        assert_eq!(&preview[24..28], &[64, 128, 191, 128]);
        assert_eq!(&preview[28..32], &[255, 128, 128, 128]);
    }

    #[test]
    fn oversized_gaussian_preview_is_evenly_compacted() {
        let source = (0u8..5).flat_map(|value| [value; 32]).collect::<Vec<_>>();
        let compact = compact_splat_preview(&source, 3).unwrap();

        assert_eq!(compact.len(), 96);
        assert_eq!(compact[0], 0);
        assert_eq!(compact[32], 2);
        assert_eq!(compact[64], 4);
    }

    #[test]
    fn mesh_preview_keeps_both_triangles_of_a_depth_cell() {
        let path = std::env::temp_dir().join(format!("scanlan-mesh-{}.obj", uuid::Uuid::new_v4()));
        fs::write(
            &path,
            concat!(
                "v 0 0 0\n",
                "v 0 1 0\n",
                "v 1 0 0\n",
                "v 1 1 0\n",
                "vt 0 0\n",
                "vt 0 1\n",
                "vt 1 0\n",
                "vt 1 1\n",
                "f 1/1 2/2 3/3\n",
                "f 2/2 4/4 3/3\n"
            ),
        )
        .unwrap();

        // The old preview sampler treated this metadata count as a reason to
        // discard every other face, leaving visibly disconnected triangles.
        let packed = pack_preview_mesh(&path, 1_000_001).unwrap();
        fs::remove_file(path).ok();

        assert_eq!(u32::from_le_bytes(packed[8..12].try_into().unwrap()), 6);
    }

    #[test]
    fn project_normalization_accepts_one_mm_point_spacing() {
        let mut project = ProjectSummary::placeholder();
        project.settings.voxel_size_mm = 1;
        assert!(!normalize_project(&mut project));
        assert_eq!(project.settings.voxel_size_mm, 1);

        project.settings.voxel_size_mm = 0;
        assert!(normalize_project(&mut project));
        assert_eq!(project.settings.voxel_size_mm, 1);
    }

    #[test]
    fn unity_point_cloud_export_flips_only_z_and_preserves_rgb() {
        let header = concat!(
            "ply\n",
            "format binary_little_endian 1.0\n",
            "element vertex 2\n",
            "property float x\n",
            "property float y\n",
            "property float z\n",
            "property uchar red\n",
            "property uchar green\n",
            "property uchar blue\n",
            "end_header\n"
        );
        let mut source = header.as_bytes().to_vec();
        for (position, color) in [
            ([1.25_f32, 2.5, -3.75], [10_u8, 20, 30]),
            ([-4.5_f32, 5.75, 6.0], [40_u8, 50, 60]),
        ] {
            for value in position {
                source.extend_from_slice(&value.to_le_bytes());
            }
            source.extend_from_slice(&color);
        }

        let exported = unity_compatible_ply(source).unwrap();
        let marker = b"end_header\n";
        let payload_start = exported
            .windows(marker.len())
            .position(|window| window == marker)
            .unwrap()
            + marker.len();
        let vertices: Vec<_> = exported[payload_start..]
            .chunks_exact(15)
            .map(|vertex| {
                (
                    [
                        f32::from_le_bytes(vertex[0..4].try_into().unwrap()),
                        f32::from_le_bytes(vertex[4..8].try_into().unwrap()),
                        f32::from_le_bytes(vertex[8..12].try_into().unwrap()),
                    ],
                    [vertex[12], vertex[13], vertex[14]],
                )
            })
            .collect();

        assert!(String::from_utf8_lossy(&exported[..payload_start])
            .contains("comment Unity-ready coordinates: Z axis flipped"));
        assert_eq!(vertices[0], ([1.25, 2.5, 3.75], [10, 20, 30]));
        assert_eq!(vertices[1], ([-4.5, 5.75, -6.0], [40, 50, 60]));
    }

    #[test]
    fn unity_mesh_export_bakes_the_required_y_half_turn() {
        let exported = unity_compatible_obj(
            "v 1.25 2.5 -3.75\nvn 0.25 0.5 -0.75\nvt 0.2 0.8\nf 1/1/1 2/2/2 3/3/3\n",
        )
        .unwrap();
        let values = |prefix: &str| {
            exported
                .lines()
                .find_map(|line| line.strip_prefix(prefix))
                .unwrap()
                .split_whitespace()
                .map(|value| value.parse::<f32>().unwrap())
                .collect::<Vec<_>>()
        };
        for (actual, expected) in values("v ").iter().zip([-1.25, 2.5, 3.75]) {
            assert!((*actual - expected).abs() < 1e-5);
        }
        let normal_length = (0.25_f32 * 0.25 + 0.5 * 0.5 + 0.75 * 0.75).sqrt();
        for (actual, expected) in values("vn ").iter().zip([
            -0.25 / normal_length,
            0.5 / normal_length,
            0.75 / normal_length,
        ]) {
            assert!((*actual - expected).abs() < 1e-5);
        }
        assert!(exported.contains("vt 0.2 0.8"));
        assert!(exported.contains("f 1/1/1 2/2/2 3/3/3"));
    }

    #[test]
    fn unity_gaussian_export_bakes_the_z_reflection() {
        let names = [
            "x", "y", "z", "nx", "ny", "nz", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1",
            "rot_2", "rot_3",
        ];
        let header = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex 1\n{}end_header\n",
            names
                .iter()
                .map(|name| format!("property float {name}\n"))
                .collect::<String>()
        );
        let mut source = header.as_bytes().to_vec();
        for value in [
            1.25_f32,
            2.5,
            -3.75,
            0.0,
            0.0,
            1.0,
            0.3_f32.ln(),
            0.2_f32.ln(),
            0.01_f32.ln(),
            1.0,
            0.0,
            0.0,
            0.0,
        ] {
            source.extend_from_slice(&value.to_le_bytes());
        }

        let exported = unity_compatible_gaussian_ply(source).unwrap();
        let values = exported[header.len()..]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(&values[0..3], &[1.25, 2.5, 3.75]);
        assert_eq!(&values[3..6], &[0.0, 0.0, -1.0]);
    }

    #[test]
    fn point_cloud_bounds_are_applied_after_the_edit_pose() {
        let header = concat!(
            "ply\n",
            "format binary_little_endian 1.0\n",
            "element vertex 2\n",
            "property float x\n",
            "property float y\n",
            "property float z\n",
            "property uchar red\n",
            "property uchar green\n",
            "property uchar blue\n",
            "end_header\n"
        );
        let mut source = header.as_bytes().to_vec();
        for position in [[0.0_f32, 0.0, 0.0], [2.0_f32, 0.0, 0.0]] {
            position
                .into_iter()
                .for_each(|value| source.extend_from_slice(&value.to_le_bytes()));
            source.extend_from_slice(&[10, 20, 30]);
        }
        let transform = CloudTransform {
            position: [10.0, 0.0, 0.0],
            rotation: [0.0; 3],
            scale: [1.0; 3],
        };
        let bounds = BoundingBoxClip {
            min: [11.5, -1.0, -1.0],
            max: [12.5, 1.0, 1.0],
        };

        let transformed = transformed_cloud_ply(source, &transform).unwrap();
        let clipped = clipped_binary_ply(transformed, &bounds, "point cloud").unwrap();
        let marker = b"end_header\n";
        let payload_start = clipped
            .windows(marker.len())
            .position(|window| window == marker)
            .unwrap()
            + marker.len();
        assert!(String::from_utf8_lossy(&clipped[..payload_start]).contains("element vertex 1"));
        assert_eq!(clipped.len() - payload_start, 15);
        assert_eq!(
            f32::from_le_bytes(
                clipped[payload_start..payload_start + 4]
                    .try_into()
                    .unwrap()
            ),
            12.0
        );
    }

    #[test]
    fn mesh_clip_cuts_crossing_triangles_and_interpolates_attributes() {
        let source = concat!(
            "mtllib room-mesh.mtl\n",
            "v -1 0 0\n",
            "v 1 0 0\n",
            "v 0 1 0\n",
            "vn 0 0 1\n",
            "vn 0 0 1\n",
            "vn 0 0 1\n",
            "vt 0 0\n",
            "vt 1 0\n",
            "vt 0.5 1\n",
            "f 1/1/1 2/2/2 3/3/3\n"
        );
        let bounds = BoundingBoxClip {
            min: [-0.5, -1.0, -1.0],
            max: [1.0, 2.0, 1.0],
        };

        let clipped = clipped_obj(source, &bounds).unwrap();
        let positions = clipped
            .lines()
            .filter_map(|line| line.strip_prefix("v "))
            .map(|line| {
                line.split_whitespace()
                    .map(|value| value.parse::<f32>().unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(positions.iter().all(|position| position[0] >= -0.500_001));
        assert!(positions
            .iter()
            .any(|position| (position[0] + 0.5).abs() < 1e-6));
        assert_eq!(
            clipped
                .lines()
                .filter(|line| line.starts_with("f "))
                .count(),
            2
        );
        assert!(
            clipped
                .lines()
                .filter(|line| line.starts_with("vt "))
                .count()
                >= 4
        );
    }

    #[test]
    fn edit_pose_transforms_point_and_mesh_exports() {
        let transform = CloudTransform {
            position: [1.0, 2.0, 3.0],
            rotation: [0.0, 0.0, 90.0],
            scale: [2.0, 1.0, 1.0],
        };
        let header = concat!(
            "ply\n",
            "format binary_little_endian 1.0\n",
            "element vertex 1\n",
            "property float x\n",
            "property float y\n",
            "property float z\n",
            "property uchar red\n",
            "property uchar green\n",
            "property uchar blue\n",
            "end_header\n"
        );
        let mut source = header.as_bytes().to_vec();
        for value in [1.0_f32, 0.0, 0.0] {
            source.extend_from_slice(&value.to_le_bytes());
        }
        source.extend_from_slice(&[10, 20, 30]);

        let transformed = transformed_cloud_ply(source, &transform).unwrap();
        let vertex = &transformed[header.len()..];
        let position = [
            f32::from_le_bytes(vertex[0..4].try_into().unwrap()),
            f32::from_le_bytes(vertex[4..8].try_into().unwrap()),
            f32::from_le_bytes(vertex[8..12].try_into().unwrap()),
        ];
        assert!((position[0] - 1.0).abs() < 1e-5);
        assert!((position[1] - 4.0).abs() < 1e-5);
        assert!((position[2] - 3.0).abs() < 1e-5);
        assert_eq!(&vertex[12..15], &[10, 20, 30]);

        let obj = transformed_obj("v 1 0 0\nvn 1 0 0\nf 1 2 3\n", &transform).unwrap();
        let values = |prefix: &str| {
            obj.lines()
                .find_map(|line| line.strip_prefix(prefix))
                .unwrap()
                .split_whitespace()
                .map(|value| value.parse::<f32>().unwrap())
                .collect::<Vec<_>>()
        };
        let vertex = values("v ");
        let normal = values("vn ");
        for (actual, expected) in vertex.iter().zip([1.0, 4.0, 3.0]) {
            assert!((*actual - expected).abs() < 1e-5);
        }
        for (actual, expected) in normal.iter().zip([0.0, 1.0, 0.0]) {
            assert!((*actual - expected).abs() < 1e-5);
        }

        let mirrored = CloudTransform {
            position: [0.0; 3],
            rotation: [0.0; 3],
            scale: [-1.0, 1.0, 1.0],
        };
        assert!(transformed_obj("f 1 2 3\n", &mirrored)
            .unwrap()
            .contains("f 1 3 2"));
    }

    #[test]
    fn edit_pose_transforms_gaussian_means_normals_and_covariances() {
        let names = [
            "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "scale_0", "scale_1", "scale_2", "rot_0",
            "rot_1", "rot_2", "rot_3",
        ];
        let header = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex 1\n{}end_header\n",
            names
                .iter()
                .map(|name| format!("property float {name}\n"))
                .collect::<String>()
        );
        let half_angle = 15.0_f32.to_radians();
        let source_values = [
            1.0_f32,
            -2.0,
            0.5,
            1.0,
            0.0,
            0.0,
            7.25,
            0.1_f32.ln(),
            0.2_f32.ln(),
            0.3_f32.ln(),
            half_angle.cos(),
            0.0,
            half_angle.sin(),
            0.0,
        ];
        let mut source = header.as_bytes().to_vec();
        for value in source_values {
            source.extend_from_slice(&value.to_le_bytes());
        }
        let transform = CloudTransform {
            position: [1.0, 2.0, 3.0],
            rotation: [0.0, 0.0, 90.0],
            scale: [2.0, 1.0, -0.5],
        };

        let output = transformed_gaussian_ply(source, &transform).unwrap();
        let values = output[header.len()..]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let expected_position = transformed_position([1.0, -2.0, 0.5], &transform);
        let expected_normal = transformed_normal([1.0, 0.0, 0.0], &transform);
        for (actual, expected) in values[0..3].iter().zip(expected_position) {
            assert!((*actual - expected).abs() < 1e-5);
        }
        for (actual, expected) in values[3..6].iter().zip(expected_normal) {
            assert!((*actual - expected).abs() < 1e-5);
        }
        assert_eq!(values[6], 7.25);

        let original_rotation = quaternion_matrix([
            source_values[10] as f64,
            source_values[11] as f64,
            source_values[12] as f64,
            source_values[13] as f64,
        ]);
        let mut original_basis = original_rotation;
        for axis in 0..3 {
            for row in &mut original_basis {
                row[axis] *= [0.1, 0.2, 0.3][axis];
            }
        }
        let expected_basis = matrix_product(gaussian_edit_matrix(&transform), original_basis);
        let expected_covariance: [[f64; 3]; 3] = std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                (0..3)
                    .map(|axis| expected_basis[row][axis] * expected_basis[column][axis])
                    .sum()
            })
        });

        let actual_rotation = quaternion_matrix([
            values[10] as f64,
            values[11] as f64,
            values[12] as f64,
            values[13] as f64,
        ]);
        let mut actual_basis = actual_rotation;
        for axis in 0..3 {
            let scale = (values[7 + axis] as f64).exp();
            for row in &mut actual_basis {
                row[axis] *= scale;
            }
        }
        let actual_covariance: [[f64; 3]; 3] = std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                (0..3)
                    .map(|axis| actual_basis[row][axis] * actual_basis[column][axis])
                    .sum()
            })
        });
        for row in 0..3 {
            for column in 0..3 {
                assert!(
                    (actual_covariance[row][column] - expected_covariance[row][column]).abs()
                        < 1e-6,
                    "covariance[{row}][{column}] differs: {} != {}",
                    actual_covariance[row][column],
                    expected_covariance[row][column]
                );
            }
        }
    }

    #[test]
    fn modern_sensor_accepts_wide_binned_depth() {
        let mut settings = CaptureSettings {
            sensor_kind: "femto_mega".to_string(),
            depth_field_of_view: "wide".to_string(),
            depth_binned: true,
            ..CaptureSettings::default()
        };
        assert!(validate_sensor_settings(&mut settings).is_ok());
    }

    #[test]
    fn modern_sensor_configuration_is_forwarded_to_the_worker() {
        let settings = CaptureSettings {
            sensor_kind: "femto_mega".to_string(),
            rgb_resolution: "1080p".to_string(),
            rgb_auto_exposure: false,
            rgb_exposure_us: 8_300,
            rgb_gain: 30,
            imu_accel_rate_hz: 200,
            imu_accel_range_g: 4,
            imu_gyro_rate_hz: 500,
            imu_gyro_range_dps: 500,
            ..CaptureSettings::default()
        };
        let mut command = Command::new("sensor-worker");
        append_sensor_args(&mut command, &settings);
        let arguments = command
            .get_args()
            .map(|value| value.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        for expected in [
            "--sensor-fps",
            "0",
            "--rgb-resolution",
            "1080p",
            "--rgb-exposure-us",
            "8300",
            "--rgb-gain",
            "30",
            "--imu-accel-rate",
            "200",
            "--imu-accel-range",
            "4",
            "--imu-gyro-rate",
            "500",
            "--imu-gyro-range",
        ] {
            assert!(arguments.iter().any(|argument| argument == expected));
        }
    }

    #[test]
    fn legacy_capture_settings_receive_camera_control_defaults() {
        let mut value = serde_json::to_value(CaptureSettings::default()).unwrap();
        let object = value.as_object_mut().unwrap();
        for field in [
            "sensorFps",
            "rgbResolution",
            "rgbAutoExposure",
            "rgbExposureUs",
            "rgbGain",
            "rgbAutoWhiteBalance",
            "rgbWhiteBalanceK",
            "rgbColorAdjustmentsEnabled",
            "rgbBrightness",
            "rgbContrast",
            "rgbSaturation",
            "rgbSharpness",
            "rgbBacklightCompensation",
            "rgbPowerlineHz",
            "imuAccelRateHz",
            "imuAccelRangeG",
            "imuGyroRateHz",
            "imuGyroRangeDps",
            "liveMapMemoryMib",
            "lingbotDepthRefinement",
            "depthRefinementBackend",
            "experimentalRgbPreview",
        ] {
            object.remove(field);
        }
        let settings: CaptureSettings = serde_json::from_value(value).unwrap();
        assert_eq!(settings.rgb_resolution, "auto");
        assert!(settings.rgb_auto_exposure);
        assert!(settings.rgb_auto_white_balance);
        assert_eq!(settings.rgb_exposure_us, 8_330);
        assert_eq!(settings.imu_accel_rate_hz, 0);
        assert_eq!(settings.live_map_memory_mib, 1024);
        assert!(!settings.lingbot_depth_refinement);
        assert_eq!(settings.depth_refinement_backend, "off");
        assert!(!settings.experimental_rgb_preview);
    }

    #[test]
    fn capture_settings_reject_unknown_live_reconstruction_modes() {
        let mut settings = CaptureSettings {
            live_reconstruction: "teleport".to_string(),
            ..CaptureSettings::default()
        };
        assert_eq!(
            validate_sensor_settings(&mut settings),
            Err("Unknown live reconstruction mode".to_string())
        );
    }

    #[test]
    fn active_artifact_job_keeps_gaussian_preview_live_between_workers() {
        let mut project = ProjectSummary::placeholder();
        project.processing_status = "complete".to_string();
        project.active_job = Some("splat-job".to_string());

        assert!(gaussian_splat_preview_is_live(&project));
    }
}
