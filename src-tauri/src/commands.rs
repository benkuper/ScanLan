use crate::models::{
    AvailableSensor, CameraFrame, CaptureSettings, CaptureStatus, CloudTransform,
    LiveReconstructionStatus, LiveWorkerStatus, MediaSource, PreviewPoint, ProjectSummary,
    ReconstructionProgress, RuntimeInfo,
};
use crate::storage;
use chrono::Utc;
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
    live_reconstruction: Option<Child>,
    project_root: PathBuf,
    phase_root: PathBuf,
    phase_id: String,
}

pub struct ActiveLivePreview {
    child: Child,
    root: PathBuf,
    sensor_key: String,
    latest: Arc<Mutex<Option<LivePreviewFrame>>>,
}

#[derive(Clone)]
struct LivePreviewFrame {
    updated: Instant,
    frame_count: u32,
    stream_fps: f32,
    point_count: usize,
    packet: Arc<Vec<u8>>,
}

#[derive(Clone)]
pub struct AppState {
    pub project: Arc<Mutex<ProjectSummary>>,
    pub active_capture: Arc<Mutex<Option<ActiveCapture>>>,
    pub active_live_preview: Arc<Mutex<Option<ActiveLivePreview>>>,
    pub jobs: crate::jobs::JobManager,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            project: Arc::new(Mutex::new(ProjectSummary::placeholder())),
            active_capture: Arc::new(Mutex::new(None)),
            active_live_preview: Arc::new(Mutex::new(None)),
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
            stop_live_reconstruction(&mut capture, Duration::from_secs(2));
        }
    }
    terminate_live_preview(state);
}

fn stop_live_reconstruction(capture: &mut ActiveCapture, timeout: Duration) {
    let Some(mut child) = capture.live_reconstruction.take() else {
        return;
    };
    File::create(capture.phase_root.join("live-reconstruction.stop")).ok();
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
    let expected = indexed_frame_count(&capture.phase_root);
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if live_reconstruction_status(&capture.phase_root)
            .is_some_and(|status| status.processed_frames >= expected)
        {
            break;
        }
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

fn terminate_live_preview(state: &AppState) {
    if let Ok(mut active) = state.active_live_preview.lock() {
        if let Some(mut preview) = active.take() {
            File::create(preview.root.join("stop.flag")).ok();
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                match preview.child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) if Instant::now() < deadline => {
                        thread::sleep(Duration::from_millis(40))
                    }
                    _ => {
                        preview.child.kill().ok();
                        preview.child.wait().ok();
                        break;
                    }
                }
            }
        }
    }
}

fn project_base(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_data_dir()
        .map(|path| path.join("projects"))
        .map_err(|error| error.to_string())
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
    let safe_voxel_size = project.settings.voxel_size_mm.clamp(1, 40);
    if safe_voxel_size != project.settings.voxel_size_mm {
        project.settings.voxel_size_mm = safe_voxel_size;
        changed = true;
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

pub(crate) fn configure_media_tools(command: &mut Command, resource_root: Option<&Path>) {
    let mut paths = storage::media_tool_directories(resource_root);
    if let Some(current_path) = std::env::var_os("PATH") {
        paths.extend(std::env::split_paths(&current_path));
    }
    if let Ok(path) = std::env::join_paths(paths) {
        command.env("PATH", path);
    }
}

fn sensor_name(settings: &CaptureSettings) -> &'static str {
    match settings.sensor_kind.as_str() {
        "azure_kinect" => "Azure Kinect DK",
        "femto_mega" => "Orbbec Femto Mega",
        _ => "Kinect v2",
    }
}

fn sensor_key(settings: &CaptureSettings) -> String {
    format!(
        "{}|{}|{}|{}|{}|{}|{}",
        settings.sensor_kind,
        settings.sensor_id,
        settings.sensor_connection,
        settings.sensor_address.trim(),
        settings.use_imu,
        settings.depth_field_of_view,
        settings.depth_binned
    )
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
    if !matches!(settings.live_reconstruction.as_str(), "off" | "points" | "mesh") {
        return Err("Unknown live reconstruction mode".to_string());
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
    first_existing(candidates).ok_or_else(|| {
        format!(
            "{} capture support is missing from this app build",
            sensor_name(settings)
        )
    })
}

fn start_live_reconstruction(
    app: &AppHandle,
    phase_root: &Path,
    settings: &CaptureSettings,
) -> Result<Option<Child>, String> {
    if settings.live_reconstruction == "off" {
        return Ok(None);
    }
    let worker = first_existing(storage::candidate_reconstruction_worker_paths(
        resource_root(app).as_deref(),
    ))
    .ok_or_else(|| {
        "Live reconstruction support is missing from this app build; choose Sensor frames or install the reconstruction runtime"
            .to_string()
    })?;
    for name in [
        "live-reconstruction.stop",
        "live-reconstruction.json",
        "live-reconstruction.points",
        "live-reconstruction.mesh",
        "live-frame-selection.csv",
    ] {
        fs::remove_file(phase_root.join(name)).ok();
    }
    let stderr = File::create(phase_root.join("live-reconstruction.log"))
        .map_err(|error| error.to_string())?;
    let live_voxel_size_m = (settings.voxel_size_mm.max(10) as f32) / 1000.0;
    let mut command = worker_command(&worker);
    command
        .arg("live")
        .arg(phase_root)
        .arg("--mode")
        .arg(&settings.live_reconstruction)
        .arg("--voxel-size")
        .arg(live_voxel_size_m.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr));
    command
        .spawn()
        .map(Some)
        .map_err(|error| format!("Could not start live reconstruction: {error}"))
}

fn append_sensor_args(command: &mut Command, settings: &CaptureSettings) {
    command
        .arg("--rgb-quality")
        .arg(settings.rgb_jpeg_quality.to_string());
    if settings.max_rgb_dimension > 0 {
        command
            .arg("--max-rgb-dimension")
            .arg(settings.max_rgb_dimension.to_string());
    }
    if settings.sensor_kind == "kinect_v2" {
        return;
    }
    command
        .arg("--sensor")
        .arg(&settings.sensor_kind)
        .arg("--connection")
        .arg(&settings.sensor_connection)
        .arg("--depth-fov")
        .arg(&settings.depth_field_of_view);
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
        command.arg("--imu");
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

fn read_child_stderr(child: &mut Child) -> String {
    let mut message = String::new();
    if let Some(mut stderr) = child.stderr.take() {
        stderr.read_to_string(&mut message).ok();
    }
    message.trim().to_string()
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

fn load_phase_preview(phase_root: &Path) -> Result<Vec<PreviewPoint>, String> {
    let manifest: crate::models::PhaseManifest = serde_json::from_reader(
        File::open(phase_root.join("phase.json")).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if manifest.camera.fx <= 0.0 || manifest.camera.fy <= 0.0 {
        return Err("The sensor frame has invalid camera calibration".to_string());
    }

    let index =
        fs::read_to_string(phase_root.join("frames.csv")).map_err(|error| error.to_string())?;
    let latest = index
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .last()
        .ok_or_else(|| "The capture has not produced a frame yet".to_string())?;
    let columns: Vec<&str> = latest.split(',').collect();
    if columns.len() < 4 {
        return Err("The sensor frame index is invalid".to_string());
    }
    let depth = fs::read(phase_root.join(columns[2])).map_err(|error| error.to_string())?;
    let color = fs::read(phase_root.join(columns[3])).map_err(|error| error.to_string())?;
    let width = manifest.camera.width as usize;
    let height = manifest.camera.height as usize;
    let pixel_count = width * height;
    if depth.len() != pixel_count * 2 || color.len() != pixel_count * 3 {
        return Err("The latest sensor frame is incomplete".to_string());
    }

    let mut preview = Vec::with_capacity(pixel_count / 16);
    let flip_x = manifest
        .sensor
        .as_ref()
        .map(|sensor| sensor.kind == "kinect_v2")
        .unwrap_or(true);
    for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            let pixel = y * width + x;
            let depth_mm = u16::from_le_bytes([depth[pixel * 2], depth[pixel * 2 + 1]]);
            if depth_mm == 0 {
                continue;
            }
            let z = depth_mm as f64 / manifest.camera.depth_scale;
            if z > manifest.camera.max_depth_m {
                continue;
            }
            let point_x = (x as f64 - manifest.camera.cx) * z / manifest.camera.fx;
            let point_y = (y as f64 - manifest.camera.cy) * z / manifest.camera.fy;
            let color_offset = pixel * 3;
            preview.push(PreviewPoint {
                // Capture-time Fusion poses are useful diagnostics, but are not stable
                // enough to drive the operator preview. Keep this view camera-relative;
                // the validated offline registration owns final frame placement.
                position: [
                    if flip_x {
                        -point_x as f32
                    } else {
                        point_x as f32
                    },
                    -point_y as f32,
                    -z as f32,
                ],
                color: [
                    color[color_offset],
                    color[color_offset + 1],
                    color[color_offset + 2],
                ],
            });
        }
    }
    Ok(preview)
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

fn live_reconstruction_status(root: &Path) -> Option<LiveReconstructionStatus> {
    let path = root.join("live-reconstruction.json");
    let modified = fs::metadata(&path).ok()?.modified().ok()?;
    if modified.elapsed().ok()? > Duration::from_secs(3) {
        return None;
    }
    serde_json::from_reader(File::open(path).ok()?).ok()
}

fn live_reconstruction_packet(root: &Path, name: &str, magic: &[u8; 4], after_frame: u32) -> Vec<u8> {
    let bytes = match fs::read(root.join(name)) {
        Ok(bytes) => bytes,
        Err(_) => return Vec::new(),
    };
    if bytes.len() < 8 || &bytes[0..4] != magic {
        return Vec::new();
    }
    let frame_count = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if frame_count == after_frame {
        Vec::new()
    } else {
        bytes
    }
}

fn read_live_preview_stream(
    stdout: std::process::ChildStdout,
    latest: Arc<Mutex<Option<LivePreviewFrame>>>,
) {
    let mut reader = BufReader::new(stdout);
    loop {
        let mut header = [0_u8; 24];
        if reader.read_exact(&mut header).is_err() {
            break;
        }
        if &header[0..4] != b"K2P1" {
            break;
        }
        let frame_count = u32::from_le_bytes(header[4..8].try_into().unwrap());
        let stream_fps = f32::from_le_bytes(header[16..20].try_into().unwrap());
        let count = u32::from_le_bytes(header[20..24].try_into().unwrap()) as usize;
        if count > 100_000 {
            break;
        }
        let mut bytes = vec![0_u8; count * 15];
        if reader.read_exact(&mut bytes).is_err() {
            break;
        }
        // Keep the worker's compact binary representation intact. Converting
        // every live frame into nested PreviewPoint objects here made the later
        // JSON IPC response several times larger and limited the viewer to only
        // a few updates per second even while the sensor held 30 fps.
        let mut packet = Vec::with_capacity(header.len() + bytes.len());
        packet.extend_from_slice(&header);
        packet.extend_from_slice(&bytes);
        if let Ok(mut slot) = latest.lock() {
            *slot = Some(LivePreviewFrame {
                updated: Instant::now(),
                frame_count,
                stream_fps,
                point_count: count,
                packet: Arc::new(packet),
            });
        }
    }
}

fn live_preview_snapshot(state: &AppState) -> Option<LivePreviewFrame> {
    let active = state.active_live_preview.lock().ok()?;
    let frame = active.as_ref()?.latest.lock().ok()?.clone()?;
    (frame.updated.elapsed() <= Duration::from_secs(2)).then_some(frame)
}

#[tauri::command]
pub fn live_preview_frame(
    after_frame: u32,
    state: State<'_, AppState>,
) -> Result<tauri::ipc::Response, String> {
    let capture_root = state
        .active_capture
        .lock()
        .ok()
        .and_then(|active| active.as_ref().map(|capture| capture.phase_root.clone()));
    if let Some(root) = capture_root {
        let body = live_reconstruction_packet(
            &root,
            "live-reconstruction.points",
            b"K2P1",
            after_frame,
        );
        return Ok(tauri::ipc::Response::new(body));
    }
    let frame = live_preview_snapshot(state.inner());
    let body = match frame {
        Some(frame) if frame.frame_count != after_frame => frame.packet.as_ref().clone(),
        _ => Vec::new(),
    };
    Ok(tauri::ipc::Response::new(body))
}

#[tauri::command]
pub fn live_reconstruction_mesh(
    after_frame: u32,
    state: State<'_, AppState>,
) -> Result<tauri::ipc::Response, String> {
    let capture_root = state
        .active_capture
        .lock()
        .ok()
        .and_then(|active| active.as_ref().map(|capture| capture.phase_root.clone()));
    let body = capture_root
        .map(|root| {
            live_reconstruction_packet(
                &root,
                "live-reconstruction.mesh",
                b"K2M2",
                after_frame,
            )
        })
        .unwrap_or_default();
    Ok(tauri::ipc::Response::new(body))
}

fn ensure_live_preview(
    app: &AppHandle,
    state: &AppState,
    settings: &CaptureSettings,
) -> Result<PathBuf, String> {
    let desired_key = sensor_key(settings);
    let mut active = state
        .active_live_preview
        .lock()
        .map_err(|_| "Sensor preview state is unavailable".to_string())?;

    if let Some(preview) = active.as_mut() {
        if preview.sensor_key == desired_key {
            match preview
                .child
                .try_wait()
                .map_err(|error| error.to_string())?
            {
                None => return Ok(preview.root.clone()),
                Some(status) => {
                    let detail = read_child_stderr(&mut preview.child);
                    *active = None;
                    if !status.success() && !detail.is_empty() {
                        return Err(detail);
                    }
                }
            }
        } else if let Some(mut previous) = active.take() {
            File::create(previous.root.join("stop.flag")).ok();
            previous.child.kill().ok();
            previous.child.wait().ok();
        }
    }

    let worker = sensor_worker(app, settings)?;
    let root = app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("live-preview");
    fs::create_dir_all(&root).map_err(|error| error.to_string())?;
    for relative in [
        "stop.flag",
        "live.json",
        "latest.points",
        "frames.csv",
        "phase.json",
    ] {
        fs::remove_file(root.join(relative)).ok();
    }
    let mut command = worker_command(&worker);
    command
        .arg("--preview")
        .arg(&root)
        .arg("--fps")
        .arg("30")
        .arg("--max-depth")
        .arg(settings.max_depth_m.to_string());
    append_sensor_args(&mut command, settings);
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            format!(
                "Could not start live {} preview: {error}",
                sensor_name(settings)
            )
        })?;
    let latest = Arc::new(Mutex::new(None));
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Could not open the live sensor preview stream".to_string())?;
    let reader_state = Arc::clone(&latest);
    thread::spawn(move || read_live_preview_stream(stdout, reader_state));
    *active = Some(ActiveLivePreview {
        child,
        root: root.clone(),
        sensor_key: desired_key,
        latest,
    });
    Ok(root)
}

fn load_project_preview(project_root: &Path) -> Result<Vec<PreviewPoint>, String> {
    let project = storage::read_project(project_root)?;
    if project.processing_status == "complete" {
        let output = project_root.join("outputs").join("preview.json");
        if output.exists() {
            return serde_json::from_reader(File::open(output).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string());
        }
    }

    for phase in project.phases.iter().rev() {
        let phase_root = project_root.join("phases").join(&phase.id);
        if indexed_frame_count(&phase_root) > 0 {
            if let Ok(preview) = load_phase_preview(&phase_root) {
                return Ok(preview);
            }
        }
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
    let preview_active = state
        .active_live_preview
        .lock()
        .map(|preview| preview.is_some())
        .unwrap_or(false);
    tauri::async_runtime::spawn_blocking(move || {
        let mut sensors = Vec::new();

        if preview_active {
            sensors.push(AvailableSensor {
                id: if saved.sensor_id.is_empty() {
                    format!("{}:default", saved.sensor_kind)
                } else {
                    saved.sensor_id.clone()
                },
                kind: saved.sensor_kind.clone(),
                name: sensor_name(&saved).to_string(),
                connection: saved.sensor_connection.clone(),
                address: saved.sensor_address.clone(),
                serial: String::new(),
                supports_imu: saved.sensor_kind != "kinect_v2",
            });
        }

        // The Kinect v2 SDK has no passive enumeration API: its old --probe
        // path opens the camera, turns on the light, and starts its streams.
        // Advertise installed capture support without touching the hardware;
        // the camera is opened only after the user explicitly selects it.
        if !sensors.iter().any(|sensor| sensor.kind == "kinect_v2")
            && first_existing(storage::candidate_kinect_worker_paths(resources.as_deref()))
                .is_some()
        {
            sensors.push(AvailableSensor {
                id: "kinect_v2:default".to_string(),
                kind: "kinect_v2".to_string(),
                name: "Kinect v2".to_string(),
                connection: "usb".to_string(),
                address: String::new(),
                serial: String::new(),
                supports_imu: false,
            });
        }

        if let Some(worker) = first_existing(storage::candidate_modern_sensor_worker_paths(
            resources.as_deref(),
        )) {
            if let Ok(output) = worker_command(&worker).arg("--list").output() {
                sensors.extend(parse_available_sensors(&output));
            }

            if saved.sensor_kind == "femto_mega"
                && saved.sensor_connection == "network"
                && !saved.sensor_address.is_empty()
                && !sensors.iter().any(|sensor| {
                    sensor.kind == "femto_mega"
                        && sensor.connection == "network"
                        && sensor.address == saved.sensor_address
                })
            {
                let mut command = worker_command(&worker);
                command.arg("--probe");
                append_sensor_args(&mut command, &saved);
                if command
                    .output()
                    .map(|output| output.status.success())
                    .unwrap_or(false)
                {
                    sensors.push(AvailableSensor {
                        id: if saved.sensor_id.is_empty() {
                            format!("femto_mega:network:{}", saved.sensor_address)
                        } else {
                            saved.sensor_id.clone()
                        },
                        kind: "femto_mega".to_string(),
                        name: "Orbbec Femto Mega".to_string(),
                        connection: "network".to_string(),
                        address: saved.sensor_address.clone(),
                        serial: String::new(),
                        supports_imu: true,
                    });
                }
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
        let sensor_worker = if settings.sensor_kind == "kinect_v2" {
            first_existing(storage::candidate_kinect_worker_paths(resources.as_deref()))
        } else {
            first_existing(storage::candidate_modern_sensor_worker_paths(
                resources.as_deref(),
            ))
        };
        let reconstruction_worker_available = first_existing(
            storage::candidate_reconstruction_worker_paths(resources.as_deref()),
        )
        .is_some();
        let splat_worker =
            first_existing(storage::candidate_splat_worker_paths(resources.as_deref()));
        let (splat_worker_available, splat_status) = match splat_worker {
            Some(worker) => {
                let mut command = worker_command(&worker);
                configure_media_tools(&mut command, resources.as_deref());
                command.arg("diagnostics");
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
        let ffmpeg = first_existing(storage::candidate_ffmpeg_paths(resources.as_deref()))
            .unwrap_or_else(|| PathBuf::from("ffmpeg"));
        let mut ffmpeg_command = worker_command(&ffmpeg);
        configure_media_tools(&mut ffmpeg_command, resources.as_deref());
        let ffmpeg_available = ffmpeg_command
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
        let colmap = first_existing(storage::candidate_colmap_paths(resources.as_deref()))
            .unwrap_or_else(|| PathBuf::from("colmap"));
        let mut colmap_command = worker_command(&colmap);
        configure_media_tools(&mut colmap_command, resources.as_deref());
        let colmap_available = colmap_command
            .arg("-h")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success());

        let (sensor_connected, sensor_status) = match &sensor_worker {
            Some(worker) => {
                let mut command = worker_command(worker);
                command.arg("--probe");
                append_sensor_args(&mut command, &settings);
                match command.output() {
                    Ok(output) if output.status.success() => (
                        true,
                        format!("{} connected and streaming", sensor_name(&settings)),
                    ),
                    Ok(output) => {
                        let detail = output_message(&output);
                        (
                            false,
                            if detail.is_empty() {
                                format!("{} could not be opened", sensor_name(&settings))
                            } else {
                                detail
                            },
                        )
                    }
                    Err(error) => (false, format!("Could not start sensor support: {error}")),
                }
            }
            None => (
                false,
                format!(
                    "{} capture support is missing from this app build",
                    sensor_name(&settings)
                ),
            ),
        };

        RuntimeInfo {
            platform: std::env::consts::OS.to_string(),
            sensor_worker_available: sensor_worker.is_some(),
            sensor_connected,
            sensor_status,
            reconstruction_worker_available,
            splat_worker_available,
            splat_status,
            ffmpeg_available,
            colmap_available,
        }
    })
    .await
    .unwrap_or_else(|error| RuntimeInfo {
        platform: std::env::consts::OS.to_string(),
        sensor_worker_available: false,
        sensor_connected: false,
        sensor_status: format!("Sensor connection check failed: {error}"),
        reconstruction_worker_available: false,
        splat_worker_available: false,
        splat_status: "Splat runtime detection failed".to_string(),
        ffmpeg_available: false,
        colmap_available: false,
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
pub fn create_project(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    let previous_settings = state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())?
        .settings
        .clone();
    let folder = format!("scan-{}", Utc::now().format("%Y%m%d-%H%M%S-%3f"));
    let mut project = storage::create_project(&project_base(&app)?.join(folder))?;
    project.settings = previous_settings;
    storage::write_project(&project)?;
    *state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
    Ok(project)
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
    if settings.environment != "indoor" && settings.environment != "outdoor_low_light" {
        return Err("Unknown capture environment".to_string());
    }
    validate_sensor_settings(&mut settings)?;
    let sensor_changed = sensor_key(&project.settings) != sensor_key(&settings);
    project.settings = settings;
    storage::write_project(&project)?;
    write_sensor_preference(&app, &project.settings)?;
    *state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
    if sensor_changed {
        // Close the previous device before a later, explicitly requested status
        // refresh can open the newly selected one.
        terminate_live_preview(state.inner());
    }
    Ok(project)
}

#[tauri::command]
pub fn start_sensor_phase(
    app: AppHandle,
    project_path: String,
    mut settings: CaptureSettings,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    validate_sensor_settings(&mut settings)?;
    terminate_live_preview(state.inner());
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
    project.confidence_score = None;
    project.confidence_label = None;
    project.confidence_detail = None;
    project.frames_used = None;
    let phase_id = Uuid::new_v4().to_string();
    let phase_name = format!(
        "{} phase {}",
        sensor_name(&project.settings),
        project.phases.len() + 1
    );
    let phase_root = project_root.join("phases").join(&phase_id);
    fs::create_dir_all(&phase_root).map_err(|error| error.to_string())?;

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
        .arg(project.settings.max_depth_m.to_string());
    append_sensor_args(&mut command, &project.settings);
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            format!(
                "Could not start {} capture: {error}",
                sensor_name(&project.settings)
            )
        })?;

    let manifest_path = phase_root.join("phase.json");
    let startup_deadline = Instant::now() + Duration::from_secs(12);
    while !manifest_path.exists() {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            let detail = read_child_stderr(&mut child);
            return Err(if detail.is_empty() {
                format!("Sensor capture stopped during startup ({status})")
            } else {
                detail
            });
        }
        if Instant::now() >= startup_deadline {
            child.kill().ok();
            child.wait().ok();
            let detail = read_child_stderr(&mut child);
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

    let live_reconstruction = match start_live_reconstruction(&app, &phase_root, &project.settings) {
        Ok(child) => child,
        Err(error) => {
            File::create(phase_root.join("stop.flag")).ok();
            child.kill().ok();
            child.wait().ok();
            return Err(error);
        }
    };

    let is_reference = project.phases.is_empty();
    project.phases.push(crate::models::PhaseSummary {
        id: phase_id.clone(),
        name: phase_name,
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
    *active = Some(ActiveCapture {
        child,
        live_reconstruction,
        project_root,
        phase_root,
        phase_id,
    });
    Ok(project)
}

#[tauri::command]
pub fn capture_status(app: AppHandle, state: State<'_, AppState>) -> Result<CaptureStatus, String> {
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
                    active_snapshot = Some((capture.phase_root.clone(), capture.phase_id.clone()));
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
        drain_live_reconstruction(&mut capture, Duration::from_secs(1));
        stop_live_reconstruction(&mut capture, Duration::from_secs(2));
        let detail = read_child_stderr(&mut capture.child);
        let frame_count = indexed_frame_count(&capture.phase_root);
        let completed = status.success() && frame_count > 0;
        if let Some(phase) = project
            .phases
            .iter_mut()
            .find(|phase| phase.id == capture.phase_id)
        {
            phase.frame_count = frame_count;
            phase.duration_seconds =
                (frame_count / project.settings.capture_fps.max(1)).max(u32::from(frame_count > 0));
            phase.status = if completed { "complete" } else { "failed" }.to_string();
            phase.overlap_hint = if completed {
                "Capture ended; ready for alignment".to_string()
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
            preview: Vec::new(),
            capturing: false,
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
            imu_active: false,
            imu_rate_hz: 0.0,
            live_reconstruction_active: false,
            live_reconstruction_mode: project.settings.live_reconstruction.clone(),
            live_processed_frame_count: 0,
            live_integrated_frame_count: 0,
            live_rejected_frame_count: 0,
            live_triangle_count: 0,
            live_reconstruction_backend: None,
            reconstruction,
            error: (!completed).then_some(if detail.is_empty() {
                "The sensor capture stopped before a usable phase was completed".to_string()
            } else {
                detail
            }),
        });
    }

    if let Some((phase_root, phase_id)) = active_snapshot {
        let frame_count = indexed_frame_count(&phase_root);
        if let Some(phase) = project.phases.iter_mut().find(|phase| phase.id == phase_id) {
            phase.frame_count = frame_count;
            phase.duration_seconds = frame_count / project.settings.capture_fps.max(1);
        }
        *state
            .project
            .lock()
            .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
        let live = live_worker_status(&phase_root);
        let live_reconstruction = live_reconstruction_status(&phase_root);
        let preview = if live_reconstruction.is_some() {
            Vec::new()
        } else {
            load_phase_preview(&phase_root).unwrap_or_default()
        };
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
            preview_point_count: live_reconstruction
                .as_ref()
                .map(|status| status.point_count)
                .unwrap_or(preview.len() as u64),
            preview,
            capturing: true,
            sensor_connected: live.is_some(),
            sensor_paused: false,
            sensor_status: live
                .as_ref()
                .map(|status| {
                    format!(
                        "{} streaming at {:.1} fps",
                        selected_sensor_name, status.stream_fps
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
                .unwrap_or(project.settings.live_reconstruction != "off"),
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
            live_reconstruction_backend: live_reconstruction
                .as_ref()
                .map(|status| status.backend.clone()),
            reconstruction,
            error: None,
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
            preview_point_count: preview.len() as u64,
            preview,
            capturing: false,
            sensor_connected: false,
            sensor_paused: true,
            sensor_status: "Sensor preview paused while reconstructing".to_string(),
            sensor_name: selected_sensor_name,
            frame_count: 0,
            total_frame_count,
            stream_fps: 0.0,
            tracking: false,
            tracking_status: "Reconstruction preview".to_string(),
            imu_active: false,
            imu_rate_hz: 0.0,
            live_reconstruction_active: false,
            live_reconstruction_mode: project.settings.live_reconstruction.clone(),
            live_processed_frame_count: 0,
            live_integrated_frame_count: 0,
            live_rejected_frame_count: 0,
            live_triangle_count: 0,
            live_reconstruction_backend: None,
            reconstruction: reconstruction_progress(&project_root),
            error: None,
        });
    }
    let live_result = ensure_live_preview(&app, state.inner(), &project.settings);
    let live_error = live_result.as_ref().err().cloned();
    let worker_status = live_result
        .as_ref()
        .ok()
        .and_then(|root| live_worker_status(root));
    let live = live_preview_snapshot(state.inner());
    // Live point data travels through live_preview_frame as a compact binary
    // response. Keep capture_status lightweight so status and point-cloud
    // refreshes cannot block one another on JSON serialization.
    let preview = if live.is_some() {
        Vec::new()
    } else {
        load_project_preview(&project_root).unwrap_or_default()
    };
    let preview_point_count = live
        .as_ref()
        .map(|frame| frame.point_count as u64)
        .unwrap_or(preview.len() as u64);
    let total_frame_count = project.phases.iter().map(|phase| phase.frame_count).sum();
    let frame_count = live.as_ref().map(|frame| frame.frame_count).unwrap_or(0);
    let selected_sensor_name = worker_status
        .as_ref()
        .map(|status| status.sensor_name.trim())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| sensor_name(&project.settings).to_string());
    Ok(CaptureStatus {
        project: project.clone(),
        preview_point_count,
        preview,
        capturing: false,
        sensor_connected: live.is_some(),
        sensor_paused: false,
        sensor_status: live
            .as_ref()
            .map(|frame| {
                format!(
                    "{} streaming at {:.1} fps",
                    selected_sensor_name, frame.stream_fps
                )
            })
            .or_else(|| live_error.clone())
            .unwrap_or_else(|| format!("Opening the {} stream", selected_sensor_name)),
        sensor_name: selected_sensor_name,
        frame_count,
        total_frame_count,
        stream_fps: live.as_ref().map(|frame| frame.stream_fps).unwrap_or(0.0),
        tracking: false,
        tracking_status: worker_status
            .as_ref()
            .map(|status| status.tracking_status.clone())
            .unwrap_or_else(|| "30 Hz live depth preview".to_string()),
        imu_active: worker_status
            .as_ref()
            .map(|status| status.imu_active)
            .unwrap_or(false),
        imu_rate_hz: worker_status
            .as_ref()
            .map(|status| status.imu_rate_hz)
            .unwrap_or(0.0),
        live_reconstruction_active: false,
        live_reconstruction_mode: project.settings.live_reconstruction.clone(),
        live_processed_frame_count: 0,
        live_integrated_frame_count: 0,
        live_rejected_frame_count: 0,
        live_triangle_count: 0,
        live_reconstruction_backend: None,
        reconstruction: reconstruction_progress(&project_root),
        error: live_error,
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

fn supported_photo(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "tif" | "tiff" | "webp"
            )
        })
}

#[tauri::command]
pub fn import_media_source(
    project_path: String,
    kind: String,
    paths: Vec<String>,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    if !matches!(kind.as_str(), "photos" | "video") {
        return Err("Media kind must be photos or video".to_string());
    }
    if paths.is_empty() {
        return Err("Choose photos or a video to import".to_string());
    }
    let root = PathBuf::from(project_path);
    let mut selected = Vec::new();
    for raw in paths {
        let path = PathBuf::from(raw);
        if path.is_dir() && kind == "photos" {
            for entry in fs::read_dir(&path).map_err(|error| error.to_string())? {
                let entry = entry.map_err(|error| error.to_string())?;
                if entry
                    .file_type()
                    .map_err(|error| error.to_string())?
                    .is_file()
                    && supported_photo(&entry.path())
                {
                    selected.push(entry.path());
                }
            }
        } else if path.is_file() {
            selected.push(path);
        }
    }
    if kind == "photos" {
        selected.retain(|path| supported_photo(path));
    } else if selected.len() != 1 {
        return Err("Choose one video at a time".to_string());
    }
    selected.sort();
    if selected.is_empty() {
        return Err("No supported media files were selected".to_string());
    }

    let id = Uuid::new_v4().to_string();
    let relative_root = PathBuf::from("sources").join(&id);
    let originals_root = root.join(&relative_root).join("originals");
    fs::create_dir_all(&originals_root).map_err(|error| error.to_string())?;
    let mut originals = Vec::new();
    for (index, source) in selected.iter().enumerate() {
        let original_name = source
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("media");
        let destination_name = format!("{index:05}-{original_name}");
        fs::copy(source, originals_root.join(&destination_name))
            .map_err(|error| format!("Could not copy {}: {error}", source.display()))?;
        originals.push(format!("originals/{destination_name}"));
    }
    let mut project = storage::read_project(&root)?;
    project.media_sources.push(MediaSource {
        id: id.clone(),
        kind: kind.clone(),
        name: if kind == "video" {
            selected[0]
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("Imported video")
                .to_string()
        } else {
            format!("Photo set {}", project.media_sources.len() + 1)
        },
        created_at: Utc::now().to_rfc3339(),
        path: relative_root.to_string_lossy().replace('\\', "/"),
        originals,
        status: "ready".to_string(),
        image_count: if kind == "photos" {
            selected.len() as u32
        } else {
            0
        },
        metric: false,
        quality: None,
    });
    project.artifacts.gaussian_splat = None;
    storage::write_project(&project)?;
    *state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
    Ok(project)
}

#[tauri::command]
pub fn remove_media_source(
    project_path: String,
    source_id: String,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    let root = PathBuf::from(project_path);
    let mut project = storage::read_project(&root)?;
    if project.active_job.is_some() {
        return Err("Cancel the active artifact job before removing media".to_string());
    }
    let index = project
        .media_sources
        .iter()
        .position(|source| source.id == source_id)
        .ok_or_else(|| "The media source no longer exists".to_string())?;
    let sources_root = root.join("sources");
    let source_root = sources_root.join(&source_id);
    if source_root.parent() != Some(sources_root.as_path()) {
        return Err("Refusing an invalid media-source path".to_string());
    }
    if source_root.exists() {
        fs::remove_dir_all(&source_root)
            .map_err(|error| format!("Could not remove media source: {error}"))?;
    }
    project.media_sources.remove(index);
    project.artifacts.gaussian_splat = None;
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
        File::create(capture.phase_root.join("stop.flag")).map_err(|error| error.to_string())?;
        let deadline = Instant::now() + Duration::from_secs(15);
        let status = loop {
            if let Some(status) = capture
                .child
                .try_wait()
                .map_err(|error| error.to_string())?
            {
                break status;
            }
            if Instant::now() >= deadline {
                capture.child.kill().map_err(|error| error.to_string())?;
                break capture.child.wait().map_err(|error| error.to_string())?;
            }
            thread::sleep(Duration::from_millis(80));
        };
        drain_live_reconstruction(&mut capture, Duration::from_secs(3));
        stop_live_reconstruction(&mut capture, Duration::from_secs(5));

        let manifest_path = capture.phase_root.join("phase.json");
        let capture_summary = if manifest_path.exists() {
            let manifest: crate::models::PhaseManifest = serde_json::from_reader(
                File::open(&manifest_path).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            (manifest.frame_count, manifest.duration_seconds)
        } else {
            (0, 0)
        };
        let mut project = storage::read_project(&capture.project_root)?;
        let frame_count = capture_summary
            .0
            .max(indexed_frame_count(&capture.phase_root));
        let duration_seconds = capture_summary.1.max(
            (frame_count / project.settings.capture_fps.max(1)).max(u32::from(frame_count > 0)),
        );
        let completed = status.success() && frame_count > 0;
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
            phase.overlap_hint = if !completed {
                "No usable sensor frames were saved".to_string()
            } else if phase_count == 1 {
                "Reference phase".to_string()
            } else {
                "Ready for offline alignment".to_string()
            };
        }
        fs::remove_file(capture.phase_root.join("stop.flag")).ok();
        fs::remove_file(capture.phase_root.join("live-reconstruction.stop")).ok();
        storage::write_project(&project)?;
        *project_state
            .lock()
            .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
        Ok(project)
    })
    .await
    .map_err(|error| error.to_string())?
}

fn run_worker(project_path: &Path, resource_root: Option<&Path>) -> Result<(), String> {
    let worker = first_existing(storage::candidate_reconstruction_worker_paths(
        resource_root,
    ))
    .ok_or_else(|| {
        "Point-cloud reconstruction support is missing from this app build".to_string()
    })?;
    let output = worker_command(&worker)
        .arg("reconstruct")
        .arg(project_path)
        .arg("--engine")
        .arg("auto")
        .output()
        .map_err(|error| format!("Could not start point-cloud reconstruction: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    let detail = output_message(&output);
    Err(if detail.is_empty() {
        format!("Point-cloud reconstruction failed ({})", output.status)
    } else {
        detail
    })
}

#[tauri::command]
pub async fn reconstruct_project(
    app: AppHandle,
    project_path: String,
    settings: CaptureSettings,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, String> {
    terminate_live_preview(state.inner());
    let root = PathBuf::from(project_path);
    let mut project = storage::read_project(&root)?;
    project.settings = settings;
    project.settings.voxel_size_mm = project.settings.voxel_size_mm.clamp(1, 40);
    project.processing_status = "processing".to_string();
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
    fs::remove_file(root.join("outputs").join("progress.json")).ok();
    fs::remove_file(root.join("outputs").join("build-preview.json")).ok();
    storage::write_project(&project)?;
    *state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())? = project.clone();

    // Publish the processing state before yielding to the worker thread. This
    // prevents a fast status poll from restarting the sensor preview in the
    // small gap between stopping the sensor and starting reconstruction.
    let project_state = Arc::clone(&state.project);
    let resources = resource_root(&app);
    tauri::async_runtime::spawn_blocking(move || {
        if let Err(error) = run_worker(&root, resources.as_deref()) {
            let mut failed = storage::read_project(&root).unwrap_or_else(|_| project.clone());
            failed.processing_status = "failed".to_string();
            failed.processing_error = Some(error.clone());
            failed.point_count = None;
            failed.output_path = None;
            failed.mesh_triangle_count = None;
            failed.mesh_output_path = None;
            failed.camera_frame_count = None;
            failed.confidence_score = None;
            failed.confidence_label = None;
            failed.confidence_detail = None;
            failed.frames_used = None;
            failed.processing_backend = None;
            failed.processing_duration_seconds = None;
            storage::write_project(&failed)?;
            *project_state
                .lock()
                .map_err(|_| "Project state is unavailable".to_string())? = failed;
            return Err(error);
        }
        let project = storage::read_project(&root)?;
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

#[tauri::command]
pub async fn load_camera_frames(project_path: String) -> Result<Vec<CameraFrame>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let path = PathBuf::from(project_path)
            .join("outputs")
            .join("camera-poses.json");
        if !path.is_file() {
            return Ok(Vec::new());
        }
        serde_json::from_reader(File::open(path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("Could not read reconstructed camera poses: {error}"))
    })
    .await
    .map_err(|error| error.to_string())?
}

const PLY_VERTEX_STRIDE: usize = 15;

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
        let x = f32::from_le_bytes(vertex[0..4].try_into().unwrap());
        vertex[0..4].copy_from_slice(&(-x).to_le_bytes());
    }

    let unity_comment = b"comment Unity-ready coordinates: X axis flipped\n";
    let mut output = Vec::with_capacity(bytes.len() + unity_comment.len());
    output.extend_from_slice(&bytes[..header_end]);
    output.extend_from_slice(unity_comment);
    output.extend_from_slice(marker);
    output.extend_from_slice(&bytes[payload_start..]);
    Ok(output)
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
pub fn export_ply(project_path: String, destination_path: String) -> Result<String, String> {
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
    let unity_bytes = unity_compatible_ply(bytes)?;
    write_export(&destination, &unity_bytes)?;
    Ok(destination.to_string_lossy().into_owned())
}

#[tauri::command]
pub fn export_textured_mesh(
    project_path: String,
    destination_path: String,
) -> Result<String, String> {
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

    let obj = fs::read_to_string(source_obj)
        .map_err(|error| format!("Could not read the reconstructed OBJ: {error}"))?
        .replace("mtllib room-mesh.mtl", &format!("mtllib {mtl_name}"));
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
        if project.processing_status == "processing" {
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

#[tauri::command]
pub fn export_gaussian_splat(
    project_path: String,
    destination_path: String,
) -> Result<String, String> {
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
    write_export(
        &destination,
        &fs::read(&source).map_err(|error| format!("Could not read Gaussian PLY: {error}"))?,
    )?;
    let stem = destination
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("room-splat");
    for (source_name, suffix) in [
        ("room-splat.transform.json", "transform.json"),
        ("splat-manifest.json", "manifest.json"),
    ] {
        let sidecar = root.join("outputs").join(source_name);
        if sidecar.is_file() {
            let destination_sidecar = destination.with_file_name(format!("{stem}.{suffix}"));
            write_export(
                &destination_sidecar,
                &fs::read(&sidecar).map_err(|error| error.to_string())?,
            )?;
        }
    }
    Ok(destination.to_string_lossy().into_owned())
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
    // Normals use the inverse transpose of the scale before the same rotation
    // as positions. transformed_direction applies scale once, so pre-dividing
    // by scale squared produces R * inverse(S) here.
    let adjusted = std::array::from_fn(|axis| {
        let scale = transform.scale[axis];
        if scale.abs() > f32::EPSILON {
            normal[axis] / (scale * scale)
        } else {
            normal[axis]
        }
    });
    let transformed = transformed_direction(adjusted, transform);
    let length = transformed.iter().map(|value| value * value).sum::<f32>().sqrt();
    if length > f32::EPSILON {
        transformed.map(|value| value / length)
    } else {
        transformed
    }
}

fn transform_camera_frame(frame: &mut CameraFrame, transform: &CloudTransform) {
    let origin = transformed_position(
        [frame.matrix[3], frame.matrix[7], frame.matrix[11]],
        transform,
    );
    for column in 0..3 {
        let direction = transformed_direction(
            [
                frame.matrix[column],
                frame.matrix[4 + column],
                frame.matrix[8 + column],
            ],
            transform,
        );
        frame.matrix[column] = direction[0];
        frame.matrix[4 + column] = direction[1];
        frame.matrix[8 + column] = direction[2];
    }
    frame.matrix[3] = origin[0];
    frame.matrix[7] = origin[1];
    frame.matrix[11] = origin[2];
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

#[tauri::command]
pub fn apply_cloud_transform(
    project_path: String,
    transform: CloudTransform,
) -> Result<Vec<PreviewPoint>, String> {
    let root = PathBuf::from(project_path);
    let project = storage::read_project(&root)?;
    if project.processing_status != "complete" {
        return Err(
            "Build the point cloud before applying its orientation to the export".to_string(),
        );
    }
    let output_root = root.join("outputs");
    let ply_path = output_root.join("room-cloud.ply");
    let backup_path = output_root.join("room-cloud.untransformed.ply");
    if !backup_path.exists() {
        fs::copy(&ply_path, &backup_path).map_err(|error| error.to_string())?;
    }

    let mut bytes = fs::read(&ply_path).map_err(|error| error.to_string())?;
    let marker = b"end_header\n";
    let payload_start = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .map(|position| position + marker.len())
        .ok_or_else(|| "The generated PLY header is invalid".to_string())?;
    for vertex in bytes[payload_start..].chunks_exact_mut(15) {
        let position = [
            f32::from_le_bytes(vertex[0..4].try_into().unwrap()),
            f32::from_le_bytes(vertex[4..8].try_into().unwrap()),
            f32::from_le_bytes(vertex[8..12].try_into().unwrap()),
        ];
        let transformed = transformed_position(position, &transform);
        vertex[0..4].copy_from_slice(&transformed[0].to_le_bytes());
        vertex[4..8].copy_from_slice(&transformed[1].to_le_bytes());
        vertex[8..12].copy_from_slice(&transformed[2].to_le_bytes());
    }
    let temporary = output_root.join("room-cloud.ply.tmp");
    File::create(&temporary)
        .and_then(|mut file| file.write_all(&bytes))
        .map_err(|error| error.to_string())?;
    fs::remove_file(&ply_path).map_err(|error| error.to_string())?;
    fs::rename(&temporary, &ply_path).map_err(|error| error.to_string())?;

    let preview_path = output_root.join("preview.json");
    let mut preview: Vec<PreviewPoint> =
        serde_json::from_reader(File::open(&preview_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    for point in &mut preview {
        point.position = transformed_position(point.position, &transform);
    }
    storage::write_json(&preview_path, &preview)?;

    let camera_path = output_root.join("camera-poses.json");
    if camera_path.is_file() {
        let camera_backup = output_root.join("camera-poses.untransformed.json");
        if !camera_backup.exists() {
            fs::copy(&camera_path, &camera_backup).map_err(|error| error.to_string())?;
        }
        let mut frames: Vec<CameraFrame> =
            serde_json::from_reader(File::open(&camera_path).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        for frame in &mut frames {
            transform_camera_frame(frame, &transform);
        }
        storage::write_json(&camera_path, &frames)?;
    }

    let mesh_path = output_root.join("room-mesh.obj");
    if mesh_path.is_file() {
        let mesh_backup = output_root.join("room-mesh.untransformed.obj");
        if !mesh_backup.exists() {
            fs::copy(&mesh_path, &mesh_backup).map_err(|error| error.to_string())?;
        }
        let transformed = transformed_obj(
            &fs::read_to_string(&mesh_path).map_err(|error| error.to_string())?,
            &transform,
        )?;
        let temporary_mesh = output_root.join("room-mesh.obj.tmp");
        File::create(&temporary_mesh)
            .and_then(|mut file| file.write_all(transformed.as_bytes()))
            .map_err(|error| error.to_string())?;
        fs::remove_file(&mesh_path).map_err(|error| error.to_string())?;
        fs::rename(&temporary_mesh, &mesh_path).map_err(|error| error.to_string())?;
    }
    Ok(preview)
}

#[cfg(test)]
mod tests {
    use super::{
        compact_splat_preview, convert_3dgs_ply_to_splat, normalize_project, pack_preview_mesh,
        transformed_normal, transformed_position, unity_compatible_ply, valid_packed_preview_mesh,
        validate_sensor_settings,
    };
    use crate::models::{CaptureSettings, CloudTransform, ProjectSummary};
    use std::fs;

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
    fn cloud_transform_applies_axis_flip_before_translation() {
        let transform = CloudTransform {
            position: [0.5, -1.0, 2.0],
            rotation: [0.0, 0.0, 0.0],
            scale: [-1.0, 1.0, 1.0],
        };
        assert_eq!(
            transformed_position([1.0, 2.0, 3.0], &transform),
            [-0.5, 1.0, 5.0]
        );
    }

    #[test]
    fn mesh_normals_use_inverse_transpose_scale() {
        let transform = CloudTransform {
            position: [4.0, 5.0, 6.0],
            rotation: [0.0, 0.0, 0.0],
            scale: [2.0, 1.0, 0.5],
        };
        let normal = transformed_normal([1.0, 0.0, 1.0], &transform);
        let length = (0.5_f32 * 0.5 + 2.0 * 2.0).sqrt();
        assert!((normal[0] - 0.5 / length).abs() < 1e-6);
        assert_eq!(normal[1], 0.0);
        assert!((normal[2] - 2.0 / length).abs() < 1e-6);
    }

    #[test]
    fn unity_export_flips_only_x_and_preserves_rgb() {
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
            .contains("comment Unity-ready coordinates: X axis flipped"));
        assert_eq!(vertices[0], ([-1.25, 2.5, -3.75], [10, 20, 30]));
        assert_eq!(vertices[1], ([4.5, 5.75, 6.0], [40, 50, 60]));
    }

    #[test]
    fn legacy_cloud_transform_defaults_to_unit_scale() {
        let transform: CloudTransform =
            serde_json::from_str(r#"{"position":[0,0,0],"rotation":[0,0,0]}"#).unwrap();
        assert_eq!(transform.scale, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn legacy_capture_settings_default_to_narrow_unbinned_depth() {
        let settings: CaptureSettings = serde_json::from_str(
            r#"{
                "captureFps": 10,
                "maxDepthM": 4.2,
                "voxelSizeMm": 15,
                "environment": "indoor",
                "sensorKind": "azure_kinect",
                "sensorId": "",
                "sensorConnection": "usb",
                "sensorAddress": "",
                "useImu": true
            }"#,
        )
        .unwrap();
        assert_eq!(settings.depth_field_of_view, "narrow");
        assert!(!settings.depth_binned);
        assert_eq!(settings.rgb_jpeg_quality, 92);
        assert_eq!(settings.max_rgb_dimension, 0);
        assert_eq!(settings.live_reconstruction, "points");
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
}
