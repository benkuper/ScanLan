use crate::models::{
    AvailableSensor, CaptureSettings, CaptureStatus, LiveReconstructionStatus, LiveWorkerStatus,
    PreviewPoint, ProjectSummary, ReconstructionProgress, RuntimeInfo,
};
use crate::storage;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
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
    points: Option<LiveGeometryFrame>,
    mesh: Option<LiveGeometryFrame>,
    error: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RealtimeEngineStatusMessage {
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
}

#[derive(Clone)]
pub struct AppState {
    pub project: Arc<Mutex<ProjectSummary>>,
    pub active_capture: Arc<Mutex<Option<ActiveCapture>>>,
    pub jobs: crate::jobs::JobManager,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            project: Arc::new(Mutex::new(ProjectSummary::placeholder())),
            active_capture: Arc::new(Mutex::new(None)),
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
    if !matches!(settings.live_reconstruction.as_str(), "points" | "mesh") {
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
    .ok_or_else(|| {
        "Realtime reconstruction support is missing from this app build".to_string()
    })?;
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
        .arg("--session")
        .arg(phase_root)
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
                Err(error) => return Err(format!("Could not read realtime engine output: {error}")),
            }
            if &header[0..8] != b"SCANENG1" {
                return Err("Realtime engine emitted an unknown protocol".to_string());
            }
            let version = u16::from_le_bytes(header[8..10].try_into().unwrap());
            let kind = u16::from_le_bytes(header[10..12].try_into().unwrap());
            let payload_size =
                u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
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
                    snapshot.status = LiveReconstructionStatus {
                        active: message.active,
                        mode: mode.clone(),
                        tracking: message.state == "tracking",
                        tracking_status: if message.detail.is_empty() {
                            message.state
                        } else {
                            message.detail
                        },
                        processed_frames: message.processed_frames,
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
                    };
                }
                2 => {
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
                    snapshot.status.point_count = point_count as u64;
                    snapshot.points = Some(LiveGeometryFrame {
                        frame_count,
                        packet: Arc::new(payload),
                    });
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
    mesh: bool,
    after_frame: u32,
) -> Vec<u8> {
    let Ok(snapshot) = snapshot.lock() else {
        return Vec::new();
    };
    let geometry = if mesh {
        snapshot.mesh.as_ref()
    } else {
        snapshot.points.as_ref()
    };
    match geometry {
        Some(frame) if frame.frame_count != after_frame => frame.packet.as_ref().clone(),
        _ => Vec::new(),
    }
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
    let body = capture_root
        .map(|snapshot| realtime_packet(&snapshot, false, after_frame))
        .unwrap_or_default();
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
        realtime_packet(&snapshot, true, after_frame)
    } else {
        Vec::new()
    };
    Ok(tauri::ipc::Response::new(body))
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
    validate_sensor_settings(&mut settings)?;
    project.settings = settings;
    storage::write_project(&project)?;
    write_sensor_preference(&app, &project.settings)?;
    *state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())? = project.clone();
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
    let phase_id = Uuid::new_v4().to_string();
    let phase_name = format!(
        "{} phase {}",
        sensor_name(&project.settings),
        project.phases.len() + 1
    );
    let phase_root = project_root.join("phases").join(&phase_id);
    fs::create_dir_all(&phase_root).map_err(|error| error.to_string())?;

    // PyInstaller's one-file Open3D runtime can take several seconds to unpack
    // on a cold launch. Warm it completely before opening the sensor so the
    // first captured frames can already contribute to the visible map.
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
    let pending_capture = ActiveCapture {
        child,
        sensor_relay: Some(sensor_relay),
        live_reconstruction,
        realtime,
        project_root,
        phase_root,
        phase_id,
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
            tracking_fps: 0.0,
            source_drop_count: 0,
            tracking_queue_drop_count: 0,
            mapping_drop_count: 0,
            tracking_overlap: 0.0,
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

    if let Some((phase_root, phase_id, realtime)) = active_snapshot {
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
        if let Some(phase) = project.phases.iter_mut().find(|phase| phase.id == phase_id) {
            phase.frame_count = frame_count;
            phase.duration_seconds = frame_count / project.settings.capture_fps.max(1);
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
            tracking_overlap: live_reconstruction
                .as_ref()
                .map(|status| status.overlap)
                .unwrap_or(0.0),
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
            tracking_fps: 0.0,
            source_drop_count: 0,
            tracking_queue_drop_count: 0,
            mapping_drop_count: 0,
            tracking_overlap: 0.0,
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
        preview_point_count: preview.len() as u64,
        preview,
        capturing: false,
        sensor_connected: false,
        sensor_paused: false,
        sensor_status: format!("{} opens when capture starts", selected_sensor_name),
        sensor_name: selected_sensor_name,
        frame_count: 0,
        total_frame_count,
        stream_fps: 0.0,
        tracking: false,
        tracking_status: "Ready to capture".to_string(),
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
        tracking_overlap: 0.0,
        depth_rmse_mm: None,
        live_reconstruction_backend: None,
        reconstruction: reconstruction_progress(&project_root),
        error: None,
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
        drain_sensor_relay(&mut capture, Duration::from_secs(1));

        let manifest_path = capture.phase_root.join("phase.json");
        let capture_summary = File::open(&manifest_path)
            .ok()
            .and_then(|file| {
                serde_json::from_reader::<_, crate::models::PhaseManifest>(file).ok()
            })
            .map(|manifest| (manifest.frame_count, manifest.duration_seconds))
            .unwrap_or((0, 0));
        let mut project = storage::read_project(&capture.project_root)?;
        let frame_count = capture_summary
            .0
            .max(indexed_frame_count(&capture.phase_root));
        let duration_seconds = capture_summary.1.max(
            (frame_count / project.settings.capture_fps.max(1)).max(u32::from(frame_count > 0)),
        );
        let clean_stop = stop_error.is_none()
            && status.as_ref().is_some_and(|status| status.success());
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


#[cfg(test)]
mod tests {
    use super::{
        compact_splat_preview, convert_3dgs_ply_to_splat, normalize_project, pack_preview_mesh,
        unity_compatible_ply, valid_packed_preview_mesh, validate_sensor_settings,
    };
    use crate::models::{CaptureSettings, ProjectSummary};
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
