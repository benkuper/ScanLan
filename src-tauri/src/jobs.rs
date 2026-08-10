use crate::commands::{worker_command, AppState};
use crate::models::{ArtifactJob, ProjectSummary};
use crate::storage;
use chrono::Utc;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};
use tauri::{AppHandle, Manager, State};
use uuid::Uuid;

#[cfg(windows)]
use std::os::windows::io::AsRawHandle;
#[cfg(windows)]
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
#[cfg(windows)]
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

#[derive(Clone, Default)]
pub struct JobManager {
    cancellations: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    accelerator_owner: Arc<Mutex<Option<String>>>,
}

impl JobManager {
    pub fn cancel_all(&self) {
        if let Ok(cancellations) = self.cancellations.lock() {
            for cancellation in cancellations.values() {
                cancellation.store(true, Ordering::SeqCst);
            }
        }
    }

    pub fn is_running(&self, job_id: &str) -> bool {
        self.cancellations
            .lock()
            .is_ok_and(|jobs| jobs.contains_key(job_id))
    }
}

fn validate_job_id(job_id: &str) -> Result<(), String> {
    Uuid::parse_str(job_id)
        .map(|_| ())
        .map_err(|_| "The artifact job identifier is invalid".to_string())
}

fn splat_checkpoint_available(project_root: &Path, job: &ArtifactJob) -> bool {
    job.targets.iter().any(|target| target == "gaussianSplat")
        && project_root
            .join("outputs")
            .join("splat-checkpoint.pt")
            .is_file()
}

fn accelerator_lock_path() -> PathBuf {
    std::env::temp_dir().join("scanlan-artifact-accelerator.lock")
}

fn acquire_accelerator_lock_at(path: &Path) -> Result<File, String> {
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| format!("Could not open the reconstruction accelerator lock: {error}"))?;
    fs2::FileExt::try_lock_exclusive(&lock).map_err(|error| {
        if matches!(
            error.kind(),
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::PermissionDenied
        ) || error.raw_os_error() == Some(33)
        {
            "Another ScanLan artifact worker is already using the reconstruction accelerator"
                .to_string()
        } else {
            format!("Could not lock the reconstruction accelerator: {error}")
        }
    })?;
    Ok(lock)
}

fn acquire_accelerator_lock() -> Result<File, String> {
    acquire_accelerator_lock_at(&accelerator_lock_path())
}

#[cfg(windows)]
struct ChildLifetimeGuard(HANDLE);

#[cfg(windows)]
impl ChildLifetimeGuard {
    fn attach(child: &Child) -> Result<Self, String> {
        // A kill-on-close Job Object makes Windows terminate the worker even if
        // the app is force-closed or the Tauri dev runner replaces the process.
        // Without it, an orphan can keep writing the shared project progress and
        // checkpoint files while a restarted app launches a second worker.
        unsafe {
            let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if handle.is_null() {
                return Err(format!(
                    "Could not create the artifact worker lifetime guard: {}",
                    std::io::Error::last_os_error()
                ));
            }
            let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            ) == 0
            {
                let error = std::io::Error::last_os_error();
                CloseHandle(handle);
                return Err(format!(
                    "Could not configure the artifact worker lifetime guard: {error}"
                ));
            }
            if AssignProcessToJobObject(handle, child.as_raw_handle() as HANDLE) == 0 {
                let error = std::io::Error::last_os_error();
                CloseHandle(handle);
                return Err(format!(
                    "Could not attach the artifact worker lifetime guard: {error}"
                ));
            }
            Ok(Self(handle))
        }
    }
}

#[cfg(windows)]
impl Drop for ChildLifetimeGuard {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}

#[cfg(not(windows))]
struct ChildLifetimeGuard;

#[cfg(not(windows))]
impl ChildLifetimeGuard {
    fn attach(_child: &Child) -> Result<Self, String> {
        Ok(Self)
    }
}

pub fn recover_interrupted_job(project: &mut ProjectSummary, manager: &JobManager) -> bool {
    let Some(job_id) = project.active_job.clone() else {
        return false;
    };
    if manager.is_running(&job_id) {
        return false;
    }
    let root = Path::new(&project.path);
    let mut resumable = false;
    if let Ok(mut job) = read_job(root, &job_id) {
        if matches!(job.status.as_str(), "queued" | "running" | "cancelling") {
            job.status = "failed".to_string();
            job.stage = "interrupted".to_string();
            job.error = Some("The app closed while this artifact job was running".to_string());
            job.resumable = splat_checkpoint_available(root, &job);
            resumable = job.resumable;
            job.updated_at = Utc::now().to_rfc3339();
            let _ = write_job(root, &job);
        }
    }
    project.active_job = None;
    project.processing_status = "failed".to_string();
    project.processing_error = Some(if resumable {
        "The previous artifact job was interrupted; existing artifacts are safe and its Gaussian checkpoint can be resumed."
            .to_string()
    } else {
        "The previous artifact job was interrupted; existing completed artifacts are safe, but the stopped job must be restarted."
            .to_string()
    });
    true
}

fn job_path(project_root: &Path, job_id: &str) -> PathBuf {
    project_root
        .join("outputs")
        .join("jobs")
        .join(format!("{job_id}.json"))
}

fn read_job(project_root: &Path, job_id: &str) -> Result<ArtifactJob, String> {
    let file = File::open(job_path(project_root, job_id)).map_err(|error| error.to_string())?;
    serde_json::from_reader(file).map_err(|error| error.to_string())
}

fn write_job(project_root: &Path, job: &ArtifactJob) -> Result<(), String> {
    storage::write_json(&job_path(project_root, &job.id), job)
}

fn source_fingerprint(project_root: &Path, project: &ProjectSummary) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    project.schema_version.hash(&mut hasher);
    project.settings.capture_fps.hash(&mut hasher);
    project.settings.voxel_size_mm.hash(&mut hasher);
    project.settings.repair_mesh.hash(&mut hasher);
    project.settings.mesh_repair_profile.hash(&mut hasher);
    project.settings.fill_inferred_mesh_holes.hash(&mut hasher);
    project.settings.produce_watertight_mesh.hash(&mut hasher);
    for phase in &project.phases {
        phase.id.hash(&mut hasher);
        phase.frame_count.hash(&mut hasher);
        for name in ["phase.json", "frames.csv"] {
            let path = project_root.join("phases").join(&phase.id).join(name);
            hash_file_metadata(&path, &mut hasher);
        }
    }
    for source in &project.media_sources {
        source.id.hash(&mut hasher);
        source.kind.hash(&mut hasher);
        source.byte_size.hash(&mut hasher);
        hash_file_metadata(&project_root.join(&source.path), &mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

fn hash_file_metadata(path: &Path, hasher: &mut impl Hasher) {
    path.to_string_lossy().hash(hasher);
    if let Ok(metadata) = fs::metadata(path) {
        metadata.len().hash(hasher);
        metadata
            .modified()
            .ok()
            .and_then(|value| value.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|value| value.as_nanos())
            .unwrap_or_default()
            .hash(hasher);
    }
}

fn progress_file(project_root: &Path, splat: bool) -> PathBuf {
    project_root.join("outputs").join(if splat {
        "splat-progress.json"
    } else {
        "progress.json"
    })
}

fn stage_plan(job: &ArtifactJob) -> Vec<(&'static str, f32)> {
    if job.source_kind == "media" {
        return vec![
            ("prepare", 0.05),
            ("media", 0.35),
            ("splat", 0.55),
            ("publish", 0.05),
        ];
    }
    if job.source_kind == "hybrid" {
        return vec![
            ("prepare", 0.05),
            ("track", 0.12),
            ("trajectory", 0.10),
            ("fuse", 0.08),
            ("cloud", 0.18),
            ("media", 0.18),
            ("dataset", 0.08),
            ("mesh", 0.18),
            ("splat", 0.55),
            ("publish", 0.05),
        ];
    }
    let wants_mesh = job.targets.iter().any(|target| target == "texturedMesh");
    let wants_splat = job.targets.iter().any(|target| target == "gaussianSplat");
    let mut plan = vec![
        ("prepare", 0.05),
        ("track", 0.13),
        ("trajectory", 0.12),
        ("fuse", 0.08),
        ("cloud", 0.24),
    ];
    if wants_splat {
        plan.push(("dataset", 0.08));
    }
    if wants_mesh {
        plan.push(("mesh", 0.20));
    }
    if wants_splat {
        plan.push(("splat", 0.55));
    }
    plan.push(("publish", 0.05));
    plan
}

fn stage_key(stage: &str) -> Option<&'static str> {
    let stage = stage.to_ascii_lowercase().replace('_', " ");
    if stage.contains("complete") || stage.contains("export") || stage.contains("publish") {
        Some("publish")
    } else if stage.contains("media")
        || stage.contains("lingbot")
        || stage.contains("da3")
        || stage.contains("camera refinement")
        || stage.contains("feature")
        || stage.contains("camera solving")
        || stage.contains("undistort")
    {
        Some("media")
    } else if stage.contains("splat") && (stage.contains("train") || stage.contains("initial")) {
        Some("splat")
    } else if stage.contains("mesh")
        || stage.contains("textur")
        || stage.contains("topology")
        || stage.contains("opening")
        || stage.contains("repairing")
        || stage.contains("validating")
    {
        Some("mesh")
    } else if stage.contains("preparing splat") || stage.contains("posed frame") {
        Some("dataset")
    } else if stage.contains("building") || stage.contains("cleaning cloud") {
        Some("cloud")
    } else if stage.contains("fusing")
        || stage.contains("previewing")
        || stage.contains("loading cache")
    {
        Some("fuse")
    } else if stage.contains("stabiliz") || stage.contains("aligning") {
        Some("trajectory")
    } else if stage.contains("tracking") || stage.contains("placing") || stage.contains("keyframe")
    {
        Some("track")
    } else if matches!(stage.as_str(), "queued" | "preparing" | "resuming") {
        Some("prepare")
    } else {
        None
    }
}

fn planned_progress(job: &ArtifactJob, stage_progress: Option<f32>) -> Option<f32> {
    let plan = stage_plan(job);
    let current = stage_key(&job.stage)?;
    let total_weight: f32 = plan.iter().map(|(_, weight)| *weight).sum();
    let index = plan.iter().position(|(key, _)| *key == current)?;
    let completed_weight: f32 = plan[..index].iter().map(|(_, weight)| *weight).sum();
    let fraction = if current == "publish" && job.stage.to_ascii_lowercase().contains("complete") {
        1.0
    } else {
        stage_progress.unwrap_or(0.0).clamp(0.0, 1.0)
    };
    Some(((completed_weight + plan[index].1 * fraction) / total_weight).clamp(0.0, 1.0))
}

fn update_job_elapsed(job: &mut ArtifactJob) {
    let now = Utc::now();
    let Some(started) = job
        .started_at
        .as_deref()
        .and_then(|value| chrono::DateTime::parse_from_rfc3339(value).ok())
    else {
        return;
    };
    let elapsed = now
        .signed_duration_since(started.with_timezone(&Utc))
        .num_seconds()
        .max(0) as u32;
    job.elapsed_seconds = Some(elapsed);
}

fn merge_progress(project_root: &Path, job: &mut ArtifactJob, splat: bool) {
    let Ok(file) = File::open(progress_file(project_root, splat)) else {
        return;
    };
    let Ok(value) = serde_json::from_reader::<_, Value>(file) else {
        return;
    };
    let reconstruction_finished_before_splat = value
        .get("stage")
        .and_then(Value::as_str)
        .is_some_and(|stage| {
            !splat
                && job.targets.iter().any(|target| target == "gaussianSplat")
                && (stage.eq_ignore_ascii_case("complete")
                    || stage.eq_ignore_ascii_case("exporting"))
        });
    if let Some(stage) = value.get("stage").and_then(Value::as_str) {
        job.stage = if reconstruction_finished_before_splat {
            "splat_initializing".to_string()
        } else {
            stage.to_string()
        };
    }
    if reconstruction_finished_before_splat {
        job.detail = "RGB-D artifacts ready · initializing CUDA Gaussian optimization".to_string();
    } else if let Some(detail) = value.get("detail").and_then(Value::as_str) {
        job.detail = detail.to_string();
    }
    let worker_progress = value
        .get("progress")
        .and_then(Value::as_f64)
        .map(|value| value as f32);
    let explicit_stage_progress = value
        .get("stageProgress")
        .and_then(Value::as_f64)
        .map(|value| value as f32);
    job.stage_progress = if reconstruction_finished_before_splat {
        Some(0.0)
    } else {
        explicit_stage_progress.or(if splat { worker_progress } else { None })
    };
    if let Some(progress) = planned_progress(job, job.stage_progress) {
        job.progress = job.progress.max(progress);
    }
    job.iteration = value
        .get("iteration")
        .and_then(Value::as_u64)
        .map(|value| value as u32)
        .or(job.iteration);
    job.total_iterations = value
        .get("totalIterations")
        .and_then(Value::as_u64)
        .map(|value| value as u32)
        .or(job.total_iterations);
    job.loss = value
        .get("loss")
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .or(job.loss);
    job.smoothed_loss = value
        .get("smoothedLoss")
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .or(job.smoothed_loss);
    job.stage_eta_seconds = value
        .get("stageEtaSeconds")
        .and_then(Value::as_u64)
        .map(|value| value as u32)
        .or_else(|| {
            splat
                .then(|| {
                    value
                        .get("etaSeconds")
                        .and_then(Value::as_u64)
                        .map(|value| value as u32)
                })
                .flatten()
        });
    // Only workers observe actual work units and throughput.  In particular,
    // overall weighted progress includes stage weights that are not a clock,
    // so extrapolating it produced ETAs that grew forever while an opaque
    // COLMAP call was still making progress.  Preserve an absent worker ETA as
    // absent instead of inventing one in the desktop process.
    job.eta_seconds = value
        .get("etaSeconds")
        .and_then(Value::as_u64)
        .map(|value| value as u32);
    job.compute_backend = value
        .get("computeBackend")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| splat.then(|| "CUDA AMP / gsplat".to_string()))
        .or_else(|| job.compute_backend.clone());
    if let Some(preview) = value
        .get("metrics")
        .and_then(|metrics| metrics.get("rgbPreview"))
    {
        job.rgb_preview_active = preview
            .get("active")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        job.rgb_preview_scale_status = preview
            .get("scaleStatus")
            .and_then(Value::as_str)
            .map(str::to_string);
        job.rgb_preview_confidence = preview
            .get("confidence")
            .and_then(Value::as_f64)
            .map(|value| value as f32);
        job.rgb_preview_drift_risk = preview
            .get("driftRisk")
            .and_then(Value::as_f64)
            .map(|value| value as f32);
        job.rgb_preview_submap_count = preview
            .get("residentSubmapCount")
            .and_then(Value::as_u64)
            .map(|value| value as u32);
        job.rgb_preview_accepted_frames = preview
            .get("acceptedFrameCount")
            .and_then(Value::as_u64)
            .map(|value| value as u32);
        job.rgb_preview_rejected_frames = preview
            .get("rejectedFrameCount")
            .and_then(Value::as_u64)
            .map(|value| value as u32);
    } else if job.stage != "rgb_preview_streaming" {
        job.rgb_preview_active = false;
    }
    update_job_elapsed(job);
    job.updated_at = Utc::now().to_rfc3339();
    let _ = write_job(project_root, job);
}

fn append_log_reader<R: std::io::Read + Send + 'static>(reader: R, log: Arc<Mutex<File>>) {
    thread::spawn(move || {
        for line in BufReader::new(reader).lines().map_while(Result::ok) {
            if let Ok(mut output) = log.lock() {
                let _ = writeln!(output, "{line}");
                let _ = output.flush();
            }
        }
    });
}

fn run_command(
    mut command: Command,
    project_root: &Path,
    job: &mut ArtifactJob,
    cancel: &AtomicBool,
    splat: bool,
) -> Result<(), String> {
    let log_path = project_root.join(&job.log_path);
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|error| error.to_string())?;
    let log = Arc::new(Mutex::new(log_file));
    if let Ok(mut output) = log.lock() {
        let _ = writeln!(
            output,
            "\n[{}] starting {:?}",
            Utc::now().to_rfc3339(),
            command
        );
    }
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("Could not start artifact worker: {error}"))?;
    let _lifetime_guard = match ChildLifetimeGuard::attach(&child) {
        Ok(guard) => guard,
        Err(error) => {
            child.kill().ok();
            child.wait().ok();
            return Err(error);
        }
    };
    if let Some(stdout) = child.stdout.take() {
        append_log_reader(stdout, Arc::clone(&log));
    }
    if let Some(stderr) = child.stderr.take() {
        append_log_reader(stderr, Arc::clone(&log));
    }
    let mut cancellation_started = None;
    loop {
        if cancel.load(Ordering::SeqCst) && cancellation_started.is_none() {
            File::create(project_root.join("outputs").join("cancel.flag")).ok();
            cancellation_started = Some(Instant::now());
        }
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            if cancel.load(Ordering::SeqCst) {
                merge_progress(project_root, job, splat);
                return Err("Artifact job cancelled".to_string());
            }
            if status.success() {
                merge_progress(project_root, job, splat);
                return Ok(());
            }
            merge_progress(project_root, job, splat);
            if job.stage == "failed" && !job.detail.trim().is_empty() {
                return Err(job.detail.clone());
            }
            return Err(format!(
                "Artifact worker exited with {status}; see {}",
                log_path.display()
            ));
        }
        if cancellation_started.is_some_and(|started| {
            started.elapsed()
                >= if splat {
                    Duration::from_secs(15)
                } else {
                    Duration::from_secs(1)
                }
        }) {
            child.kill().ok();
            child.wait().ok();
            return Err("Artifact job cancelled".to_string());
        }
        merge_progress(project_root, job, splat);
        thread::sleep(Duration::from_millis(150));
    }
}

fn existing_runtime(resources: Option<&Path>, splat: bool) -> Result<PathBuf, String> {
    let candidates = if splat {
        storage::candidate_splat_worker_paths(resources)
    } else {
        storage::candidate_reconstruction_worker_paths(resources)
    };
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| {
            if splat {
                "Gaussian-splat support is not installed. Run npm run prepare:splat.".to_string()
            } else {
                "Reconstruction support is missing from this app build".to_string()
            }
        })
}

fn existing_geometry_runtime(resources: Option<&Path>) -> Result<PathBuf, String> {
    storage::candidate_geometry_worker_paths(resources)
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| {
            "Learned geometry support is not installed. Run npm run prepare:splat.".to_string()
        })
}

fn current_media_observation_manifest(project_root: &Path) -> Result<PathBuf, String> {
    let observations_root = project_root
        .join("outputs")
        .join("cache")
        .join("media-observations");
    let pointer_path = observations_root.join("current.json");
    let pointer: Value = serde_json::from_reader(
        File::open(&pointer_path)
            .map_err(|error| format!("Media observation pointer is missing: {error}"))?,
    )
    .map_err(|error| format!("Media observation pointer is invalid: {error}"))?;
    let relative = Path::new(
        pointer
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| "Media observation pointer has no path".to_string())?,
    );
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err("Media observation pointer contains an unsafe path".to_string());
    }
    let manifest = observations_root.join(relative).join("observations.json");
    if !manifest.is_file() {
        return Err(format!(
            "Media observation manifest is missing: {}",
            manifest.display()
        ));
    }
    Ok(manifest)
}

fn remove_cache_directory(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    if !path.is_dir() {
        return Err(format!("Expected a cache directory at {}", path.display()));
    }
    fs::remove_dir_all(path)
        .map_err(|error| format!("Could not discard cache {}: {error}", path.display()))
}

fn is_media_observation_record(value: &Value) -> bool {
    value
        .get("sourcePath")
        .and_then(Value::as_str)
        .is_some_and(|source| {
            source
                .replace('\\', "/")
                .to_ascii_lowercase()
                .contains("/outputs/cache/media-observations/")
        })
}

fn discard_media_localizations(project_root: &Path) -> Result<(), String> {
    let manifest_path = project_root.join("supplemental-photos.json");
    if !manifest_path.is_file() {
        return Ok(());
    }
    let mut manifest: Value =
        serde_json::from_reader(File::open(&manifest_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("Supplemental-photo manifest is invalid: {error}"))?;
    let mut changed = false;
    for collection in ["photos", "attempts"] {
        if let Some(records) = manifest.get_mut(collection).and_then(Value::as_array_mut) {
            let before = records.len();
            records.retain(|record| !is_media_observation_record(record));
            changed |= records.len() != before;
        }
    }
    if changed {
        storage::write_json(&manifest_path, &manifest)?;
    }
    fs::remove_file(
        project_root
            .join("outputs")
            .join("photo-localization-progress.json"),
    )
    .ok();
    Ok(())
}

fn invalidate_pipeline_cache(project_root: &Path, job: &ArtifactJob) -> Result<(), String> {
    let cache_root = project_root.join("outputs").join("cache");
    let media_restart = if job.media_restart.is_empty() {
        "reuse"
    } else {
        job.media_restart.as_str()
    };
    if matches!(media_restart, "analysis" | "decode") || job.rebuild_rgbd {
        remove_cache_directory(&cache_root.join("datasets"))?;
        discard_media_localizations(project_root)?;
    }
    if media_restart == "decode" {
        remove_cache_directory(&cache_root.join("media-observations"))?;
    }
    if job.rebuild_rgbd {
        for directory in ["local-phases", "meshes", "mesh-repair", "lingbot-depth"] {
            remove_cache_directory(&cache_root.join(directory))?;
        }
    }
    Ok(())
}

fn run_pipeline(
    resources: Option<&Path>,
    project_root: &Path,
    job: &mut ArtifactJob,
    cancel: &AtomicBool,
    iterations: u32,
    resume: bool,
) -> Result<(), String> {
    fs::remove_file(project_root.join("outputs").join("cancel.flag")).ok();
    if !resume {
        invalidate_pipeline_cache(project_root, job)?;
    }
    let project = storage::read_project(project_root)?;
    if job.source_kind == "media" {
        let splat_worker = existing_runtime(resources, true)?;
        job.stage = "media_preparation".to_string();
        job.detail = "Selecting sharp views and solving cameras".to_string();
        job.stage_progress = Some(0.0);
        write_job(project_root, job)?;
        let mut prepare = worker_command(&splat_worker);
        prepare
            .arg("prepare-media")
            .arg("--project")
            .arg(project_root)
            // Inspect video densely, then let optical-flow keyframing retain
            // views according to camera motion and tracked visual overlap.
            .arg("--video-fps")
            .arg("15")
            .arg("--maximum-video-frames")
            .arg("3000");
        let geometry_worker = existing_geometry_runtime(resources)?;
        prepare.arg("--geometry-worker").arg(geometry_worker);
        if project
            .media_sources
            .iter()
            .any(|source| source.kind == "video")
        {
            if project.settings.experimental_rgb_preview {
                prepare.arg("--progressive-rgb-preview");
            }
        }
        run_command(prepare, project_root, job, cancel, true)?;
        job.stage = "splat_training".to_string();
        job.detail = "Initializing photoreal 3D Gaussian optimization".to_string();
        job.stage_progress = Some(0.0);
        write_job(project_root, job)?;
        let dataset = project_root
            .join("outputs")
            .join("cache")
            .join("datasets")
            .join("current.json");
        let mut train = worker_command(&splat_worker);
        train
            .arg("train")
            .arg("--project")
            .arg(project_root)
            .arg("--dataset")
            .arg(dataset)
            .arg("--iterations")
            .arg(iterations.to_string());
        if resume {
            train.arg("--resume");
        }
        return run_command(train, project_root, job, cancel, true);
    }
    let reconstruction = existing_runtime(resources, false)?;
    let depth_refinement_backend = match project.settings.depth_refinement_backend.as_str() {
        "lingbot" | "mapanything" | "da3" => project.settings.depth_refinement_backend.as_str(),
        _ if project.settings.lingbot_depth_refinement => "lingbot",
        _ => "off",
    };
    let depth_refiner = if depth_refinement_backend != "off" {
        Some(existing_geometry_runtime(resources)?)
    } else {
        None
    };
    let targets = job
        .targets
        .iter()
        .map(|target| match target.as_str() {
            "pointCloud" => "point_cloud",
            "texturedMesh" => "textured_mesh",
            "gaussianSplat" => "gaussian_splat",
            value => value,
        })
        .collect::<Vec<_>>()
        .join(",");
    let reconstruction_command = |selected_targets: &str| {
        let mut command = worker_command(&reconstruction);
        command
            .arg("reconstruct")
            .arg(project_root)
            .arg("--engine")
            .arg("auto")
            .arg("--targets")
            .arg(selected_targets)
            .arg("--mesh-repair")
            .arg(if project.settings.repair_mesh {
                "on"
            } else {
                "off"
            })
            .arg("--mesh-repair-profile")
            .arg(&project.settings.mesh_repair_profile)
            .arg(if project.settings.fill_inferred_mesh_holes {
                "--fill-inferred-holes"
            } else {
                "--no-fill-inferred-holes"
            })
            .arg(if project.settings.produce_watertight_mesh {
                "--produce-watertight-copy"
            } else {
                "--no-produce-watertight-copy"
            })
            .arg("--mesh-repair-fallback");
        if let Some(refiner) = depth_refiner.as_ref() {
            command
                .arg("--depth-refinement")
                .arg(depth_refinement_backend)
                .arg("--depth-refiner")
                .arg(refiner);
        }
        command
    };
    if job.source_kind == "hybrid" {
        // First publish only the metric trajectory and RGB-D anchor cameras.
        // The final reconstruction below reuses its caches after media poses
        // have been validated and made available to all artifact builders.
        let base = reconstruction_command("localization_map");
        run_command(base, project_root, job, cancel, false)?;

        let splat_worker = existing_runtime(resources, true)?;
        let mut extract = worker_command(&splat_worker);
        extract
            .arg("extract-media")
            .arg("--project")
            .arg(project_root)
            .arg("--video-fps")
            .arg("15")
            .arg("--maximum-video-frames")
            .arg("3000");
        run_command(extract, project_root, job, cancel, true)?;

        let observation_manifest = current_media_observation_manifest(project_root)?;
        let mut localize = worker_command(&reconstruction);
        localize
            .arg("localize-media")
            .arg(project_root)
            .arg(observation_manifest);
        run_command(localize, project_root, job, cancel, false)?;
    }
    let command = reconstruction_command(&targets);
    run_command(command, project_root, job, cancel, false)?;
    if job.targets.iter().any(|target| target == "gaussianSplat") {
        job.stage = "splat_training".to_string();
        job.detail = "Initializing depth-aware 2D Gaussian optimization".to_string();
        job.stage_progress = Some(0.0);
        if let Some(progress) = planned_progress(job, job.stage_progress) {
            job.progress = job.progress.max(progress);
        }
        write_job(project_root, job)?;
        let splat_worker = existing_runtime(resources, true)?;
        let mut command = worker_command(&splat_worker);
        let dataset = project_root
            .join("outputs")
            .join("cache")
            .join("datasets")
            .join("current.json");
        command
            .arg("train")
            .arg("--project")
            .arg(project_root)
            .arg("--dataset")
            .arg(dataset)
            .arg("--iterations")
            .arg(iterations.to_string());
        if resume {
            command.arg("--resume");
        }
        run_command(command, project_root, job, cancel, true)?;
    }
    Ok(())
}

fn update_project_job(
    root: &Path,
    state: &Arc<Mutex<ProjectSummary>>,
    job_id: Option<String>,
    failed: Option<&str>,
) {
    if let Ok(mut project) = storage::read_project(root) {
        project.active_job = job_id;
        if let Some(error) = failed {
            project.processing_status = "failed".to_string();
            project.processing_error = Some(error.to_string());
        } else if project.active_job.is_some() {
            project.processing_status = "processing".to_string();
            project.processing_error = None;
        } else {
            project.processing_status = "complete".to_string();
            project.processing_error = None;
        }
        let _ = storage::write_project(&project);
        if let Ok(mut current) = state.lock() {
            *current = project;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_job(
    manager: JobManager,
    resources: Option<PathBuf>,
    project_root: PathBuf,
    mut job: ArtifactJob,
    iterations: u32,
    resume: bool,
    project_state: Arc<Mutex<ProjectSummary>>,
) -> Result<ArtifactJob, String> {
    let cancel = Arc::new(AtomicBool::new(false));
    // This OS lock complements the in-memory owner. It covers multiple ScanLan
    // app instances and is released automatically if the owning app exits.
    let accelerator_lock = acquire_accelerator_lock()?;
    {
        let mut owner = manager
            .accelerator_owner
            .lock()
            .map_err(|_| "Job manager is unavailable".to_string())?;
        if let Some(active) = owner.as_ref() {
            return Err(format!(
                "Artifact job {active} is already using the reconstruction accelerator"
            ));
        }
        *owner = Some(job.id.clone());
    }
    if let Err(error) = manager
        .cancellations
        .lock()
        .map(|mut jobs| jobs.insert(job.id.clone(), Arc::clone(&cancel)))
    {
        if let Ok(mut owner) = manager.accelerator_owner.lock() {
            *owner = None;
        }
        return Err(format!("Job manager is unavailable: {error}"));
    }
    job.status = "running".to_string();
    job.started_at = Some(Utc::now().to_rfc3339());
    job.eta_seconds = None;
    job.stage_eta_seconds = None;
    job.elapsed_seconds = Some(0);
    job.updated_at = Utc::now().to_rfc3339();
    fs::remove_file(project_root.join("outputs").join("progress.json")).ok();
    fs::remove_file(project_root.join("outputs").join("splat-progress.json")).ok();
    if !resume {
        fs::remove_file(project_root.join("outputs").join("build-preview.json")).ok();
        fs::remove_file(project_root.join("outputs").join("rgb-preview-status.json")).ok();
    }
    if !resume && job.targets.iter().any(|target| target == "gaussianSplat") {
        fs::remove_file(project_root.join("outputs").join("splat-checkpoint.pt")).ok();
        fs::remove_file(
            project_root
                .join("outputs")
                .join("room-splat.preview.splat"),
        )
        .ok();
    }
    if let Err(error) = write_job(&project_root, &job) {
        if let Ok(mut cancellations) = manager.cancellations.lock() {
            cancellations.remove(&job.id);
        }
        if let Ok(mut owner) = manager.accelerator_owner.lock() {
            *owner = None;
        }
        return Err(error);
    }
    update_project_job(&project_root, &project_state, Some(job.id.clone()), None);

    let returned = job.clone();
    thread::spawn(move || {
        let _accelerator_lock = accelerator_lock;
        let result = run_pipeline(
            resources.as_deref(),
            &project_root,
            &mut job,
            cancel.as_ref(),
            iterations,
            resume,
        );
        fs::remove_file(project_root.join("outputs").join("cancel.flag")).ok();
        match result {
            Ok(()) => {
                job.status = "complete".to_string();
                job.stage = "complete".to_string();
                job.progress = 1.0;
                job.error = None;
                job.resumable = false;
                update_project_job(&project_root, &project_state, None, None);
            }
            Err(error) => {
                let cancelled = cancel.load(Ordering::SeqCst);
                job.status = if cancelled { "cancelled" } else { "failed" }.to_string();
                job.stage = job.status.clone();
                job.error = Some(error.clone());
                job.resumable = splat_checkpoint_available(&project_root, &job);
                if cancelled {
                    if let Ok(mut project) = storage::read_project(&project_root) {
                        project.active_job = None;
                        project.processing_status = "idle".to_string();
                        project.processing_error = None;
                        let _ = storage::write_project(&project);
                        if let Ok(mut current) = project_state.lock() {
                            *current = project;
                        }
                    }
                } else {
                    update_project_job(&project_root, &project_state, None, Some(&error));
                }
            }
        }
        job.updated_at = Utc::now().to_rfc3339();
        let _ = write_job(&project_root, &job);
        if let Ok(mut cancellations) = manager.cancellations.lock() {
            cancellations.remove(&job.id);
        }
        if let Ok(mut owner) = manager.accelerator_owner.lock() {
            if owner.as_deref() == Some(job.id.as_str()) {
                *owner = None;
            }
        }
    });
    Ok(returned)
}

#[tauri::command]
pub fn start_artifact_job(
    app: AppHandle,
    project_path: String,
    targets: Vec<String>,
    iterations: Option<u32>,
    media_restart: Option<String>,
    rebuild_rgbd: Option<bool>,
    state: State<'_, AppState>,
) -> Result<ArtifactJob, String> {
    let root = PathBuf::from(project_path);
    if state
        .active_capture
        .lock()
        .map_err(|_| "Capture state is unavailable".to_string())?
        .is_some()
    {
        return Err("Stop RGB-D capture before starting reconstruction".to_string());
    }
    if state
        .active_preview
        .lock()
        .map_err(|_| "Preview state is unavailable".to_string())?
        .is_some()
    {
        return Err("Stop the camera preview before starting reconstruction".to_string());
    }
    if *state
        .active_photo_localization
        .lock()
        .map_err(|_| "Photo localization state is unavailable".to_string())?
    {
        return Err(
            "Wait for texture-photo localization to finish before reconstructing".to_string(),
        );
    }
    if targets.is_empty() {
        return Err("Choose at least one artifact target".to_string());
    }
    if let Some(target) = targets.iter().find(|target| {
        !matches!(
            target.as_str(),
            "pointCloud" | "texturedMesh" | "gaussianSplat"
        )
    }) {
        return Err(format!("Unknown artifact target: {target}"));
    }
    let targets = ["pointCloud", "texturedMesh", "gaussianSplat"]
        .into_iter()
        .filter(|candidate| targets.iter().any(|target| target.as_str() == *candidate))
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut project = storage::read_project(&root)?;
    let active_project_path = state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())?
        .path
        .clone();
    if active_project_path != project.path {
        return Err("The selected project is no longer active".to_string());
    }
    if recover_interrupted_job(&mut project, &state.jobs) {
        storage::write_project(&project)?;
        if let Ok(mut current) = state.project.lock() {
            *current = project.clone();
        }
    }
    if project.active_job.is_some() || project.processing_status == "processing" {
        return Err("An artifact job is already running for this project".to_string());
    }
    let has_rgbd = project
        .phases
        .iter()
        .any(|phase| phase.status == "complete" && phase.frame_count > 0);
    let has_media = !project.media_sources.is_empty();
    if !has_rgbd && !has_media {
        return Err(
            "Capture RGB-D data or import overlapping photos/video before reconstruction"
                .to_string(),
        );
    }
    if !has_rgbd && targets.iter().any(|target| target != "gaussianSplat") {
        return Err("Photo/video projects currently produce Gaussian splats; disable point-cloud and mesh outputs".to_string());
    }
    let media_restart = media_restart.unwrap_or_else(|| "reuse".to_string());
    if !matches!(media_restart.as_str(), "reuse" | "analysis" | "decode") {
        return Err("Unknown media restart stage".to_string());
    }
    if !has_media && media_restart != "reuse" {
        return Err("This project has no imported media preparation to restart".to_string());
    }
    let rebuild_rgbd = rebuild_rgbd.unwrap_or(false);
    if rebuild_rgbd && !has_rgbd {
        return Err("This project has no RGB-D reconstruction cache to discard".to_string());
    }
    let fingerprint = source_fingerprint(&root, &project);
    let now = Utc::now().to_rfc3339();
    let id = Uuid::new_v4().to_string();
    let job = ArtifactJob {
        id: id.clone(),
        project_path: root.to_string_lossy().into_owned(),
        targets,
        source_kind: if has_rgbd && has_media {
            "hybrid"
        } else if has_rgbd {
            "rgbd"
        } else {
            "media"
        }
        .to_string(),
        media_restart,
        rebuild_rgbd,
        stage: "queued".to_string(),
        detail: "Preparing durable artifact job".to_string(),
        progress: 0.0,
        iteration: None,
        total_iterations: Some(iterations.unwrap_or(30_000).clamp(1_000, 100_000)),
        loss: None,
        smoothed_loss: None,
        eta_seconds: None,
        stage_progress: None,
        stage_eta_seconds: None,
        elapsed_seconds: None,
        compute_backend: None,
        rgb_preview_active: false,
        rgb_preview_scale_status: None,
        rgb_preview_confidence: None,
        rgb_preview_drift_risk: None,
        rgb_preview_submap_count: None,
        rgb_preview_accepted_frames: None,
        rgb_preview_rejected_frames: None,
        status: "queued".to_string(),
        created_at: now.clone(),
        started_at: None,
        updated_at: now,
        source_fingerprint: fingerprint,
        log_path: format!("outputs/jobs/{id}.log"),
        error: None,
        resumable: false,
    };
    let resources = app.path().resource_dir().ok();
    spawn_job(
        state.jobs.clone(),
        resources,
        root,
        job,
        iterations.unwrap_or(30_000).clamp(1_000, 100_000),
        false,
        Arc::clone(&state.project),
    )
}

#[tauri::command]
pub fn artifact_job_status(project_path: String, job_id: String) -> Result<ArtifactJob, String> {
    validate_job_id(&job_id)?;
    let root = Path::new(&project_path);
    storage::read_project(root)?;
    read_job(root, &job_id)
}

#[tauri::command]
pub fn latest_artifact_job(project_path: String) -> Result<Option<ArtifactJob>, String> {
    let root = PathBuf::from(project_path);
    storage::read_project(&root)?;
    let jobs_root = root.join("outputs").join("jobs");
    if !jobs_root.is_dir() {
        return Ok(None);
    }
    let mut jobs = fs::read_dir(jobs_root)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|value| value == "json")
        })
        .filter_map(|entry| File::open(entry.path()).ok())
        .filter_map(|file| serde_json::from_reader::<_, ArtifactJob>(file).ok())
        .collect::<Vec<_>>();
    jobs.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
    Ok(jobs.into_iter().next())
}

#[tauri::command]
pub fn cancel_artifact_job(
    project_path: String,
    job_id: String,
    state: State<'_, AppState>,
) -> Result<ArtifactJob, String> {
    let root = PathBuf::from(project_path);
    validate_job_id(&job_id)?;
    storage::read_project(&root)?;
    let cancellation = state
        .jobs
        .cancellations
        .lock()
        .map_err(|_| "Job manager is unavailable".to_string())?
        .get(&job_id)
        .cloned()
        .ok_or_else(|| "The artifact job is not running".to_string())?;
    cancellation.store(true, Ordering::SeqCst);
    File::create(root.join("outputs").join("cancel.flag")).ok();
    let mut job = read_job(&root, &job_id)?;
    job.status = "cancelling".to_string();
    job.stage = "cancelling".to_string();
    job.updated_at = Utc::now().to_rfc3339();
    write_job(&root, &job)?;
    Ok(job)
}

fn discard_job_record(root: &Path, job_id: &str) -> Result<ArtifactJob, String> {
    let mut job = read_job(root, job_id)?;
    if !matches!(job.status.as_str(), "failed" | "cancelled") {
        return Err("Only a stopped artifact job can be discarded".to_string());
    }

    job.status = "cancelled".to_string();
    job.stage = "cancelled".to_string();
    job.detail = "Interrupted artifact job cancelled".to_string();
    job.error = None;
    job.resumable = false;
    job.updated_at = Utc::now().to_rfc3339();
    write_job(root, &job)?;

    let outputs = root.join("outputs");
    for transient in [
        "cancel.flag",
        "splat-checkpoint.pt",
        "splat-progress.json",
        "progress.json",
        "build-preview.json",
        "rgb-preview-status.json",
        "room-splat.preview.splat",
    ] {
        fs::remove_file(outputs.join(transient)).ok();
    }
    Ok(job)
}

#[tauri::command]
pub fn discard_artifact_job(
    project_path: String,
    job_id: String,
    state: State<'_, AppState>,
) -> Result<ArtifactJob, String> {
    let root = PathBuf::from(project_path);
    validate_job_id(&job_id)?;
    if state.jobs.is_running(&job_id) {
        return Err("Cancel the running artifact job before discarding its checkpoint".to_string());
    }

    let mut project = storage::read_project(&root)?;
    if project
        .active_job
        .as_deref()
        .is_some_and(|active| active != job_id)
    {
        return Err("A different artifact job is active for this project".to_string());
    }
    let job = discard_job_record(&root, &job_id)?;
    project.active_job = None;
    project.processing_status = "idle".to_string();
    project.processing_error = None;
    storage::write_project(&project)?;
    if let Ok(mut current) = state.project.lock() {
        if current.path == project.path {
            *current = project;
        }
    }
    Ok(job)
}

#[tauri::command]
pub fn resume_artifact_job(
    app: AppHandle,
    project_path: String,
    job_id: String,
    state: State<'_, AppState>,
) -> Result<ArtifactJob, String> {
    let root = PathBuf::from(project_path);
    validate_job_id(&job_id)?;
    if state
        .active_capture
        .lock()
        .map_err(|_| "Capture state is unavailable".to_string())?
        .is_some()
    {
        return Err("Stop RGB-D capture before resuming reconstruction".to_string());
    }
    if state
        .active_preview
        .lock()
        .map_err(|_| "Preview state is unavailable".to_string())?
        .is_some()
    {
        return Err("Stop the camera preview before resuming reconstruction".to_string());
    }
    if *state
        .active_photo_localization
        .lock()
        .map_err(|_| "Photo localization state is unavailable".to_string())?
    {
        return Err(
            "Wait for texture-photo localization to finish before reconstructing".to_string(),
        );
    }
    let mut job = read_job(&root, &job_id)?;
    if !matches!(job.status.as_str(), "failed" | "cancelled") || !job.resumable {
        return Err("This artifact job has no resumable checkpoint".to_string());
    }
    let project = storage::read_project(&root)?;
    let active_project_path = state
        .project
        .lock()
        .map_err(|_| "Project state is unavailable".to_string())?
        .path
        .clone();
    if active_project_path != project.path {
        return Err("The selected project is no longer active".to_string());
    }
    if project.active_job.is_some() || project.processing_status == "processing" {
        return Err("An artifact job is already running for this project".to_string());
    }
    let current_fingerprint = source_fingerprint(&root, &project);
    if current_fingerprint != job.source_fingerprint {
        return Err("Project source media changed; start a new reconstruction job".to_string());
    }
    if !splat_checkpoint_available(&root, &job) {
        job.resumable = false;
        job.updated_at = Utc::now().to_rfc3339();
        write_job(&root, &job)?;
        return Err("The Gaussian checkpoint is missing; start a new artifact job".to_string());
    }
    job.status = "queued".to_string();
    job.stage = "resuming".to_string();
    job.error = None;
    let iterations = job.total_iterations.unwrap_or(30_000);
    spawn_job(
        state.jobs.clone(),
        app.path().resource_dir().ok(),
        root,
        job,
        iterations,
        true,
        Arc::clone(&state.project),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(targets: &[&str], stage: &str) -> ArtifactJob {
        ArtifactJob {
            id: "job".to_string(),
            project_path: "project".to_string(),
            targets: targets.iter().map(|value| (*value).to_string()).collect(),
            source_kind: "rgbd".to_string(),
            media_restart: "reuse".to_string(),
            rebuild_rgbd: false,
            stage: stage.to_string(),
            detail: String::new(),
            progress: 0.0,
            iteration: None,
            total_iterations: None,
            loss: None,
            smoothed_loss: None,
            eta_seconds: None,
            stage_progress: None,
            stage_eta_seconds: None,
            elapsed_seconds: None,
            compute_backend: None,
            rgb_preview_active: false,
            rgb_preview_scale_status: None,
            rgb_preview_confidence: None,
            rgb_preview_drift_risk: None,
            rgb_preview_submap_count: None,
            rgb_preview_accepted_frames: None,
            rgb_preview_rejected_frames: None,
            status: "running".to_string(),
            created_at: String::new(),
            started_at: None,
            updated_at: String::new(),
            source_fingerprint: String::new(),
            log_path: String::new(),
            error: None,
            resumable: false,
        }
    }

    #[test]
    fn overall_progress_is_weighted_separately_from_mesh_stage_progress() {
        let job = job(&["pointCloud", "texturedMesh"], "Meshing");
        let overall = planned_progress(&job, Some(0.5)).expect("meshing belongs to the job plan");
        assert!((overall - 0.8276).abs() < 0.001);
        assert!((overall - 0.5).abs() > 0.3);
    }

    #[test]
    fn mesh_only_jobs_do_not_plan_a_gaussian_dataset_stage() {
        let mesh = job(&["pointCloud", "texturedMesh"], "Meshing");
        let splat = job(&["pointCloud", "gaussianSplat"], "Preparing splat data");
        assert!(!stage_plan(&mesh).iter().any(|(key, _)| *key == "dataset"));
        assert!(stage_plan(&splat).iter().any(|(key, _)| *key == "dataset"));
    }

    #[test]
    fn media_jobs_plan_camera_solving_without_rgbd_stages() {
        let mut media = job(&["gaussianSplat"], "feature_matching");
        media.source_kind = "media".to_string();
        let plan = stage_plan(&media);

        assert!(plan.iter().any(|(key, _)| *key == "media"));
        assert!(plan.iter().any(|(key, _)| *key == "splat"));
        assert!(!plan.iter().any(|(key, _)| *key == "track"));
        assert_eq!(stage_key(&media.stage), Some("media"));
    }

    #[test]
    fn desktop_preserves_progressive_rgb_preview_telemetry() {
        let root = std::env::temp_dir().join(format!("scanlan-rgb-preview-{}", Uuid::new_v4()));
        fs::create_dir_all(root.join("outputs")).unwrap();
        storage::write_json(
            &root.join("outputs").join("splat-progress.json"),
            &serde_json::json!({
                "stage": "rgb_preview_streaming",
                "progress": 0.15,
                "metrics": {"rgbPreview": {
                    "active": true,
                    "scaleStatus": "MODEL_METRIC_UNVERIFIED",
                    "confidence": 0.76,
                    "driftRisk": 0.18,
                    "residentSubmapCount": 3,
                    "acceptedFrameCount": 22,
                    "rejectedFrameCount": 2
                }}
            }),
        )
        .unwrap();
        let mut media = job(&["gaussianSplat"], "queued");
        media.source_kind = "media".to_string();
        merge_progress(&root, &mut media, true);
        assert!(media.rgb_preview_active);
        assert_eq!(
            media.rgb_preview_scale_status.as_deref(),
            Some("MODEL_METRIC_UNVERIFIED")
        );
        assert_eq!(media.rgb_preview_submap_count, Some(3));
        assert_eq!(media.rgb_preview_accepted_frames, Some(22));
        assert_eq!(media.rgb_preview_rejected_frames, Some(2));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn hybrid_jobs_plan_metric_anchors_media_and_all_output_builders() {
        let mut hybrid = job(
            &["pointCloud", "texturedMesh", "gaussianSplat"],
            "media_localization",
        );
        hybrid.source_kind = "hybrid".to_string();
        let plan = stage_plan(&hybrid);

        for expected in [
            "track",
            "trajectory",
            "cloud",
            "media",
            "dataset",
            "mesh",
            "splat",
        ] {
            assert!(plan.iter().any(|(key, _)| *key == expected));
        }
        assert_eq!(stage_key(&hybrid.stage), Some("media"));
    }

    #[test]
    fn desktop_preserves_an_unknown_worker_eta() {
        let root = std::env::temp_dir().join(format!(
            "scanlan-unknown-worker-eta-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(root.join("outputs").join("jobs")).unwrap();
        let mut media = job(&["gaussianSplat"], "media_decode");
        media.source_kind = "media".to_string();
        media.started_at = Some(Utc::now().to_rfc3339());
        media.eta_seconds = Some(4_260);
        fs::write(
            progress_file(&root, true),
            br#"{"stage":"media_decode","detail":"Scanning video","progress":0.03,"stageProgress":0.03,"etaSeconds":null,"stageEtaSeconds":null}"#,
        )
        .unwrap();

        merge_progress(&root, &mut media, true);

        assert_eq!(media.eta_seconds, None);
        assert_eq!(media.stage_eta_seconds, None);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn desktop_uses_a_measured_worker_eta() {
        let root = std::env::temp_dir().join(format!(
            "scanlan-measured-worker-eta-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(root.join("outputs").join("jobs")).unwrap();
        let mut training = job(&["gaussianSplat"], "splat_training");
        training.started_at = Some(Utc::now().to_rfc3339());
        fs::write(
            progress_file(&root, true),
            br#"{"stage":"splat_training","detail":"Training","progress":0.5,"stageProgress":0.5,"etaSeconds":37,"stageEtaSeconds":37}"#,
        )
        .unwrap();

        merge_progress(&root, &mut training, true);

        assert_eq!(training.eta_seconds, Some(37));
        assert_eq!(training.stage_eta_seconds, Some(37));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn accelerator_lock_rejects_a_second_owner_and_recovers_after_drop() {
        let path = std::env::temp_dir().join(format!(
            "scanlan-accelerator-test-{}.lock",
            uuid::Uuid::new_v4()
        ));
        let first = acquire_accelerator_lock_at(&path).expect("first owner should acquire lock");
        let error = acquire_accelerator_lock_at(&path)
            .expect_err("a second owner must not acquire the accelerator");
        assert!(error.contains("already using"), "unexpected error: {error}");
        drop(first);
        let second = acquire_accelerator_lock_at(&path)
            .expect("lock should become available when its owner exits");
        drop(second);
        fs::remove_file(path).ok();
    }

    #[test]
    fn media_analysis_restart_keeps_decoded_views_and_discards_downstream_work() {
        let root = std::env::temp_dir().join(format!(
            "scanlan-media-restart-test-{}",
            uuid::Uuid::new_v4()
        ));
        let cache = root.join("outputs").join("cache");
        fs::create_dir_all(cache.join("media-observations").join("media-test")).unwrap();
        fs::create_dir_all(cache.join("datasets").join("analysis-test")).unwrap();
        let manual = serde_json::json!({
            "id": "manual",
            "sourcePath": root.join("detail.jpg").to_string_lossy()
        });
        let imported = serde_json::json!({
            "id": "media",
            "sourcePath": cache
                .join("media-observations")
                .join("media-test")
                .join("images")
                .join("video.jpg")
                .to_string_lossy()
        });
        storage::write_json(
            &root.join("supplemental-photos.json"),
            &serde_json::json!({
                "schemaVersion": 1,
                "photos": [manual.clone(), imported.clone()],
                "attempts": [manual, imported]
            }),
        )
        .unwrap();
        let mut rebuild = job(&["gaussianSplat"], "queued");
        rebuild.source_kind = "hybrid".to_string();
        rebuild.media_restart = "analysis".to_string();

        invalidate_pipeline_cache(&root, &rebuild).unwrap();

        assert!(cache.join("media-observations").is_dir());
        assert!(!cache.join("datasets").exists());
        let manifest: Value =
            serde_json::from_reader(File::open(root.join("supplemental-photos.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["photos"].as_array().unwrap().len(), 1);
        assert_eq!(manifest["photos"][0]["id"], "manual");
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn media_decode_restart_discards_decoded_views_too() {
        let root = std::env::temp_dir().join(format!(
            "scanlan-media-decode-test-{}",
            uuid::Uuid::new_v4()
        ));
        let cache = root.join("outputs").join("cache");
        fs::create_dir_all(cache.join("media-observations")).unwrap();
        fs::create_dir_all(cache.join("datasets")).unwrap();
        let mut rebuild = job(&["gaussianSplat"], "queued");
        rebuild.source_kind = "media".to_string();
        rebuild.media_restart = "decode".to_string();

        invalidate_pipeline_cache(&root, &rebuild).unwrap();

        assert!(!cache.join("media-observations").exists());
        assert!(!cache.join("datasets").exists());
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn discarding_an_interrupted_job_removes_its_resume_state() {
        let root =
            std::env::temp_dir().join(format!("scanlan-discard-job-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(root.join("outputs").join("jobs")).unwrap();
        let mut interrupted = job(&["gaussianSplat"], "interrupted");
        interrupted.status = "failed".to_string();
        interrupted.resumable = true;
        write_job(&root, &interrupted).unwrap();
        for transient in [
            "splat-checkpoint.pt",
            "splat-progress.json",
            "build-preview.json",
        ] {
            fs::write(root.join("outputs").join(transient), b"partial").unwrap();
        }
        assert!(splat_checkpoint_available(&root, &interrupted));

        let discarded = discard_job_record(&root, &interrupted.id).unwrap();

        assert_eq!(discarded.status, "cancelled");
        assert!(!discarded.resumable);
        assert!(discarded.error.is_none());
        assert!(!root.join("outputs").join("splat-checkpoint.pt").exists());
        assert!(!root.join("outputs").join("splat-progress.json").exists());
        assert!(!root.join("outputs").join("build-preview.json").exists());
        assert!(!splat_checkpoint_available(&root, &discarded));
        fs::remove_dir_all(root).ok();
    }

    #[cfg(windows)]
    #[test]
    fn child_lifetime_guard_terminates_its_worker_when_dropped() {
        let mut child = Command::new("powershell.exe")
            .args(["-NoProfile", "-Command", "Start-Sleep -Seconds 30"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("test worker should start");
        let guard = ChildLifetimeGuard::attach(&child)
            .expect("test worker should attach to a kill-on-close Job Object");
        drop(guard);

        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if child
                .try_wait()
                .expect("test worker status should be readable")
                .is_some()
            {
                return;
            }
            thread::sleep(Duration::from_millis(20));
        }
        child.kill().ok();
        child.wait().ok();
        panic!("test worker survived after its lifetime guard closed");
    }
}
