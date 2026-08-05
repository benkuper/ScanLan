use crate::models::{PhaseManifest, ProjectSummary};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

pub type StorageResult<T> = Result<T, String>;

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> StorageResult<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source_wide: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination_wide: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let moved = unsafe {
        MoveFileExW(
            source_wide.as_ptr(),
            destination_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved == 0 {
        return Err(format!(
            "Could not publish {}: {}",
            destination.display(),
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> StorageResult<()> {
    fs::rename(source, destination).map_err(|error| error.to_string())
}

pub fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> StorageResult<()> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("No parent directory for {}", path.display()))?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let filename = path
        .file_name()
        .ok_or_else(|| format!("No file name for {}", path.display()))?
        .to_string_lossy();
    let temporary = parent.join(format!(".{filename}.{}.tmp", Uuid::new_v4()));
    let result = (|| {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| error.to_string())?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, value).map_err(|error| error.to_string())?;
        writer.flush().map_err(|error| error.to_string())?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| error.to_string())?;
        drop(writer);
        replace_file(&temporary, path)
    })();
    if result.is_err() {
        fs::remove_file(&temporary).ok();
    }
    result
}

pub fn read_project(path: &Path) -> StorageResult<ProjectSummary> {
    let file = File::open(path.join("project.json")).map_err(|error| error.to_string())?;
    let project: ProjectSummary =
        serde_json::from_reader(file).map_err(|error| error.to_string())?;
    if project.schema_version != 3 {
        return Err(format!(
            "Project schema {} is unsupported; ScanLan requires schema 3",
            project.schema_version
        ));
    }
    if project.path.is_empty() || Path::new(&project.path) != path {
        return Err("Project path does not match its manifest".to_string());
    }
    Ok(project)
}

pub fn latest_project(base: &Path) -> StorageResult<Option<ProjectSummary>> {
    if !base.exists() {
        return Ok(None);
    }

    let mut projects = Vec::new();
    for entry in fs::read_dir(base).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        if !entry
            .file_type()
            .map_err(|error| error.to_string())?
            .is_dir()
        {
            continue;
        }
        if let Ok(project) = read_project(&entry.path()) {
            projects.push(project);
        }
    }
    projects.sort_by(|left, right| right.created_at.cmp(&left.created_at));
    Ok(projects.into_iter().next())
}

pub fn recover_interrupted_phases(project: &mut ProjectSummary) -> StorageResult<bool> {
    let mut changed = false;
    let capture_fps = project.settings.capture_fps.max(1);
    for phase in &mut project.phases {
        if phase.status != "capturing" {
            continue;
        }

        let manifest_path = Path::new(&project.path)
            .join("phases")
            .join(&phase.id)
            .join("phase.json");
        if let Ok(file) = File::open(manifest_path) {
            if let Ok(manifest) = serde_json::from_reader::<_, PhaseManifest>(file) {
                phase.frame_count = manifest.frame_count;
                phase.duration_seconds = manifest.duration_seconds;
            }
        }
        let frames_path = Path::new(&project.path)
            .join("phases")
            .join(&phase.id)
            .join("frames.csv");
        if let Ok(frame_index) = fs::read_to_string(frames_path) {
            let indexed_frames = frame_index
                .lines()
                .skip(1)
                .filter(|line| !line.trim().is_empty())
                .count() as u32;
            phase.frame_count = phase.frame_count.max(indexed_frames);
            if phase.frame_count > 0 {
                phase.duration_seconds = phase
                    .duration_seconds
                    .max((phase.frame_count / capture_fps).max(1));
            }
        }
        phase.status = if phase.frame_count > 0 {
            "complete".to_string()
        } else {
            "failed".to_string()
        };
        phase.overlap_hint = "Recovered after an interrupted session".to_string();
        changed = true;
    }
    if changed {
        write_project(project)?;
    }
    Ok(changed)
}

pub fn write_project(project: &ProjectSummary) -> StorageResult<()> {
    write_json(
        Path::new(&project.path).join("project.json").as_path(),
        project,
    )
}

pub fn create_project(root: &Path) -> StorageResult<ProjectSummary> {
    fs::create_dir_all(root.join("phases")).map_err(|error| error.to_string())?;
    fs::create_dir_all(root.join("outputs")).map_err(|error| error.to_string())?;
    fs::create_dir_all(root.join("outputs").join("jobs")).map_err(|error| error.to_string())?;
    let project = ProjectSummary::at_path(root);
    write_project(&project)?;
    Ok(project)
}

pub fn candidate_splat_worker_paths(resource_root: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let packaged_binary = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("splat-worker")
        .join("dist")
        .join("scanlan-splat")
        .join("scanlan-splat.exe");
    let development_binary = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("splat-worker")
        .join(".venv")
        .join("Scripts")
        .join("scanlan-splat.exe");
    if cfg!(debug_assertions) {
        candidates.push(packaged_binary.clone());
        candidates.push(development_binary.clone());
    }
    if let Some(root) = resource_root {
        candidates.push(root.join("splat-runtime").join("scanlan-splat.exe"));
        candidates.push(
            root.join("splat-runtime")
                .join("Scripts")
                .join("scanlan-splat.exe"),
        );
        candidates.push(root.join("scanlan-splat.exe"));
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            candidates.push(parent.join("splat-runtime").join("scanlan-splat.exe"));
            candidates.push(
                parent
                    .join("splat-runtime")
                    .join("Scripts")
                    .join("scanlan-splat.exe"),
            );
            candidates.push(
                parent
                    .join("resources")
                    .join("splat-runtime")
                    .join("scanlan-splat.exe"),
            );
            candidates.push(
                parent
                    .join("resources")
                    .join("splat-runtime")
                    .join("Scripts")
                    .join("scanlan-splat.exe"),
            );
        }
    }
    if !cfg!(debug_assertions) {
        candidates.push(packaged_binary);
        candidates.push(development_binary);
    }
    candidates
}

pub fn candidate_reconstruction_worker_paths(resource_root: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let development_binary = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("worker")
        .join("dist")
        .join("scanlan-worker.exe");
    if cfg!(debug_assertions) {
        candidates.push(development_binary.clone());
    }
    if let Some(root) = resource_root {
        candidates.push(root.join("scanlan-worker.exe"));
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            candidates.push(parent.join("scanlan-worker.exe"));
            candidates.push(parent.join("resources").join("scanlan-worker.exe"));
        }
    }
    if !cfg!(debug_assertions) {
        candidates.push(development_binary);
    }
    candidates
}

pub fn candidate_kinect_worker_paths(resource_root: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let development_binary = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("build")
        .join("kinect-capture")
        .join("Release")
        .join("kinect2-capture-worker.exe");
    if cfg!(debug_assertions) {
        candidates.push(development_binary.clone());
    }
    if let Some(root) = resource_root {
        candidates.push(root.join("kinect2-capture-worker.exe"));
        candidates.push(root.join("kinect2").join("kinect2-capture-worker.exe"));
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            candidates.push(parent.join("kinect2-capture-worker.exe"));
            candidates.push(parent.join("resources").join("kinect2-capture-worker.exe"));
            candidates.push(
                parent
                    .join("resources")
                    .join("kinect2")
                    .join("kinect2-capture-worker.exe"),
            );
        }
    }
    if !cfg!(debug_assertions) {
        candidates.push(development_binary);
    }
    candidates
}

pub fn candidate_modern_sensor_worker_paths(resource_root: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let development_binary = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("build")
        .join("modern-capture")
        .join("Release")
        .join("rgbd-capture-worker.exe");
    if cfg!(debug_assertions) {
        candidates.push(development_binary.clone());
    }
    if let Some(root) = resource_root {
        candidates.push(root.join("rgbd-capture-worker.exe"));
        candidates.push(root.join("modern").join("rgbd-capture-worker.exe"));
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            candidates.push(parent.join("rgbd-capture-worker.exe"));
            candidates.push(parent.join("resources").join("rgbd-capture-worker.exe"));
            candidates.push(
                parent
                    .join("resources")
                    .join("modern")
                    .join("rgbd-capture-worker.exe"),
            );
        }
    }
    if !cfg!(debug_assertions) {
        candidates.push(development_binary);
    }
    candidates
}
