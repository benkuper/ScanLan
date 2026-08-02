use crate::models::{ArtifactSummary, PhaseManifest, ProjectSummary};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

pub type StorageResult<T> = Result<T, String>;

pub fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> StorageResult<()> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("No parent directory for {}", path.display()))?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let temporary = path.with_extension("json.tmp");
    let file = File::create(&temporary).map_err(|error| error.to_string())?;
    serde_json::to_writer_pretty(BufWriter::new(file), value).map_err(|error| error.to_string())?;
    if path.exists() {
        fs::remove_file(path).map_err(|error| error.to_string())?;
    }
    fs::rename(&temporary, path).map_err(|error| error.to_string())?;
    Ok(())
}

pub fn read_project(path: &Path) -> StorageResult<ProjectSummary> {
    let file = File::open(path.join("project.json")).map_err(|error| error.to_string())?;
    let mut project: ProjectSummary =
        serde_json::from_reader(file).map_err(|error| error.to_string())?;
    let mut changed = false;
    if project.schema_version < 2 {
        project.schema_version = 2;
        changed = true;
    }
    if project.path.is_empty() || Path::new(&project.path) != path {
        project.path = path.to_string_lossy().into_owned();
        changed = true;
    }
    if project.artifacts.point_cloud.is_none() {
        if let Some(output_path) = &project.output_path {
            project.artifacts.point_cloud = Some(ArtifactSummary {
                path: output_path.clone(),
                status: if project.processing_status == "complete" {
                    "ready".to_string()
                } else {
                    "stale".to_string()
                },
                source_fingerprint: String::new(),
                updated_at: project.created_at.clone(),
                metric: true,
                stale: project.processing_status != "complete",
            });
            changed = true;
        }
    }
    if project.artifacts.textured_mesh.is_none() {
        if let Some(mesh_path) = &project.mesh_output_path {
            project.artifacts.textured_mesh = Some(ArtifactSummary {
                path: mesh_path.clone(),
                status: if project.processing_status == "complete" {
                    "ready".to_string()
                } else {
                    "stale".to_string()
                },
                source_fingerprint: String::new(),
                updated_at: project.created_at.clone(),
                metric: true,
                stale: project.processing_status != "complete",
            });
            changed = true;
        }
    }
    if changed {
        write_json(path.join("project.json").as_path(), &project)?;
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
    fs::create_dir_all(root.join("sources")).map_err(|error| error.to_string())?;
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
        candidates.push(
            root.join("splat-runtime")
                .join("Scripts")
                .join("scanlan-splat.exe"),
        );
        candidates.push(root.join("scanlan-splat.exe"));
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
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
        .join("legacy-capture-worker.exe");
    if cfg!(debug_assertions) {
        candidates.push(development_binary.clone());
    }
    if let Some(root) = resource_root {
        candidates.push(root.join("legacy-capture-worker.exe"));
        candidates.push(root.join("legacy").join("legacy-capture-worker.exe"));
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            candidates.push(parent.join("legacy-capture-worker.exe"));
            candidates.push(parent.join("resources").join("legacy-capture-worker.exe"));
            candidates.push(
                parent
                    .join("resources")
                    .join("legacy")
                    .join("legacy-capture-worker.exe"),
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

#[cfg(test)]
mod tests {
    use super::read_project;
    use std::fs;

    #[test]
    fn schema_one_project_migrates_additively() {
        let root = std::env::temp_dir().join(format!("scanlan-migrate-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("project.json"),
            format!(
                r#"{{
                  "schemaVersion": 1,
                  "id": "legacy",
                  "name": "Legacy",
                  "path": "{}",
                  "createdAt": "2026-01-01T00:00:00Z",
                  "phases": [],
                  "settings": {{
                    "captureFps": 10,
                    "maxDepthM": 4.2,
                    "voxelSizeMm": 15,
                    "environment": "indoor"
                  }},
                  "processingStatus": "complete",
                  "pointCount": 42,
                  "outputPath": "outputs/room-cloud.ply"
                }}"#,
                root.to_string_lossy().replace('\\', "\\\\")
            ),
        )
        .unwrap();
        let project = read_project(&root).unwrap();
        assert_eq!(project.schema_version, 2);
        assert!(project.media_sources.is_empty());
        assert!(project.active_job.is_none());
        assert_eq!(
            project
                .artifacts
                .point_cloud
                .as_ref()
                .map(|value| value.path.as_str()),
            Some("outputs/room-cloud.ply")
        );
        fs::remove_dir_all(root).ok();
    }
}
