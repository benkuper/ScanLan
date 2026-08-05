mod commands;
mod jobs;
mod models;
mod storage;

use commands::AppState;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::available_sensors,
            commands::runtime_info,
            commands::current_project,
            commands::create_project,
            commands::update_project_settings,
            commands::start_sensor_phase,
            commands::capture_status,
            commands::live_preview_frame,
            commands::live_reconstruction_mesh,
            commands::stop_sensor_phase,
            commands::remove_capture,
            commands::load_preview,
            commands::load_preview_mesh_geometry,
            commands::load_preview_mesh_texture,
            commands::load_gaussian_splat,
            commands::export_ply,
            commands::export_textured_mesh,
            commands::export_gaussian_splat,
            jobs::start_artifact_job,
            jobs::artifact_job_status,
            jobs::latest_artifact_job,
            jobs::cancel_artifact_job,
            jobs::discard_artifact_job,
            jobs::resume_artifact_job,
        ])
        .build(tauri::generate_context!())
        .expect("error while building ScanLan");

    app.run(|app_handle, event| {
        if matches!(
            event,
            tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit
        ) {
            commands::terminate_active_capture(app_handle.state::<AppState>().inner());
        }
    });
}
