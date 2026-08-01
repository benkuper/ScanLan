mod commands;
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
            commands::stop_sensor_phase,
            commands::remove_capture,
            commands::reconstruct_project,
            commands::load_preview,
            commands::load_preview_mesh_geometry,
            commands::load_preview_mesh_texture,
            commands::load_camera_frames,
            commands::apply_cloud_transform,
            commands::export_ply,
            commands::export_textured_mesh,
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
