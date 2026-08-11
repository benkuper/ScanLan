#include <Kinect.h>
#include <NuiKinectFusionApi.h>
#include <Windows.h>
#include <fcntl.h>
#include <io.h>
#include "rgbd_archive.h"
#include "rgbd_stream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

constexpr int DepthWidth = 512;
constexpr int DepthHeight = 424;
constexpr int ColorWidth = 1920;
constexpr int ColorHeight = 1080;

template <class T>
void safe_release(T*& value) {
    if (value != nullptr) {
        value->Release();
        value = nullptr;
    }
}

struct Options {
    bool capabilities = false;
    bool probe = false;
    bool stream_rgbd = false;
    fs::path phase_root;
    std::string id;
    std::string name = "Kinect capture";
    int fps = 10;
    float max_depth_m = 4.2F;
};

std::string json_escape(const std::string& input) {
    std::ostringstream output;
    for (const char character : input) {
        switch (character) {
        case '\\': output << "\\\\"; break;
        case '"': output << "\\\""; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default: output << character; break;
        }
    }
    return output.str();
}

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_s(&utc, &time);
    std::ostringstream result;
    result << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return result.str();
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&]() -> std::string {
            if (index + 1 >= argc) throw std::runtime_error("Missing value after " + argument);
            return argv[++index];
        };
        if (argument == "--capabilities") options.capabilities = true;
        else if (argument == "--probe") options.probe = true;
        else if (argument == "--phase") options.phase_root = fs::path(value());
        else if (argument == "--stream-rgbd") options.stream_rgbd = true;
        else if (argument == "--id") options.id = value();
        else if (argument == "--name") options.name = value();
        else if (argument == "--fps") options.fps = std::stoi(value());
        else if (argument == "--max-depth") options.max_depth_m = std::stof(value());
        else if (argument == "--help") {
            std::cout << "kinect2-capture-worker --capabilities\n"
                         "kinect2-capture-worker --probe\n"
                         "kinect2-capture-worker --phase PATH --id ID [--name NAME] [--fps 1..30] [--max-depth METERS] [--stream-rgbd]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }
    if (!options.capabilities && !options.probe && (options.phase_root.empty() || options.id.empty())) {
        throw std::runtime_error("--phase and --id are required");
    }
    options.fps = std::clamp(options.fps, 1, 30);
    options.max_depth_m = std::clamp(options.max_depth_m, 0.5F, 4.5F);
    return options;
}

void set_identity(Matrix4& matrix) {
    matrix = {};
    matrix.M11 = matrix.M22 = matrix.M33 = matrix.M44 = 1.0F;
}

Matrix4 invert_pose(const Matrix4& transform) {
    Matrix4 inverse{};
    inverse.M11 = transform.M11; inverse.M12 = transform.M21; inverse.M13 = transform.M31;
    inverse.M21 = transform.M12; inverse.M22 = transform.M22; inverse.M23 = transform.M32;
    inverse.M31 = transform.M13; inverse.M32 = transform.M23; inverse.M33 = transform.M33;
    inverse.M41 = -(transform.M41 * inverse.M11 + transform.M42 * inverse.M21 + transform.M43 * inverse.M31);
    inverse.M42 = -(transform.M41 * inverse.M12 + transform.M42 * inverse.M22 + transform.M43 * inverse.M32);
    inverse.M43 = -(transform.M41 * inverse.M13 + transform.M42 * inverse.M23 + transform.M43 * inverse.M33);
    inverse.M44 = 1.0F;
    return inverse;
}

class FusionTracker {
public:
    ~FusionTracker() {
        if (depth_float_ != nullptr) NuiFusionReleaseImageFrame(depth_float_);
        safe_release(volume_);
    }

    void initialize(const CameraIntrinsics& intrinsics, float max_depth_m) {
        max_depth_m_ = max_depth_m;
        NUI_FUSION_RECONSTRUCTION_PARAMETERS reconstruction{};
        reconstruction.voxelsPerMeter = 64;
        reconstruction.voxelCountX = 256;
        reconstruction.voxelCountY = 256;
        reconstruction.voxelCountZ = 384;
        set_identity(world_to_camera_);
        HRESULT result = NuiFusionCreateReconstruction(
            &reconstruction,
            NUI_FUSION_RECONSTRUCTION_PROCESSOR_TYPE_AMP,
            -1,
            &world_to_camera_,
            &volume_);
        if (FAILED(result) || volume_ == nullptr) {
            status_ = "Kinect Fusion GPU tracking unavailable; offline tracking will be used";
            return;
        }
        camera_.focalLengthX = intrinsics.FocalLengthX / static_cast<float>(DepthWidth);
        camera_.focalLengthY = intrinsics.FocalLengthY / static_cast<float>(DepthHeight);
        camera_.principalPointX = intrinsics.PrincipalPointX / static_cast<float>(DepthWidth);
        camera_.principalPointY = intrinsics.PrincipalPointY / static_cast<float>(DepthHeight);
        result = NuiFusionCreateImageFrame(
            NUI_FUSION_IMAGE_TYPE_FLOAT,
            DepthWidth,
            DepthHeight,
            &camera_,
            &depth_float_);
        if (FAILED(result) || depth_float_ == nullptr) {
            safe_release(volume_);
            status_ = "Kinect Fusion tracking image could not be created; offline tracking will be used";
            return;
        }
        available_ = true;
        status_ = "Kinect Fusion tracking locked";
    }

    bool track(const std::vector<std::uint16_t>& depth, Matrix4& camera_to_world) {
        if (!available_) return false;
        HRESULT result = volume_->DepthToDepthFloatFrame(
            const_cast<UINT16*>(depth.data()),
            static_cast<UINT>(depth.size() * sizeof(std::uint16_t)),
            depth_float_,
            NUI_FUSION_DEFAULT_MINIMUM_DEPTH,
            max_depth_m_,
            false);
        if (FAILED(result)) {
            status_ = "Kinect Fusion depth conversion failed";
            return false;
        }
        result = volume_->ProcessFrame(
            depth_float_,
            NUI_FUSION_DEFAULT_ALIGN_ITERATION_COUNT,
            NUI_FUSION_DEFAULT_INTEGRATION_WEIGHT,
            nullptr,
            &world_to_camera_);
        if (FAILED(result)) {
            status_ = result == E_NUI_FUSION_TRACKING_ERROR
                ? "Tracking lost - return to the last scanned area"
                : "Kinect Fusion could not process this frame";
            return false;
        }
        if (SUCCEEDED(volume_->GetCurrentWorldToCameraTransform(&world_to_camera_))) {
            camera_to_world = invert_pose(world_to_camera_);
            status_ = "Kinect Fusion tracking locked";
            return true;
        }
        status_ = "Kinect Fusion pose is unavailable";
        return false;
    }

    bool available() const { return available_; }
    const std::string& status() const { return status_; }

private:
    INuiFusionReconstruction* volume_ = nullptr;
    NUI_FUSION_IMAGE_FRAME* depth_float_ = nullptr;
    NUI_FUSION_CAMERA_PARAMETERS camera_{};
    Matrix4 world_to_camera_{};
    float max_depth_m_ = 4.2F;
    bool available_ = false;
    std::string status_ = "Initializing Kinect Fusion tracking";
};

void wait_for_sensor(IKinectSensor* sensor) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        BOOLEAN available = FALSE;
        if (SUCCEEDED(sensor->get_IsAvailable(&available)) && available) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("Kinect v2 opened but did not become available");
}

CameraIntrinsics wait_for_intrinsics(ICoordinateMapper* mapper) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        CameraIntrinsics intrinsics{};
        const HRESULT result = mapper->GetDepthCameraIntrinsics(&intrinsics);
        if (SUCCEEDED(result)
            && intrinsics.FocalLengthX > 0.0F
            && intrinsics.FocalLengthY > 0.0F) {
            return intrinsics;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("Could not read valid depth-camera intrinsics");
}

int probe_sensor() {
    IKinectSensor* sensor = nullptr;
    IMultiSourceFrameReader* reader = nullptr;
    ICoordinateMapper* mapper = nullptr;
    HRESULT result = GetDefaultKinectSensor(&sensor);
    if (FAILED(result) || sensor == nullptr) throw std::runtime_error("Kinect v2 sensor was not found");

    result = sensor->Open();
    if (FAILED(result)) throw std::runtime_error("Kinect v2 sensor could not be opened");
    wait_for_sensor(sensor);
    result = sensor->get_CoordinateMapper(&mapper);
    if (FAILED(result) || mapper == nullptr) throw std::runtime_error("Coordinate mapper is unavailable");
    result = sensor->OpenMultiSourceFrameReader(FrameSourceTypes_Depth | FrameSourceTypes_Color, &reader);
    if (FAILED(result) || reader == nullptr) throw std::runtime_error("Could not open synchronized depth/color reader");

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        IMultiSourceFrame* multi_frame = nullptr;
        result = reader->AcquireLatestFrame(&multi_frame);
        if (result == E_PENDING || multi_frame == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        if (FAILED(result)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }

        IDepthFrameReference* depth_reference = nullptr;
        IColorFrameReference* color_reference = nullptr;
        IDepthFrame* depth_frame = nullptr;
        IColorFrame* color_frame = nullptr;
        multi_frame->get_DepthFrameReference(&depth_reference);
        multi_frame->get_ColorFrameReference(&color_reference);
        if (depth_reference != nullptr) depth_reference->AcquireFrame(&depth_frame);
        if (color_reference != nullptr) color_reference->AcquireFrame(&color_frame);
        const bool ready = depth_frame != nullptr && color_frame != nullptr;

        safe_release(color_frame);
        safe_release(depth_frame);
        safe_release(color_reference);
        safe_release(depth_reference);
        safe_release(multi_frame);

        if (ready) {
            wait_for_intrinsics(mapper);
            safe_release(reader);
            safe_release(mapper);
            sensor->Close();
            safe_release(sensor);
            std::cout << "Kinect v2 connected" << std::endl;
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    safe_release(reader);
    safe_release(mapper);
    sensor->Close();
    safe_release(sensor);
    throw std::runtime_error("Kinect v2 opened but did not deliver synchronized depth/color frames");
}

void write_phase_manifest(
    const Options& options,
    const CameraIntrinsics& intrinsics,
    const std::string& created_at,
    std::uint32_t frame_count,
    std::uint32_t duration_seconds,
    const std::string& pose_source
) {
    const fs::path temporary = options.phase_root / "phase.json.tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) throw std::runtime_error("Could not create phase.json");
    output << std::fixed << std::setprecision(8);
    output << "{\n"
           << "  \"schemaVersion\": 3,\n"
           << "  \"id\": \"" << json_escape(options.id) << "\",\n"
           << "  \"name\": \"" << json_escape(options.name) << "\",\n"
           << "  \"createdAt\": \"" << created_at << "\",\n"
           << "  \"frameCount\": " << frame_count << ",\n"
           << "  \"durationSeconds\": " << duration_seconds << ",\n"
           << "  \"frameFormat\": \"depth=u16le,color=rgb8,aligned=true\",\n"
           << "  \"poseSource\": \"" << json_escape(pose_source) << "\",\n"
           << "  \"sensor\": {\"kind\": \"kinect_v2\", \"name\": \"Kinect v2\", "
           << "\"connection\": \"usb\", \"serial\": \"\", \"address\": \"\"},\n"
           << "  \"camera\": {\n"
           << "    \"width\": " << DepthWidth << ",\n"
           << "    \"height\": " << DepthHeight << ",\n"
           << "    \"fx\": " << intrinsics.FocalLengthX << ",\n"
           << "    \"fy\": " << intrinsics.FocalLengthY << ",\n"
           << "    \"cx\": " << intrinsics.PrincipalPointX << ",\n"
           << "    \"cy\": " << intrinsics.PrincipalPointY << ",\n"
           << "    \"depth_scale\": 1000.0,\n"
           << "    \"max_depth_m\": " << options.max_depth_m << ",\n"
           << "    \"radial_distortion\": ["
           << intrinsics.RadialDistortionSecondOrder << ", "
           << intrinsics.RadialDistortionFourthOrder << ", "
           << intrinsics.RadialDistortionSixthOrder << "]\n"
           << "  },\n"
           << "  \"rgbCamera\": {\n"
           << "    \"width\": " << DepthWidth << ", \"height\": " << DepthHeight << ",\n"
           << "    \"fx\": " << intrinsics.FocalLengthX << ", \"fy\": " << intrinsics.FocalLengthY << ",\n"
           << "    \"cx\": " << intrinsics.PrincipalPointX << ", \"cy\": " << intrinsics.PrincipalPointY << ",\n"
           << "    \"model\": \"pinhole\", \"distortion\": []\n"
           << "  },\n"
           << "  \"rgbFromDepth\": [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],\n"
           << "  \"sourceRgb\": {\"format\": \"aligned-rgb8\", \"quality\": 100, "
           << "\"nativeResolution\": false, \"droppedFrames\": 0}\n"
           << "}\n";
    output.flush();
    if (!output) throw std::runtime_error("Could not flush phase.json");
    output.close();
    if (!output) throw std::runtime_error("Could not close phase.json");
    const fs::path destination = options.phase_root / "phase.json";
    for (int attempt = 0; attempt < 40; ++attempt) {
        if (MoveFileExW(
                temporary.c_str(),
                destination.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5 + attempt));
    }
    throw std::runtime_error(
        "Could not publish phase.json (Windows error " + std::to_string(GetLastError()) + ")");
}

void write_text_atomic(const fs::path& path, const std::string& value) {
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) throw std::runtime_error("Could not write " + path.string());
    output << value;
    output.flush();
    if (!output) throw std::runtime_error("Could not flush " + temporary.string());
    output.close();
    if (!output) throw std::runtime_error("Could not close " + temporary.string());
    for (int attempt = 0; attempt < 40; ++attempt) {
        if (MoveFileExW(
                temporary.c_str(),
                path.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5 + attempt));
    }
    throw std::runtime_error(
        "Could not publish " + path.string() + " (Windows error "
        + std::to_string(GetLastError()) + ")");
}

std::array<float, 16> stream_pose(const Matrix4& pose) {
    // Kinect Fusion's Matrix4 stores translation in M41..M43 and uses row
    // vectors. The engine protocol standardizes on a column-vector
    // worldFromCamera matrix.
    return {
        pose.M11, pose.M21, pose.M31, pose.M41,
        pose.M12, pose.M22, pose.M32, pose.M42,
        pose.M13, pose.M23, pose.M33, pose.M43,
        0, 0, 0, 1,
    };
}

void write_live_status(
    const fs::path& root,
    std::uint64_t timestamp_us,
    std::uint32_t frame_count,
    double stream_fps,
    bool tracking,
    const std::string& tracking_status
) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(2)
           << "{\n"
           << "  \"connected\": true,\n"
           << "  \"timestampUs\": " << timestamp_us << ",\n"
           << "  \"frameCount\": " << frame_count << ",\n"
           << "  \"streamFps\": " << stream_fps << ",\n"
           << "  \"tracking\": " << (tracking ? "true" : "false") << ",\n"
           << "  \"trackingStatus\": \"" << json_escape(tracking_status) << "\"\n"
           << "}\n";
    write_text_atomic(root / "live.json", output.str());
}

int capture(const Options& options) {
    constexpr int depth_count = DepthWidth * DepthHeight;
    constexpr int color_count = ColorWidth * ColorHeight;
    fs::create_directories(options.phase_root / "depth");
    fs::create_directories(options.phase_root / "color");
    fs::create_directories(options.phase_root / "rgb");
    fs::remove(options.phase_root / "stop.flag");
    fs::remove(options.phase_root / "tracking-hold.flag");

    IKinectSensor* sensor = nullptr;
    IMultiSourceFrameReader* reader = nullptr;
    ICoordinateMapper* mapper = nullptr;
    HRESULT result = GetDefaultKinectSensor(&sensor);
    if (FAILED(result) || sensor == nullptr) throw std::runtime_error("Kinect v2 sensor was not found");
    result = sensor->Open();
    if (FAILED(result)) throw std::runtime_error("Kinect v2 sensor could not be opened");
    wait_for_sensor(sensor);
    result = sensor->get_CoordinateMapper(&mapper);
    if (FAILED(result)) throw std::runtime_error("Coordinate mapper is unavailable");
    result = sensor->OpenMultiSourceFrameReader(
        FrameSourceTypes_Depth | FrameSourceTypes_Color,
        &reader);
    if (FAILED(result)) throw std::runtime_error("Could not open synchronized depth/color reader");

    const CameraIntrinsics intrinsics = wait_for_intrinsics(mapper);
    FusionTracker tracker;
    tracker.initialize(intrinsics, options.max_depth_m);
    const std::string pose_source = tracker.available() ? "kinect_fusion" : "estimated_offline";

    const std::string created_at = utc_now();
    write_phase_manifest(options, intrinsics, created_at, 0, 0, pose_source);
    std::vector<std::uint16_t> depth(depth_count);
    std::vector<std::uint8_t> color_bgra(color_count * 4);
    scanlan::RgbdArchiveWriter archive(options.phase_root, 100, 0, 8);
    std::vector<std::uint8_t> aligned_rgb(depth_count * 3);
    std::vector<ColorSpacePoint> depth_to_color(depth_count);
    std::unique_ptr<scanlan::RgbdStreamWriter> rgbd_stream;
    if (options.stream_rgbd) {
        rgbd_stream = std::make_unique<scanlan::RgbdStreamWriter>(std::cout, 3);
    }
    std::uint64_t sensor_frames = 0;
    const auto started = std::chrono::steady_clock::now();
    auto last_synchronized_frame = started;

    while (!fs::exists(options.phase_root / "stop.flag")) {
        IMultiSourceFrame* multi_frame = nullptr;
        result = reader->AcquireLatestFrame(&multi_frame);
        if (result == E_PENDING) {
            if (std::chrono::steady_clock::now() - last_synchronized_frame > std::chrono::seconds(3)) {
                throw std::runtime_error("Kinect v2 stopped delivering frames; check its USB and power connections");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
            continue;
        }
        if (FAILED(result) || multi_frame == nullptr) {
            if (std::chrono::steady_clock::now() - last_synchronized_frame > std::chrono::seconds(3)) {
                throw std::runtime_error("Kinect v2 disconnected or stopped streaming");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        ++sensor_frames;
        const bool save_source = !fs::exists(options.phase_root / "tracking-hold.flag")
            && scanlan::archive_frame_due(sensor_frames, 30, options.fps);

        IDepthFrameReference* depth_reference = nullptr;
        IColorFrameReference* color_reference = nullptr;
        IDepthFrame* depth_frame = nullptr;
        IColorFrame* color_frame = nullptr;
        multi_frame->get_DepthFrameReference(&depth_reference);
        multi_frame->get_ColorFrameReference(&color_reference);
        if (depth_reference != nullptr) depth_reference->AcquireFrame(&depth_frame);
        if (color_reference != nullptr) color_reference->AcquireFrame(&color_frame);

        if (depth_frame != nullptr && color_frame != nullptr) {
            last_synchronized_frame = std::chrono::steady_clock::now();
            UINT depth_buffer_size = 0;
            UINT16* depth_buffer = nullptr;
            INT64 timestamp_100ns = 0;
            depth_frame->get_RelativeTime(&timestamp_100ns);
            if (SUCCEEDED(depth_frame->AccessUnderlyingBuffer(&depth_buffer_size, &depth_buffer))
                && depth_buffer_size == depth_count) {
                UINT16 minimum_depth = 0;
                UINT16 maximum_depth = 0;
                depth_frame->get_DepthMinReliableDistance(&minimum_depth);
                depth_frame->get_DepthMaxReliableDistance(&maximum_depth);
                const auto configured_maximum = static_cast<UINT16>(options.max_depth_m * 1000.0F);
                maximum_depth = std::min(maximum_depth, configured_maximum);
                for (int pixel = 0; pixel < depth_count; ++pixel) {
                    const UINT16 value = depth_buffer[pixel];
                    depth[pixel] = value >= minimum_depth && value <= maximum_depth ? value : 0;
                }

                if (SUCCEEDED(color_frame->CopyConvertedFrameDataToArray(
                    static_cast<UINT>(color_bgra.size()),
                    color_bgra.data(),
                    ColorImageFormat_Bgra))) {
                    result = mapper->MapDepthFrameToColorSpace(
                        depth_count,
                        depth.data(),
                        depth_count,
                        depth_to_color.data()
                    );
                    if (SUCCEEDED(result)) {
                        std::fill(aligned_rgb.begin(), aligned_rgb.end(), std::uint8_t{0});
                        for (int pixel = 0; pixel < depth_count; ++pixel) {
                            const ColorSpacePoint mapped = depth_to_color[pixel];
                            if (!std::isfinite(mapped.X) || !std::isfinite(mapped.Y)) continue;
                            const int color_x = static_cast<int>(std::lround(mapped.X));
                            const int color_y = static_cast<int>(std::lround(mapped.Y));
                            if (color_x < 0 || color_x >= ColorWidth || color_y < 0 || color_y >= ColorHeight) continue;
                            const std::size_t source = static_cast<std::size_t>((color_y * ColorWidth + color_x) * 4);
                            const std::size_t target = static_cast<std::size_t>(pixel * 3);
                            aligned_rgb[target] = color_bgra[source + 2];
                            aligned_rgb[target + 1] = color_bgra[source + 1];
                            aligned_rgb[target + 2] = color_bgra[source];
                        }

                        Matrix4 camera_to_world{};
                        const bool tracking = tracker.track(depth, camera_to_world);
                        INT64 rgb_timestamp_100ns = timestamp_100ns;
                        color_frame->get_RelativeTime(&rgb_timestamp_100ns);
                        if (rgbd_stream) {
                            const auto pose = tracking ? stream_pose(camera_to_world) : scanlan::identity_pose();
                            rgbd_stream->publish(
                                sensor_frames - 1,
                                static_cast<std::uint64_t>(timestamp_100ns / 10),
                                static_cast<std::uint64_t>(rgb_timestamp_100ns / 10),
                                DepthWidth,
                                DepthHeight,
                                intrinsics.FocalLengthX,
                                intrinsics.FocalLengthY,
                                intrinsics.PrincipalPointX,
                                intrinsics.PrincipalPointY,
                                1000.0F,
                                0.5F,
                                options.max_depth_m,
                                depth,
                                aligned_rgb,
                                nullptr,
                                tracking ? &pose : nullptr,
                                true);
                        }
                        if (save_source) {
                            scanlan::RgbdArchiveFrame archive_frame;
                            archive_frame.source_sequence = sensor_frames - 1;
                            archive_frame.depth_timestamp_us =
                                static_cast<std::uint64_t>(timestamp_100ns / 10);
                            archive_frame.color_timestamp_us =
                                static_cast<std::uint64_t>(rgb_timestamp_100ns / 10);
                            archive_frame.depth = depth;
                            archive_frame.aligned_color = aligned_rgb;
                            if(tracking) archive_frame.camera_to_world = stream_pose(camera_to_world);
                            archive.submit(std::move(archive_frame));
                        }
                        const auto elapsed_seconds = std::max(
                            0.001,
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
                        if(sensor_frames == 1 || sensor_frames % 3 == 0) {
                            write_live_status(
                                options.phase_root,
                                static_cast<std::uint64_t>(timestamp_100ns / 10),
                                archive.saved(),
                                sensor_frames / elapsed_seconds,
                                tracking,
                                tracker.status());
                        }
                    }
                }
            }
        }

        safe_release(color_frame);
        safe_release(depth_frame);
        safe_release(color_reference);
        safe_release(depth_reference);
        safe_release(multi_frame);
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - started
    );
    if (rgbd_stream) rgbd_stream->close();
    archive.close();
    const auto saved_frames = archive.saved();
    write_phase_manifest(
        options,
        intrinsics,
        created_at,
        saved_frames,
        static_cast<std::uint32_t>(std::max<std::int64_t>(1, elapsed.count())),
        pose_source
    );

    safe_release(reader);
    safe_release(mapper);
    sensor->Close();
    safe_release(sensor);
    return saved_frames > 0 ? 0 : 2;
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.capabilities) {
            std::cout << "[\"kinect_v2\"]\n";
            return 0;
        }
        if (options.stream_rgbd) _setmode(_fileno(stdout), _O_BINARY);
        return options.probe ? probe_sensor() : capture(options);
    } catch (const std::exception& error) {
        std::cerr << "kinect2-capture-worker: " << error.what() << '\n';
        return 1;
    }
}
