#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <io.h>
#include "rgbd_archive.h"
#include "gyro_integrator.h"
#include "rgbd_stream.h"

#ifdef SCANLAN_HAS_AZURE_KINECT
#include <k4a/k4a.h>
#endif

#ifdef SCANLAN_HAS_ORBBEC
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/hpp/Utils.hpp>
#endif

namespace fs = std::filesystem;

struct Options {
    bool capabilities = false;
    bool probe = false;
    bool list = false;
    bool stream_rgbd = false;
    bool use_imu = false;
    bool preview = false;
    fs::path root;
    std::string id = "phase";
    std::string name = "RGB-D phase";
    std::string sensor = "azure_kinect";
    std::string device_id;
    std::string connection = "usb";
    std::string address;
    std::string depth_fov = "narrow";
    bool depth_binned = false;
    int fps = 10;
    float max_depth_m = 4.5F;
    int rgb_quality = 92;
    std::uint32_t max_rgb_dimension = 0;
};

struct DepthModeInfo {
    int width;
    int height;
    int native_fps;
};

DepthModeInfo requested_depth_mode(const Options &options) {
    if(options.depth_fov == "wide") {
        return options.depth_binned ? DepthModeInfo{512, 512, 30} : DepthModeInfo{1024, 1024, 15};
    }
    return options.depth_binned ? DepthModeInfo{320, 288, 30} : DepthModeInfo{640, 576, 30};
}

struct CameraInfo {
    int width = 0;
    int height = 0;
    float fx = 0;
    float fy = 0;
    float cx = 0;
    float cy = 0;
};

struct RgbCameraInfo {
    int width = 0;
    int height = 0;
    float fx = 0;
    float fy = 0;
    float cx = 0;
    float cy = 0;
    std::string model = "pinhole";
    std::vector<float> distortion;
    std::array<float, 16> rgb_from_depth{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};

struct SensorInfo {
    std::string kind;
    std::string name;
    std::string connection;
    std::string serial;
    std::string address;
};

Options parse_options(int argc, char **argv) {
    Options options;
    for(int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&]() -> std::string {
            if(index + 1 >= argc) throw std::runtime_error("Missing value after " + argument);
            return argv[++index];
        };
        if(argument == "--capabilities") options.capabilities = true;
        else if(argument == "--probe") options.probe = true;
        else if(argument == "--list") options.list = true;
        else if(argument == "--stream-rgbd") options.stream_rgbd = true;
        else if(argument == "--preview") options.preview = true;
        else if(argument == "--phase") options.root = value();
        else if(argument == "--id") options.id = value();
        else if(argument == "--name") options.name = value();
        else if(argument == "--sensor") options.sensor = value();
        else if(argument == "--device") options.device_id = value();
        else if(argument == "--connection") options.connection = value();
        else if(argument == "--address") options.address = value();
        else if(argument == "--depth-fov") options.depth_fov = value();
        else if(argument == "--depth-binned") options.depth_binned = true;
        else if(argument == "--fps") options.fps = std::stoi(value());
        else if(argument == "--max-depth") options.max_depth_m = std::stof(value());
        else if(argument == "--rgb-quality") options.rgb_quality = std::stoi(value());
        else if(argument == "--max-rgb-dimension") options.max_rgb_dimension = static_cast<std::uint32_t>(std::stoul(value()));
        else if(argument == "--imu") options.use_imu = true;
        else throw std::runtime_error("Unknown argument: " + argument);
    }
    if(!options.capabilities && !options.probe && !options.list && options.root.empty()) throw std::runtime_error("Pass --phase PATH");
    options.rgb_quality = std::clamp(options.rgb_quality, 60, 100);
    if(options.sensor != "azure_kinect" && options.sensor != "femto_mega") {
        throw std::runtime_error("Modern worker supports azure_kinect or femto_mega");
    }
    if(options.connection != "usb" && options.connection != "network") {
        throw std::runtime_error("Connection must be usb or network");
    }
    if(options.depth_fov != "narrow" && options.depth_fov != "wide") {
        throw std::runtime_error("Depth field of view must be narrow or wide");
    }
    if(options.sensor != "femto_mega" && options.connection == "network") {
        throw std::runtime_error("Network capture is supported only for Orbbec Femto Mega");
    }
    if(!options.list && options.sensor == "femto_mega" && options.connection == "network" && options.address.empty()) {
        throw std::runtime_error("Femto Mega network capture requires --address IP[:PORT]");
    }
    options.fps = std::clamp(options.fps, 1, 30);
    options.max_depth_m = std::clamp(options.max_depth_m, 0.5F, 8.0F);
    return options;
}

std::string json_escape(const std::string &value) {
    std::ostringstream escaped;
    for(const unsigned char character : value) {
        switch(character) {
            case '\\': escaped << "\\\\"; break;
            case '"': escaped << "\\\""; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if(character < 0x20) escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(character);
                else escaped << character;
        }
    }
    return escaped.str();
}

struct AvailableSensorInfo {
    std::string id;
    std::string kind;
    std::string name;
    std::string connection;
    std::string address;
    std::string serial;
    bool supports_imu = true;
};

std::string azure_device_id(const std::string &serial, std::uint32_t index) {
    return serial.empty() ? "azure_kinect:index:" + std::to_string(index) : "azure_kinect:" + serial;
}

std::string orbbec_device_id(const std::string &connection, const std::string &serial,
                             const std::string &address, const std::string &uid) {
    const std::string identity = !serial.empty() ? serial : !uid.empty() ? uid : address;
    return "femto_mega:" + connection + ':' + identity;
}

void print_sensor_list(const std::vector<AvailableSensorInfo> &sensors) {
    std::cout << '[';
    for(std::size_t index = 0; index < sensors.size(); ++index) {
        if(index > 0) std::cout << ',';
        const auto &sensor = sensors[index];
        std::cout << "{\"id\":\"" << json_escape(sensor.id)
                  << "\",\"kind\":\"" << json_escape(sensor.kind)
                  << "\",\"name\":\"" << json_escape(sensor.name)
                  << "\",\"connection\":\"" << json_escape(sensor.connection)
                  << "\",\"address\":\"" << json_escape(sensor.address)
                  << "\",\"serial\":\"" << json_escape(sensor.serial)
                  << "\",\"connected\":true"
                  << ",\"supportsImu\":" << (sensor.supports_imu ? "true" : "false") << '}';
    }
    std::cout << "]\n";
}

int list_sensors() {
    std::vector<AvailableSensorInfo> sensors;
#ifdef SCANLAN_HAS_AZURE_KINECT
    const auto azure_count = k4a_device_get_installed_count();
    for(std::uint32_t index = 0; index < azure_count; ++index) {
        sensors.push_back({azure_device_id("", index), "azure_kinect", "Azure Kinect DK", "usb", "", "", true});
    }
#endif
#ifdef SCANLAN_HAS_ORBBEC
    try {
        auto context = std::make_shared<ob::Context>();
        context->enableNetDeviceEnumeration(true);
        auto devices = context->queryDeviceList();
        for(std::uint32_t index = 0; index < devices->getCount(); ++index) {
            const std::string name = devices->getName(index);
            if(name.find("Femto Mega") == std::string::npos) continue;
            const std::string raw_connection = devices->getConnectionType(index);
            std::string lower_connection = raw_connection;
            std::transform(lower_connection.begin(), lower_connection.end(), lower_connection.begin(),
                           [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
            const std::string connection = lower_connection.find("ethernet") != std::string::npos ? "network" : "usb";
            const std::string address = connection == "network" ? devices->getIpAddress(index) : "";
            const std::string serial = devices->getSerialNumber(index);
            const std::string uid = devices->getUid(index);
            sensors.push_back({orbbec_device_id(connection, serial, address, uid), "femto_mega", name,
                               connection, address == "0.0.0.0" ? "" : address, serial, true});
        }
    } catch(const ob::Error &) {
        // Enumeration of one SDK must not hide devices found through the other SDK.
    }
#endif
    print_sensor_list(sensors);
    return 0;
}

std::string utc_now() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_s(&utc, &now);
    std::ostringstream result;
    result << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return result.str();
}

void write_text_atomic(const fs::path &path, const std::string &text) {
    fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if(!output) throw std::runtime_error("Could not write " + temporary.string());
        output << text;
        output.flush();
        if(!output) throw std::runtime_error("Could not flush " + temporary.string());
        output.close();
        if(!output) throw std::runtime_error("Could not close " + temporary.string());
    }
    for(int attempt = 0; attempt < 40; ++attempt) {
        if(MoveFileExW(
            temporary.c_str(),
            path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(5 + attempt));
    }
    throw std::runtime_error(
        "Could not publish " + path.string() + " (Windows error "
        + std::to_string(GetLastError()) + ")");
}

void write_manifest(const Options &options, const SensorInfo &sensor, const CameraInfo &camera,
                    const RgbCameraInfo &source_rgb,
                    const std::string &created_at, std::uint32_t frame_count, std::uint32_t duration_seconds,
                    bool imu_active, std::uint32_t rgb_drops) {
    const double rgb_scale = options.max_rgb_dimension > 0
        && options.max_rgb_dimension < static_cast<std::uint32_t>(std::max(source_rgb.width, source_rgb.height))
        ? static_cast<double>(options.max_rgb_dimension) / std::max(source_rgb.width, source_rgb.height) : 1.0;
    const int rgb_width = static_cast<int>(std::lround(source_rgb.width * rgb_scale));
    const int rgb_height = static_cast<int>(std::lround(source_rgb.height * rgb_scale));
    const double rgb_scale_x = static_cast<double>(rgb_width) / source_rgb.width;
    const double rgb_scale_y = static_cast<double>(rgb_height) / source_rgb.height;
    const bool rgb_resized = rgb_width != source_rgb.width || rgb_height != source_rgb.height;
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"schemaVersion\": 3,\n"
           << "  \"id\": \"" << json_escape(options.id) << "\",\n"
           << "  \"name\": \"" << json_escape(options.name) << "\",\n"
           << "  \"createdAt\": \"" << created_at << "\",\n"
           << "  \"frameCount\": " << frame_count << ",\n"
           << "  \"durationSeconds\": " << duration_seconds << ",\n"
           << "  \"frameFormat\": \"raw-u16-depth+raw-rgb\",\n"
           << "  \"poseSource\": \"" << (imu_active ? "imu_aided_offline" : "estimated_offline") << "\",\n"
           << "  \"sensor\": {\"kind\": \"" << json_escape(sensor.kind)
           << "\", \"name\": \"" << json_escape(sensor.name)
           << "\", \"connection\": \"" << json_escape(sensor.connection)
           << "\", \"serial\": \"" << json_escape(sensor.serial)
           << "\", \"address\": \"" << json_escape(sensor.address) << "\"},\n";
    if(imu_active) {
        output << "  \"imu\": {\"path\": \"imu.csv\", \"coordinateFrame\": \"depth_camera\", "
               << "\"accelerationUnit\": \"m/s^2\", \"angularVelocityUnit\": \"rad/s\"},\n";
    }
    output << "  \"camera\": {\n"
           << "    \"width\": " << camera.width << ", \"height\": " << camera.height << ",\n"
           << "    \"fx\": " << camera.fx << ", \"fy\": " << camera.fy << ",\n"
           << "    \"cx\": " << camera.cx << ", \"cy\": " << camera.cy << ",\n"
           << "    \"depth_scale\": 1000.0, \"max_depth_m\": " << options.max_depth_m << ",\n"
           << "    \"depth_field_of_view\": \"" << options.depth_fov << "\", "
           << "\"depth_binned\": " << (options.depth_binned ? "true" : "false") << "\n"
           << "  },\n"
           << "  \"rgbCamera\": {\"width\": " << rgb_width
           << ", \"height\": " << rgb_height
           << ", \"fx\": " << source_rgb.fx * rgb_scale_x << ", \"fy\": " << source_rgb.fy * rgb_scale_y
           << ", \"cx\": " << (source_rgb.cx + 0.5) * rgb_scale_x - 0.5
           << ", \"cy\": " << (source_rgb.cy + 0.5) * rgb_scale_y - 0.5
           << ", \"model\": \"" << json_escape(source_rgb.model) << "\", \"distortion\": [";
    for(std::size_t index = 0; index < source_rgb.distortion.size(); ++index) {
        if(index) output << ',';
        output << source_rgb.distortion[index];
    }
    output << "]},\n  \"rgbFromDepth\": [";
    for(std::size_t index = 0; index < source_rgb.rgb_from_depth.size(); ++index) {
        if(index) output << ',';
        output << source_rgb.rgb_from_depth[index];
    }
    output << "],\n  \"sourceRgb\": {\"format\": \"jpeg\", \"quality\": " << options.rgb_quality
           << ", \"nativeResolution\": " << (rgb_resized ? "false" : "true")
           << ", \"droppedFrames\": " << rgb_drops << "}\n"
           << "}\n";
    write_text_atomic(options.root / "phase.json", output.str());
}

void write_live_status(const Options &options, const SensorInfo &sensor, std::uint64_t timestamp_us,
                       std::uint32_t frames, float stream_fps, bool imu_active, float imu_rate_hz,
                       const std::string &tracking_status) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(2)
           << "{\n"
           << "  \"timestampUs\": " << timestamp_us << ",\n"
           << "  \"frameCount\": " << frames << ",\n"
           << "  \"streamFps\": " << stream_fps << ",\n"
           << "  \"tracking\": false,\n"
           << "  \"trackingStatus\": \"" << json_escape(tracking_status) << "\",\n"
           << "  \"sensorName\": \"" << json_escape(sensor.name) << "\",\n"
           << "  \"imuActive\": " << (imu_active ? "true" : "false") << ",\n"
           << "  \"imuRateHz\": " << imu_rate_hz << "\n"
           << "}\n";
    write_text_atomic(options.root / "live.json", output.str());
}

class ImuCsv {
public:
    explicit ImuCsv(const fs::path &path) : output_(path, std::ios::trunc), started_(std::chrono::steady_clock::now()) {
        if(!output_) throw std::runtime_error("Could not create " + path.string());
        output_ << "timestamp_us,type,x,y,z,temperature_c\n";
    }

    void write(std::uint64_t timestamp_us, const char *type, float x, float y, float z, float temperature) {
        std::lock_guard lock(mutex_);
        output_ << timestamp_us << ',' << type << ',' << std::setprecision(9)
                << x << ',' << y << ',' << z << ',' << temperature << '\n';
        ++samples_;
        if(samples_ % 32 == 0) output_.flush();
    }

    float rate_hz() const {
        const double elapsed = std::max(0.001, std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count());
        return static_cast<float>(samples_.load() / elapsed);
    }

private:
    std::ofstream output_;
    mutable std::mutex mutex_;
    std::atomic<std::uint64_t> samples_{0};
    std::chrono::steady_clock::time_point started_;
};

void prepare_root(const Options &options) {
    fs::create_directories(options.root);
    fs::remove(options.root / "stop.flag");
    fs::remove(options.root / "record.flag");
    fs::remove(options.root / "reconstruction-reset.flag");
    fs::remove(options.root / "recording.flag");
    fs::remove(options.root / "preview.flag");
    if(options.preview) std::ofstream(options.root / "preview.flag").close();
    fs::remove(options.root / "live.json");
    fs::create_directories(options.root / "depth");
    fs::create_directories(options.root / "color");
    fs::create_directories(options.root / "rgb");
}

#ifdef SCANLAN_HAS_AZURE_KINECT
std::array<float, 3> rotate_k4a(const k4a_calibration_extrinsics_t &extrinsics, const k4a_float3_t &value) {
    return {
        extrinsics.rotation[0] * value.v[0] + extrinsics.rotation[1] * value.v[1] + extrinsics.rotation[2] * value.v[2],
        extrinsics.rotation[3] * value.v[0] + extrinsics.rotation[4] * value.v[1] + extrinsics.rotation[5] * value.v[2],
        extrinsics.rotation[6] * value.v[0] + extrinsics.rotation[7] * value.v[1] + extrinsics.rotation[8] * value.v[2]
    };
}

k4a_depth_mode_t azure_depth_mode(const Options &options) {
    if(options.depth_fov == "wide") {
        return options.depth_binned ? K4A_DEPTH_MODE_WFOV_2X2BINNED : K4A_DEPTH_MODE_WFOV_UNBINNED;
    }
    return options.depth_binned ? K4A_DEPTH_MODE_NFOV_2X2BINNED : K4A_DEPTH_MODE_NFOV_UNBINNED;
}

int run_azure(const Options &options) {
    const auto device_count = k4a_device_get_installed_count();
    if(device_count == 0) throw std::runtime_error("Azure Kinect DK was not found");
    k4a_device_t device = nullptr;
    for(std::uint32_t index = 0; index < device_count; ++index) {
        k4a_device_t candidate = nullptr;
        if(k4a_device_open(index, &candidate) != K4A_RESULT_SUCCEEDED) continue;
        char serial[256]{};
        std::size_t serial_size = sizeof(serial);
        k4a_device_get_serialnum(candidate, serial, &serial_size);
        const std::string candidate_id = azure_device_id(serial, index);
        const std::string index_id = azure_device_id("", index);
        if(options.device_id.empty() || options.device_id == candidate_id || options.device_id == index_id) {
            device = candidate;
            break;
        }
        k4a_device_close(candidate);
    }
    if(!device) throw std::runtime_error("The selected Azure Kinect DK is no longer connected");
    try {
        char serial[256]{};
        std::size_t serial_size = sizeof(serial);
        k4a_device_get_serialnum(device, serial, &serial_size);
        SensorInfo sensor{"azure_kinect", "Azure Kinect DK", "usb", serial, ""};
        k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
        const auto depth_mode = requested_depth_mode(options);
        config.depth_mode = azure_depth_mode(options);
        config.camera_fps = depth_mode.native_fps == 15 ? K4A_FRAMES_PER_SECOND_15 : K4A_FRAMES_PER_SECOND_30;
        config.synchronized_images_only = !options.probe;
        config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
        // Azure Kinect cannot synchronize 2160p color at 30 fps. Preserve the
        // full-rate tracking path with its highest supported RGB mode, while
        // the 15 fps wide/unbinned depth profile can retain native 4K color.
        config.color_resolution = depth_mode.native_fps == 15
            ? K4A_COLOR_RESOLUTION_2160P
            : K4A_COLOR_RESOLUTION_1536P;
        k4a_calibration_t calibration{};
        if(k4a_device_get_calibration(device, config.depth_mode, config.color_resolution, &calibration) != K4A_RESULT_SUCCEEDED) {
            throw std::runtime_error("Azure Kinect calibration is unavailable");
        }
        const auto &intrinsics = calibration.depth_camera_calibration.intrinsics.parameters.param;
        CameraInfo camera{calibration.depth_camera_calibration.resolution_width,
                          calibration.depth_camera_calibration.resolution_height,
                          intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy};
        const auto &rgb_intrinsics = calibration.color_camera_calibration.intrinsics.parameters.param;
        const auto &depth_to_rgb = calibration.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];
        RgbCameraInfo rgb_camera{
            calibration.color_camera_calibration.resolution_width,
            calibration.color_camera_calibration.resolution_height,
            rgb_intrinsics.fx, rgb_intrinsics.fy, rgb_intrinsics.cx, rgb_intrinsics.cy,
            "opencv_rational",
            {
                rgb_intrinsics.k1, rgb_intrinsics.k2,
                rgb_intrinsics.p1, rgb_intrinsics.p2,
                rgb_intrinsics.k3, rgb_intrinsics.k4,
                rgb_intrinsics.k5, rgb_intrinsics.k6
            },
            {
                depth_to_rgb.rotation[0], depth_to_rgb.rotation[1], depth_to_rgb.rotation[2], depth_to_rgb.translation[0] / 1000.0F,
                depth_to_rgb.rotation[3], depth_to_rgb.rotation[4], depth_to_rgb.rotation[5], depth_to_rgb.translation[1] / 1000.0F,
                depth_to_rgb.rotation[6], depth_to_rgb.rotation[7], depth_to_rgb.rotation[8], depth_to_rgb.translation[2] / 1000.0F,
                0, 0, 0, 1
            }
        };
        if(k4a_device_start_cameras(device, &config) != K4A_RESULT_SUCCEEDED) {
            throw std::runtime_error("Azure Kinect cameras could not start; close other camera applications and check USB 3/power");
        }

        if(options.probe) {
            k4a_capture_t capture = nullptr;
            const auto result = k4a_device_get_capture(device, &capture, 5000);
            if(capture) k4a_capture_release(capture);
            k4a_device_stop_cameras(device);
            k4a_device_close(device);
            if(result != K4A_WAIT_RESULT_SUCCEEDED) throw std::runtime_error("Azure Kinect did not deliver a depth frame");
            std::cout << "Azure Kinect DK ready\n";
            return 0;
        }

        prepare_root(options);
        std::string created_at = utc_now();
        scanlan::RgbdArchiveWriter archive(
            options.root, options.rgb_quality, options.max_rgb_dimension, 8);
        std::unique_ptr<ImuCsv> imu;
        scanlan::GyroDeltaIntegrator gyro_integrator;
        bool imu_active = false;
        bool recording = !options.preview;
        auto accel_to_depth = calibration.extrinsics[K4A_CALIBRATION_TYPE_ACCEL][K4A_CALIBRATION_TYPE_DEPTH];
        auto gyro_to_depth = calibration.extrinsics[K4A_CALIBRATION_TYPE_GYRO][K4A_CALIBRATION_TYPE_DEPTH];
        if(options.use_imu && k4a_device_start_imu(device) == K4A_RESULT_SUCCEEDED) {
            imu = std::make_unique<ImuCsv>(options.root / "imu.csv");
            imu_active = true;
        }
        const auto drain_imu = [&]() {
            if(!imu_active) return;
            k4a_imu_sample_t sample{};
            while(k4a_device_get_imu_sample(device, &sample, 0) == K4A_WAIT_RESULT_SUCCEEDED) {
                const auto accel = rotate_k4a(accel_to_depth, sample.acc_sample);
                const auto gyro = rotate_k4a(gyro_to_depth, sample.gyro_sample);
                if(recording) {
                    imu->write(sample.acc_timestamp_usec, "accel", accel[0], accel[1], accel[2], sample.temperature);
                    imu->write(sample.gyro_timestamp_usec, "gyro", gyro[0], gyro[1], gyro[2], sample.temperature);
                }
                gyro_integrator.add(sample.gyro_timestamp_usec, gyro[0], gyro[1], gyro[2]);
            }
        };
        write_manifest(options, sensor, camera, rgb_camera, created_at, 0, 0, imu_active, 0);

        std::unique_ptr<scanlan::RgbdStreamWriter> rgbd_stream;
        if(options.stream_rgbd) {
            rgbd_stream = std::make_unique<scanlan::RgbdStreamWriter>(std::cout, 3);
        }

        k4a_transformation_t transformation = k4a_transformation_create(&calibration);
        if(!transformation) throw std::runtime_error("Azure Kinect calibration transform could not be created");
        std::uint64_t source_frames = 0;
        std::uint64_t recording_frames = 0;
        const auto started = std::chrono::steady_clock::now();
        auto recording_started = started;
        while(!fs::exists(options.root / "stop.flag")) {
            k4a_capture_t capture = nullptr;
            const auto wait = k4a_device_get_capture(device, &capture, 1000);
            if(wait == K4A_WAIT_RESULT_TIMEOUT) continue;
            if(wait != K4A_WAIT_RESULT_SUCCEEDED || !capture) throw std::runtime_error("Azure Kinect stopped delivering frames");
            drain_imu();
            k4a_image_t depth_image = k4a_capture_get_depth_image(capture);
            if(!depth_image) { k4a_capture_release(capture); continue; }
            ++source_frames;
            if(!recording && fs::exists(options.root / "record.flag")) {
                recording = true;
                recording_started = std::chrono::steady_clock::now();
                created_at = utc_now();
                {
                    std::ofstream signal(options.root / "recording.flag");
                    signal << (source_frames - 1) << '\n';
                }
                fs::remove(options.root / "preview.flag");
                fs::remove(options.root / "record.flag");
            }
            if(recording) ++recording_frames;
            const bool save_source = recording && scanlan::archive_frame_due(
                recording_frames, depth_mode.native_fps, options.fps);
            const std::uint64_t timestamp_us = k4a_image_get_device_timestamp_usec(depth_image);
            std::vector<std::uint16_t> depth(static_cast<std::size_t>(camera.width * camera.height));
            const auto *depth_buffer = k4a_image_get_buffer(depth_image);
            const int depth_stride = k4a_image_get_stride_bytes(depth_image);
            for(int y = 0; y < camera.height; ++y) {
                const auto *row = reinterpret_cast<const std::uint16_t *>(depth_buffer + y * depth_stride);
                for(int x = 0; x < camera.width; ++x) {
                    const auto value = row[x];
                    depth[static_cast<std::size_t>(y * camera.width + x)] = value > 0 && value <= options.max_depth_m * 1000.0F ? value : 0;
                }
            }
            const double elapsed = std::max(0.001, std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
            const float stream_fps = static_cast<float>(source_frames / elapsed);
            k4a_image_t color_image = k4a_capture_get_color_image(capture);
            if(!color_image) {
                k4a_image_release(depth_image);
                k4a_capture_release(capture);
                continue;
            }
            k4a_image_t aligned = nullptr;
            if(k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32, camera.width, camera.height, camera.width * 4, &aligned) != K4A_RESULT_SUCCEEDED
                || k4a_transformation_color_image_to_depth_camera(transformation, depth_image, color_image, aligned) != K4A_RESULT_SUCCEEDED) {
                if(aligned) k4a_image_release(aligned);
                k4a_image_release(color_image);
                k4a_image_release(depth_image);
                k4a_capture_release(capture);
                throw std::runtime_error("Azure Kinect color-to-depth alignment failed");
            }
            const auto *bgra = k4a_image_get_buffer(aligned);
            const int color_stride = k4a_image_get_stride_bytes(aligned);
            std::vector<std::uint8_t> rgb(static_cast<std::size_t>(camera.width * camera.height * 3));
            for(int y = 0; y < camera.height; ++y) for(int x = 0; x < camera.width; ++x) {
                const auto *source = bgra + y * color_stride + x * 4;
                const std::size_t target = static_cast<std::size_t>((y * camera.width + x) * 3);
                rgb[target] = source[2]; rgb[target + 1] = source[1]; rgb[target + 2] = source[0];
            }
            const std::uint64_t rgb_timestamp_us = k4a_image_get_device_timestamp_usec(color_image);
            if(rgbd_stream) {
                const auto gyro_delta = imu_active ? gyro_integrator.take() : std::nullopt;
                rgbd_stream->publish(
                    source_frames - 1,
                    timestamp_us,
                    rgb_timestamp_us,
                    static_cast<std::uint32_t>(camera.width),
                    static_cast<std::uint32_t>(camera.height),
                    camera.fx,
                    camera.fy,
                    camera.cx,
                    camera.cy,
                    1000.0F,
                    0.25F,
                    options.max_depth_m,
                    depth,
                    rgb,
                    gyro_delta ? &*gyro_delta : nullptr);
            }
            if(save_source) {
                const int source_width = k4a_image_get_width_pixels(color_image);
                const int source_height = k4a_image_get_height_pixels(color_image);
                const int source_stride = k4a_image_get_stride_bytes(color_image);
                const auto *source_bgra = k4a_image_get_buffer(color_image);
                std::vector<std::uint8_t> native_rgb(static_cast<std::size_t>(source_width * source_height * 3));
                for(int y = 0; y < source_height; ++y) for(int x = 0; x < source_width; ++x) {
                    const auto *source = source_bgra + y * source_stride + x * 4;
                    const std::size_t target = static_cast<std::size_t>((y * source_width + x) * 3);
                    native_rgb[target] = source[2]; native_rgb[target + 1] = source[1]; native_rgb[target + 2] = source[0];
                }
                scanlan::RgbdArchiveFrame archive_frame;
                archive_frame.source_sequence = source_frames - 1;
                archive_frame.depth_timestamp_us = timestamp_us;
                archive_frame.color_timestamp_us = rgb_timestamp_us;
                archive_frame.depth = std::move(depth);
                archive_frame.aligned_color = std::move(rgb);
                archive_frame.native_color_width = static_cast<std::uint32_t>(source_width);
                archive_frame.native_color_height = static_cast<std::uint32_t>(source_height);
                archive_frame.native_color = std::move(native_rgb);
                archive.submit(std::move(archive_frame));
            }
            k4a_image_release(aligned);
            k4a_image_release(color_image);
            if(source_frames == 1 || source_frames % 3 == 0) {
                write_live_status(options, sensor, timestamp_us, archive.saved(),
                                  stream_fps, imu_active, imu ? imu->rate_hz() : 0.0F,
                                  recording
                                      ? "Full-rate RGB-D tracking and bounded background recording"
                                      : "Live RGB-D preview ready; press Capture to begin recording");
            }
            k4a_image_release(depth_image);
            k4a_capture_release(capture);
        }
        if(transformation) k4a_transformation_destroy(transformation);
        if(imu_active) {
            drain_imu();
            k4a_device_stop_imu(device);
        }
        if(rgbd_stream) rgbd_stream->close();
        const auto duration = recording_frames == 0 ? 0U : static_cast<std::uint32_t>(std::max<std::int64_t>(1,
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - recording_started).count()));
        archive.close();
        const auto saved_frames = archive.saved();
        write_manifest(options, sensor, camera, rgb_camera, created_at, saved_frames, duration, imu_active,
                       archive.dropped() + archive.failed());
        k4a_device_stop_cameras(device);
        k4a_device_close(device);
        return saved_frames > 0 || options.preview ? 0 : 2;
    } catch(...) {
        k4a_device_stop_imu(device);
        k4a_device_stop_cameras(device);
        k4a_device_close(device);
        throw;
    }
}
#else
int run_azure(const Options &) {
    throw std::runtime_error("Azure Kinect support was not compiled; install Azure Kinect Sensor SDK 1.4.x and rebuild");
}
#endif

#ifdef SCANLAN_HAS_ORBBEC
class OrbbecDepthRectifier {
public:
    OrbbecDepthRectifier(const std::shared_ptr<ob::Pipeline> &pipeline,
                         const std::shared_ptr<ob::Config> &config,
                         int width, int height)
        : width_(width), height_(height) {
        if(width_ < 2 || height_ < 2) {
            throw std::runtime_error("Femto Mega depth dimensions cannot be rectified");
        }

        const auto calibration = pipeline->getCalibrationParam(config);
        const auto &native_intrinsic = calibration.intrinsics[OB_SENSOR_DEPTH];
        std::uint32_t table_size = static_cast<std::uint32_t>(
            static_cast<std::size_t>(width_) * height_ * 2);
        xy_storage_.resize(table_size);
        if(!ob::CoordinateTransformHelper::transformationInitXYTables(
               calibration, OB_SENSOR_DEPTH, xy_storage_.data(), &table_size, &xy_tables_)) {
            throw std::runtime_error("Femto Mega depth distortion table is unavailable");
        }
        if(xy_tables_.width != width_ || xy_tables_.height != height_
            || !xy_tables_.xTable || !xy_tables_.yTable) {
            throw std::runtime_error("Femto Mega depth distortion table has unexpected dimensions");
        }
        float minimum_x = std::numeric_limits<float>::infinity();
        float maximum_x = -std::numeric_limits<float>::infinity();
        float minimum_y = std::numeric_limits<float>::infinity();
        float maximum_y = -std::numeric_limits<float>::infinity();
        const int optical_x = std::clamp(
            static_cast<int>(std::lround(native_intrinsic.cx)), 0, width_ - 1);
        const int optical_y = std::clamp(
            static_cast<int>(std::lround(native_intrinsic.cy)), 0, height_ - 1);
        for(int x = 0; x < width_; ++x) {
            const std::size_t index = static_cast<std::size_t>(optical_y * width_ + x);
            const float ray_x = xy_tables_.xTable[index];
            const float ray_y = xy_tables_.yTable[index];
            if(!valid_ray(ray_x, ray_y)) continue;
            minimum_x = std::min(minimum_x, ray_x);
            maximum_x = std::max(maximum_x, ray_x);
        }
        for(int y = 0; y < height_; ++y) {
            const std::size_t index = static_cast<std::size_t>(y * width_ + optical_x);
            const float ray_x = xy_tables_.xTable[index];
            const float ray_y = xy_tables_.yTable[index];
            if(!valid_ray(ray_x, ray_y)) continue;
            minimum_y = std::min(minimum_y, ray_y);
            maximum_y = std::max(maximum_y, ray_y);
        }
        if(!std::isfinite(minimum_x) || !std::isfinite(minimum_y)
            || maximum_x - minimum_x < 0.1F || maximum_y - minimum_y < 0.1F) {
            throw std::runtime_error("Femto Mega returned an invalid depth distortion table");
        }

        // The SDK's XY table gives the calibrated undistorted ray (X/Z,Y/Z)
        // for every pixel in the native, visibly fisheye depth raster. Build a
        // same-resolution virtual pinhole camera that contains the complete
        // horizontal and vertical FOV. Rectilinear projection cannot retain
        // the fisheye image's extreme diagonal corners without compressing the
        // useful room view into a small central area, so bounds come from the
        // calibrated optical row and column. Invert that table once for cheap
        // per-frame nearest-neighbour depth/color resampling.
        camera_ = {
            width_,
            height_,
            static_cast<float>(width_ - 1) / (maximum_x - minimum_x),
            static_cast<float>(height_ - 1) / (maximum_y - minimum_y),
            -minimum_x * static_cast<float>(width_ - 1) / (maximum_x - minimum_x),
            -minimum_y * static_cast<float>(height_ - 1) / (maximum_y - minimum_y),
        };
        source_index_.resize(pixel_count(), invalid_index);
        build_inverse_map(minimum_x, maximum_x, minimum_y, maximum_y);
    }

    const CameraInfo &camera() const { return camera_; }

    std::vector<std::uint16_t> rectify_depth(const std::vector<std::uint16_t> &source) const {
        if(source.size() != pixel_count()) {
            throw std::runtime_error("Femto Mega depth rectification input has unexpected dimensions");
        }
        std::vector<std::uint16_t> target(pixel_count(), 0);
        for(std::size_t index = 0; index < target.size(); ++index) {
            if(source_index_[index] != invalid_index) target[index] = source[source_index_[index]];
        }
        return target;
    }

    std::vector<std::uint8_t> rectify_color(const std::vector<std::uint8_t> &source) const {
        if(source.size() != pixel_count() * 3) {
            throw std::runtime_error("Femto Mega color rectification input has unexpected dimensions");
        }
        std::vector<std::uint8_t> target(pixel_count() * 3, 0);
        for(std::size_t index = 0; index < pixel_count(); ++index) {
            if(source_index_[index] == invalid_index) continue;
            const std::size_t source_offset = static_cast<std::size_t>(source_index_[index]) * 3;
            const std::size_t target_offset = index * 3;
            target[target_offset] = source[source_offset];
            target[target_offset + 1] = source[source_offset + 1];
            target[target_offset + 2] = source[source_offset + 2];
        }
        return target;
    }

private:
    static constexpr std::uint32_t invalid_index = std::numeric_limits<std::uint32_t>::max();

    std::size_t pixel_count() const {
        return static_cast<std::size_t>(width_) * height_;
    }

    static bool valid_ray(float x, float y) {
        return std::isfinite(x) && std::isfinite(y) && std::abs(x) < 10.0F && std::abs(y) < 10.0F;
    }

    bool sample_ray(float source_x, float source_y, float &ray_x, float &ray_y,
                    float &dx_ds, float &dx_dt, float &dy_ds, float &dy_dt) const {
        if(source_x < 0.0F || source_y < 0.0F
            || source_x > static_cast<float>(width_ - 1)
            || source_y > static_cast<float>(height_ - 1)) return false;
        const int x0 = std::min(static_cast<int>(std::floor(source_x)), width_ - 2);
        const int y0 = std::min(static_cast<int>(std::floor(source_y)), height_ - 2);
        const float s = source_x - static_cast<float>(x0);
        const float t = source_y - static_cast<float>(y0);
        const std::size_t i00 = static_cast<std::size_t>(y0 * width_ + x0);
        const std::size_t i10 = i00 + 1;
        const std::size_t i01 = i00 + width_;
        const std::size_t i11 = i01 + 1;
        const float x00 = xy_tables_.xTable[i00], x10 = xy_tables_.xTable[i10];
        const float x01 = xy_tables_.xTable[i01], x11 = xy_tables_.xTable[i11];
        const float y00 = xy_tables_.yTable[i00], y10 = xy_tables_.yTable[i10];
        const float y01 = xy_tables_.yTable[i01], y11 = xy_tables_.yTable[i11];
        if(!valid_ray(x00, y00) || !valid_ray(x10, y10)
            || !valid_ray(x01, y01) || !valid_ray(x11, y11)) return false;
        const float top_x = x00 + s * (x10 - x00);
        const float bottom_x = x01 + s * (x11 - x01);
        const float top_y = y00 + s * (y10 - y00);
        const float bottom_y = y01 + s * (y11 - y01);
        ray_x = top_x + t * (bottom_x - top_x);
        ray_y = top_y + t * (bottom_y - top_y);
        dx_ds = (1.0F - t) * (x10 - x00) + t * (x11 - x01);
        dx_dt = bottom_x - top_x;
        dy_ds = (1.0F - t) * (y10 - y00) + t * (y11 - y01);
        dy_dt = bottom_y - top_y;
        return true;
    }

    void build_inverse_map(float minimum_x, float maximum_x,
                           float minimum_y, float maximum_y) {
        std::size_t mapped = 0;
        double squared_pixel_error = 0.0;
        float maximum_pixel_error = 0.0F;
        for(int y = 0; y < height_; ++y) {
            const float target_y = (static_cast<float>(y) - camera_.cy) / camera_.fy;
            for(int x = 0; x < width_; ++x) {
                const float target_x = (static_cast<float>(x) - camera_.cx) / camera_.fx;
                float source_x = (target_x - minimum_x) / (maximum_x - minimum_x)
                    * static_cast<float>(width_ - 1);
                float source_y = (target_y - minimum_y) / (maximum_y - minimum_y)
                    * static_cast<float>(height_ - 1);
                bool converged = false;
                for(int iteration = 0; iteration < 10; ++iteration) {
                    float ray_x = 0.0F, ray_y = 0.0F;
                    float dx_ds = 0.0F, dx_dt = 0.0F, dy_ds = 0.0F, dy_dt = 0.0F;
                    if(!sample_ray(source_x, source_y, ray_x, ray_y,
                                   dx_ds, dx_dt, dy_ds, dy_dt)) break;
                    const float error_x = ray_x - target_x;
                    const float error_y = ray_y - target_y;
                    const float pixel_error_x = error_x * camera_.fx;
                    const float pixel_error_y = error_y * camera_.fy;
                    if(pixel_error_x * pixel_error_x + pixel_error_y * pixel_error_y < 0.01F) {
                        converged = true;
                        break;
                    }
                    const float determinant = dx_ds * dy_dt - dx_dt * dy_ds;
                    if(!std::isfinite(determinant) || std::abs(determinant) < 1e-12F) break;
                    const float delta_x = std::clamp(
                        (dy_dt * error_x - dx_dt * error_y) / determinant, -32.0F, 32.0F);
                    const float delta_y = std::clamp(
                        (-dy_ds * error_x + dx_ds * error_y) / determinant, -32.0F, 32.0F);
                    source_x -= delta_x;
                    source_y -= delta_y;
                }
                if(!converged || source_x < -0.5F || source_y < -0.5F
                    || source_x > static_cast<float>(width_) - 0.5F
                    || source_y > static_cast<float>(height_) - 0.5F) continue;
                const int nearest_x = std::clamp(static_cast<int>(std::lround(source_x)), 0, width_ - 1);
                const int nearest_y = std::clamp(static_cast<int>(std::lround(source_y)), 0, height_ - 1);
                const std::uint32_t nearest_index = static_cast<std::uint32_t>(nearest_y * width_ + nearest_x);
                source_index_[static_cast<std::size_t>(y * width_ + x)] =
                    nearest_index;
                const float nearest_error_x =
                    (xy_tables_.xTable[nearest_index] - target_x) * camera_.fx;
                const float nearest_error_y =
                    (xy_tables_.yTable[nearest_index] - target_y) * camera_.fy;
                const float nearest_error = std::sqrt(
                    nearest_error_x * nearest_error_x + nearest_error_y * nearest_error_y);
                squared_pixel_error += static_cast<double>(nearest_error) * nearest_error;
                maximum_pixel_error = std::max(maximum_pixel_error, nearest_error);
                ++mapped;
            }
        }
        std::cerr << "Femto Mega depth rectification: " << width_ << 'x' << height_
                  << " pinhole fx=" << camera_.fx << " fy=" << camera_.fy
                  << " cx=" << camera_.cx << " cy=" << camera_.cy
                  << " rays=[" << minimum_x << ',' << maximum_x << "]x["
                  << minimum_y << ',' << maximum_y << "] ("
                  << mapped << '/' << pixel_count() << " pixels, ray error rms="
                  << std::sqrt(squared_pixel_error / std::max<std::size_t>(1, mapped))
                  << " px max=" << maximum_pixel_error << " px)\n";
        if(mapped < pixel_count() / 2) {
            throw std::runtime_error("Femto Mega depth distortion map could not be inverted");
        }
    }

    int width_ = 0;
    int height_ = 0;
    CameraInfo camera_;
    std::vector<float> xy_storage_;
    OBXYTables xy_tables_{};
    std::vector<std::uint32_t> source_index_;
};

std::array<float, 9> rigid_orbbec_rotation(const float *rotation) {
    std::array<double, 3> x{rotation[0], rotation[3], rotation[6]};
    std::array<double, 3> y{rotation[1], rotation[4], rotation[7]};
    const auto normalize = [](std::array<double, 3> &value) {
        const double length = std::sqrt(
            value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
        if(length < 0.9) throw std::runtime_error("Femto Mega returned invalid camera extrinsics");
        for(auto &component : value) component /= length;
    };
    normalize(x);
    const double projection = x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    for(std::size_t index = 0; index < 3; ++index) y[index] -= projection * x[index];
    normalize(y);
    std::array<double, 3> z{
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0],
    };
    const double original_z_agreement =
        z[0] * rotation[2] + z[1] * rotation[5] + z[2] * rotation[8];
    if(original_z_agreement < 0.0) {
        for(auto &component : y) component = -component;
        for(auto &component : z) component = -component;
    }
    return {
        static_cast<float>(x[0]), static_cast<float>(y[0]), static_cast<float>(z[0]),
        static_cast<float>(x[1]), static_cast<float>(y[1]), static_cast<float>(z[1]),
        static_cast<float>(x[2]), static_cast<float>(y[2]), static_cast<float>(z[2]),
    };
}

std::pair<std::string, std::uint16_t> network_endpoint(const std::string &address) {
    const auto separator = address.rfind(':');
    if(separator == std::string::npos) return {address, 8090};
    return {address.substr(0, separator), static_cast<std::uint16_t>(std::stoi(address.substr(separator + 1)))};
}

std::array<float, 3> rotate_ob(const OBExtrinsic &extrinsic, const OBFloat3D &value) {
    return {
        extrinsic.rot[0] * value.x + extrinsic.rot[1] * value.y + extrinsic.rot[2] * value.z,
        extrinsic.rot[3] * value.x + extrinsic.rot[4] * value.y + extrinsic.rot[5] * value.z,
        extrinsic.rot[6] * value.x + extrinsic.rot[7] * value.y + extrinsic.rot[8] * value.z
    };
}

std::pair<std::string, std::vector<float>> orbbec_rgb_distortion(
    const OBCameraDistortion &distortion) {
    if(distortion.model == OB_DISTORTION_NONE) return {"pinhole", {}};
    if(distortion.model != OB_DISTORTION_BROWN_CONRADY
        && distortion.model != OB_DISTORTION_BROWN_CONRADY_K6) {
        throw std::runtime_error(
            "Femto Mega RGB calibration uses an unsupported lens model ("
            + std::to_string(static_cast<int>(distortion.model)) + ")");
    }
    return {
        "opencv_rational",
        {
            distortion.k1, distortion.k2, distortion.p1, distortion.p2,
            distortion.k3, distortion.k4, distortion.k5, distortion.k6
        }
    };
}

void complete_orbbec_narrow_color(
    const CameraInfo &depth_camera,
    const OBCameraIntrinsic &color_camera,
    const OBCameraDistortion &color_distortion,
    const OBExtrinsic &depth_to_color,
    const std::array<float, 9> &depth_to_color_rotation,
    const std::vector<std::uint16_t> &depth,
    const std::vector<std::uint8_t> &native_color,
    int native_color_width,
    int native_color_height,
    std::vector<std::uint8_t> &aligned_color) {
    const std::size_t pixel_count = static_cast<std::size_t>(
        depth_camera.width) * depth_camera.height;
    if(depth.size() != pixel_count || aligned_color.size() != pixel_count * 3
        || native_color.size() != static_cast<std::size_t>(
            native_color_width) * native_color_height * 3) {
        throw std::runtime_error("Femto Mega narrow color completion input is invalid");
    }

    // The narrow depth lens is slightly taller than the RGB camera after the
    // calibrated camera tilt and baseline are applied. Orbbec therefore emits
    // exact-black C2D pixels along the RGB boundary even though their depth is
    // valid. Preserve every SDK-projected sample, then complete only those
    // holes from the nearest calibrated native RGB boundary sample.
    for(int y = 0; y < depth_camera.height; ++y) {
        for(int x = 0; x < depth_camera.width; ++x) {
            const std::size_t index = static_cast<std::size_t>(
                y * depth_camera.width + x);
            const std::size_t target = index * 3;
            const float z = static_cast<float>(depth[index]);
            if(z <= 0.0F || aligned_color[target] != 0
                || aligned_color[target + 1] != 0 || aligned_color[target + 2] != 0) {
                continue;
            }

            const float depth_x = (static_cast<float>(x) - depth_camera.cx)
                * z / depth_camera.fx;
            const float depth_y = (static_cast<float>(y) - depth_camera.cy)
                * z / depth_camera.fy;
            const float color_x = depth_to_color_rotation[0] * depth_x
                + depth_to_color_rotation[1] * depth_y
                + depth_to_color_rotation[2] * z + depth_to_color.trans[0];
            const float color_y = depth_to_color_rotation[3] * depth_x
                + depth_to_color_rotation[4] * depth_y
                + depth_to_color_rotation[5] * z + depth_to_color.trans[1];
            const float color_z = depth_to_color_rotation[6] * depth_x
                + depth_to_color_rotation[7] * depth_y
                + depth_to_color_rotation[8] * z + depth_to_color.trans[2];
            if(!std::isfinite(color_z) || color_z <= 1.0F) continue;

            const float normalized_x = color_x / color_z;
            const float normalized_y = color_y / color_z;
            const float r2 = normalized_x * normalized_x + normalized_y * normalized_y;
            const float r4 = r2 * r2;
            const float r6 = r4 * r2;
            float distorted_x = normalized_x;
            float distorted_y = normalized_y;
            if(color_distortion.model == OB_DISTORTION_BROWN_CONRADY
                || color_distortion.model == OB_DISTORTION_BROWN_CONRADY_K6) {
                const float denominator = 1.0F + color_distortion.k4 * r2
                    + color_distortion.k5 * r4 + color_distortion.k6 * r6;
                if(std::abs(denominator) < 1e-8F) continue;
                const float radial = (1.0F + color_distortion.k1 * r2
                    + color_distortion.k2 * r4 + color_distortion.k3 * r6)
                    / denominator;
                distorted_x = normalized_x * radial
                    + 2.0F * color_distortion.p1 * normalized_x * normalized_y
                    + color_distortion.p2 * (r2 + 2.0F * normalized_x * normalized_x);
                distorted_y = normalized_y * radial
                    + color_distortion.p1 * (r2 + 2.0F * normalized_y * normalized_y)
                    + 2.0F * color_distortion.p2 * normalized_x * normalized_y;
            }
            const float projected_x = color_camera.fx * distorted_x + color_camera.cx;
            const float projected_y = color_camera.fy * distorted_y + color_camera.cy;
            if(!std::isfinite(projected_x) || !std::isfinite(projected_y)) continue;
            const int source_x = std::clamp(
                static_cast<int>(std::lround(projected_x)), 0, native_color_width - 1);
            const int source_y = std::clamp(
                static_cast<int>(std::lround(projected_y)), 0, native_color_height - 1);
            const std::size_t source = static_cast<std::size_t>(
                (source_y * native_color_width + source_x) * 3);
            aligned_color[target] = native_color[source];
            aligned_color[target + 1] = native_color[source + 1];
            aligned_color[target + 2] = native_color[source + 2];
            if(aligned_color[target] == 0 && aligned_color[target + 1] == 0
                && aligned_color[target + 2] == 0) {
                aligned_color[target] = 96;
                aligned_color[target + 1] = 96;
                aligned_color[target + 2] = 96;
            }
        }
    }
}

std::shared_ptr<ob::Config> femto_video_config(const std::shared_ptr<ob::Pipeline> &pipeline,
                                               const Options &options) {
    auto config = std::make_shared<ob::Config>();
    const auto requested = requested_depth_mode(options);
    auto color_profiles = pipeline->getStreamProfileList(OB_SENSOR_COLOR);
    std::shared_ptr<ob::StreamProfile> best_color;
    std::shared_ptr<ob::StreamProfile> best_depth;
    long best_fps_delta = LONG_MAX;
    long best_pixels = 0;
    for(std::uint32_t color_index = 0; color_index < color_profiles->getCount(); ++color_index) {
        auto color = color_profiles->getProfile(color_index);
        auto color_video = color->as<ob::VideoStreamProfile>();
        if(color_video->getFormat() != OB_FORMAT_MJPG) continue;
        auto depth_profiles = pipeline->getD2CDepthProfileList(color, ALIGN_D2C_HW_MODE);
        for(std::uint32_t depth_index = 0; depth_index < depth_profiles->getCount(); ++depth_index) {
            auto depth = depth_profiles->getProfile(depth_index);
            auto depth_video = depth->as<ob::VideoStreamProfile>();
            if(static_cast<int>(depth_video->getWidth()) != requested.width
                || static_cast<int>(depth_video->getHeight()) != requested.height) continue;
            if(depth_video->getFps() != color_video->getFps()) continue;
            const long fps_delta = std::labs(static_cast<long>(color_video->getFps()) - requested.native_fps);
            const long pixels = static_cast<long>(color_video->getWidth() * color_video->getHeight());
            if(!best_color || fps_delta < best_fps_delta
                || (fps_delta == best_fps_delta && pixels > best_pixels)) {
                best_fps_delta = fps_delta;
                best_pixels = pixels;
                best_color = color;
                best_depth = depth;
            }
        }
    }
    if(!best_color || !best_depth) {
        throw std::runtime_error("Femto Mega has no compatible MJPEG hardware depth-to-color profile for "
            + std::to_string(requested.width) + "x" + std::to_string(requested.height));
    }
    config->enableStream(best_color);
    config->enableStream(best_depth);
    // Keep the native depth sampling grid here. The capture loop first aligns
    // color to that grid, then applies one shared calibrated depth-lens remap
    // to produce the pinhole RGB-D frames used by tracking and TSDF fusion.
    config->setAlignMode(ALIGN_DISABLE);
    config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
    pipeline->enableFrameSync();
    return config;
}

int run_orbbec(const Options &options) {
    // The SDK's default console logger writes to stdout. Stdout is reserved for
    // the binary RGB-D transport, so even one SDK diagnostic corrupts the live
    // stream consumed by the reconstruction worker.
    ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_OFF);
    ob::Context::setLoggerToCallback(
        OB_LOG_SEVERITY_ERROR,
        [](OBLogSeverity, const char *message) {
            if(message) std::cerr << "Orbbec SDK: " << message;
        });

    std::shared_ptr<ob::Context> context;
    std::shared_ptr<ob::Device> device;
    std::shared_ptr<ob::Pipeline> pipeline;
    if(options.connection == "network") {
        context = std::make_shared<ob::Context>();
        const auto [host, port] = network_endpoint(options.address);
        device = context->createNetDevice(host.c_str(), port);
        pipeline = std::make_shared<ob::Pipeline>(device);
    } else {
        context = std::make_shared<ob::Context>();
        auto devices = context->queryDeviceList();
        for(std::uint32_t index = 0; index < devices->getCount(); ++index) {
            if(std::string(devices->getName(index)).find("Femto Mega") == std::string::npos) continue;
            const std::string raw_connection = devices->getConnectionType(index);
            if(raw_connection.find("Ethernet") != std::string::npos) continue;
            const std::string candidate_id = orbbec_device_id(
                "usb", devices->getSerialNumber(index), "", devices->getUid(index));
            if(!options.device_id.empty() && options.device_id != candidate_id) continue;
            device = devices->getDevice(index);
            break;
        }
        if(!device) throw std::runtime_error(options.device_id.empty()
            ? "Orbbec Femto Mega was not found over USB"
            : "The selected Orbbec Femto Mega is no longer connected over USB");
        pipeline = std::make_shared<ob::Pipeline>(device);
    }
    const auto info = device->getDeviceInfo();
    SensorInfo sensor{"femto_mega", info->getName(), options.connection, info->getSerialNumber(), options.address};
    if(sensor.name.find("Femto Mega") == std::string::npos) {
        throw std::runtime_error("Selected Orbbec device is not a Femto Mega: " + sensor.name);
    }
    auto config = femto_video_config(pipeline, options);
    pipeline->start(config);
    auto first = pipeline->waitForFrameset(5000);
    if(!first || !first->getDepthFrame()) throw std::runtime_error("Femto Mega did not deliver a depth frame");
    if(options.probe) {
        pipeline->stop();
        std::cout << sensor.name << " ready over " << options.connection << '\n';
        return 0;
    }

    prepare_root(options);
    auto aligned_first = first;
    auto first_depth = aligned_first->getDepthFrame();
    auto depth_profile = first_depth->getStreamProfile();
    const auto video_profile = depth_profile->as<ob::VideoStreamProfile>();
    OrbbecDepthRectifier depth_rectifier(
        pipeline, config, static_cast<int>(first_depth->getWidth()), static_cast<int>(first_depth->getHeight()));
    const CameraInfo camera = depth_rectifier.camera();
    auto first_color = aligned_first->getColorFrame();
    if(!first_color) throw std::runtime_error("Femto Mega did not deliver synchronized native RGB");
    auto color_profile = first_color->getStreamProfile()->as<ob::VideoStreamProfile>();
    const auto color_intrinsic = color_profile->getIntrinsic();
    const auto color_distortion = color_profile->getDistortion();
    const auto [rgb_model, rgb_distortion] = orbbec_rgb_distortion(color_distortion);
    const auto depth_to_rgb = depth_profile->getExtrinsicTo(color_profile);
    const auto rigid_depth_to_rgb = rigid_orbbec_rotation(depth_to_rgb.rot);
    RgbCameraInfo rgb_camera{
        static_cast<int>(first_color->getWidth()),
        static_cast<int>(first_color->getHeight()),
        color_intrinsic.fx,
        color_intrinsic.fy,
        color_intrinsic.cx,
        color_intrinsic.cy,
        rgb_model,
        rgb_distortion,
        {
            rigid_depth_to_rgb[0], rigid_depth_to_rgb[1], rigid_depth_to_rgb[2], depth_to_rgb.trans[0] / 1000.0F,
            rigid_depth_to_rgb[3], rigid_depth_to_rgb[4], rigid_depth_to_rgb[5], depth_to_rgb.trans[1] / 1000.0F,
            rigid_depth_to_rgb[6], rigid_depth_to_rgb[7], rigid_depth_to_rgb[8], depth_to_rgb.trans[2] / 1000.0F,
            0, 0, 0, 1
        }
    };
    std::string created_at = utc_now();
    scanlan::RgbdArchiveWriter archive(
        options.root, options.rgb_quality, options.max_rgb_dimension, 8);

    std::unique_ptr<ImuCsv> imu;
    scanlan::GyroDeltaIntegrator gyro_integrator;
    std::shared_ptr<ob::Sensor> accel_sensor;
    std::shared_ptr<ob::Sensor> gyro_sensor;
    bool imu_active = false;
    std::atomic<bool> recording{!options.preview};
    if(options.use_imu) {
        try {
            imu = std::make_unique<ImuCsv>(options.root / "imu.csv");
            accel_sensor = device->getSensor(OB_SENSOR_ACCEL);
            gyro_sensor = device->getSensor(OB_SENSOR_GYRO);
            if(!accel_sensor || !gyro_sensor) throw std::runtime_error("Femto Mega IMU sensors are unavailable");
            ImuCsv *writer = imu.get();
            auto target_profile = depth_profile;
            auto accel_profile = accel_sensor->getStreamProfileList()->getProfile(0);
            auto gyro_profile = gyro_sensor->getStreamProfileList()->getProfile(0);
            accel_sensor->start(accel_profile, [writer, target_profile, &recording](std::shared_ptr<ob::Frame> raw) {
                auto frame = raw->as<ob::AccelFrame>();
                const auto value = rotate_ob(frame->getStreamProfile()->getExtrinsicTo(target_profile), frame->getValue());
                if(recording.load()) writer->write(frame->getTimeStampUs(), "accel", value[0], value[1], value[2], frame->getTemperature());
            });
            gyro_sensor->start(gyro_profile, [writer, target_profile, &gyro_integrator, &recording](std::shared_ptr<ob::Frame> raw) {
                auto frame = raw->as<ob::GyroFrame>();
                const auto value = rotate_ob(frame->getStreamProfile()->getExtrinsicTo(target_profile), frame->getValue());
                if(recording.load()) writer->write(frame->getTimeStampUs(), "gyro", value[0], value[1], value[2], frame->getTemperature());
                gyro_integrator.add(frame->getTimeStampUs(), value[0], value[1], value[2]);
            });
            imu_active = true;
        } catch(const ob::Error &) {
            if(accel_sensor) { try { accel_sensor->stop(); } catch(...) {} }
            if(gyro_sensor) { try { gyro_sensor->stop(); } catch(...) {} }
            imu_active = false;
            imu.reset();
            accel_sensor.reset();
            gyro_sensor.reset();
        }
    }
    write_manifest(options, sensor, camera, rgb_camera, created_at, 0, 0, imu_active, 0);

    std::unique_ptr<scanlan::RgbdStreamWriter> rgbd_stream;
    if(options.stream_rgbd) {
        rgbd_stream = std::make_unique<scanlan::RgbdStreamWriter>(std::cout, 3);
    }

    auto format_converter = std::make_shared<ob::FormatConvertFilter>();
    auto color_to_depth = std::make_shared<ob::Align>(OB_STREAM_DEPTH);
    const int native_fps = std::max(1, static_cast<int>(video_profile->getFps()));
    std::uint64_t source_frames = 0;
    std::uint64_t recording_frames = 0;
    const auto started = std::chrono::steady_clock::now();
    auto recording_started = started;
    auto frame_set = aligned_first;
    while(!fs::exists(options.root / "stop.flag")) {
        if(!frame_set) frame_set = pipeline->waitForFrameset(1000);
        if(!frame_set) continue;
        auto depth_frame = frame_set->getDepthFrame();
        if(!depth_frame) { frame_set.reset(); continue; }
        ++source_frames;
        if(!recording.load() && fs::exists(options.root / "record.flag")) {
            recording.store(true);
            recording_started = std::chrono::steady_clock::now();
            created_at = utc_now();
            {
                std::ofstream signal(options.root / "recording.flag");
                signal << (source_frames - 1) << '\n';
            }
            fs::remove(options.root / "preview.flag");
            fs::remove(options.root / "record.flag");
        }
        if(recording.load()) ++recording_frames;
        const bool save_source = recording.load() && scanlan::archive_frame_due(
            recording_frames, native_fps, options.fps);
        const float scale = depth_frame->getValueScale();
        const auto *raw_depth = reinterpret_cast<const std::uint16_t *>(depth_frame->getData());
        std::vector<std::uint16_t> native_depth(static_cast<std::size_t>(camera.width * camera.height));
        for(std::size_t index = 0; index < native_depth.size(); ++index) {
            const float millimetres = raw_depth[index] * scale;
            native_depth[index] = millimetres > 0 && millimetres <= options.max_depth_m * 1000.0F
                ? static_cast<std::uint16_t>(std::lround(millimetres)) : 0;
        }
        const std::uint64_t timestamp_us = depth_frame->getTimeStampUs();
        const double elapsed = std::max(0.001, std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        const float stream_fps = static_cast<float>(source_frames / elapsed);
        auto color_frame = frame_set->getColorFrame();
        if(!color_frame) { frame_set.reset(); continue; }
        if(color_frame->getFormat() != OB_FORMAT_RGB) {
            if(color_frame->getFormat() == OB_FORMAT_MJPG) format_converter->setFormatConvertType(FORMAT_MJPG_TO_RGB);
            else if(color_frame->getFormat() == OB_FORMAT_UYVY) format_converter->setFormatConvertType(FORMAT_UYVY_TO_RGB);
            else if(color_frame->getFormat() == OB_FORMAT_YUYV) format_converter->setFormatConvertType(FORMAT_YUYV_TO_RGB);
            else throw std::runtime_error("Femto Mega returned an unsupported color format");
            auto converted = format_converter->process(color_frame);
            if(!converted) throw std::runtime_error("Femto Mega RGB conversion failed");
            color_frame = converted->as<ob::ColorFrame>();
        }
        const auto *rgb_data = color_frame->getData();
        std::vector<std::uint8_t> native_rgb(
            rgb_data,
            rgb_data + color_frame->getWidth() * color_frame->getHeight() * 3);

        // Align only after replacing the camera's MJPEG frame with its RGB
        // conversion. Orbbec SDK 2.9.x does not support MJPEG C2D input and
        // returns a null frame; dereferencing that result previously caused an
        // access violation during the first captured frame.
        auto align_input = ob::FrameFactory::createFrameSet();
        align_input->pushFrame(depth_frame);
        align_input->pushFrame(color_frame);
        auto aligned = color_to_depth->process(align_input);
        if(!aligned) throw std::runtime_error("Femto Mega color-to-depth alignment failed");
        auto aligned_set = aligned->as<ob::FrameSet>();
        if(!aligned_set) throw std::runtime_error("Femto Mega color-to-depth alignment returned an invalid frame set");
        auto aligned_color = aligned_set->getColorFrame();
        if(!aligned_color) throw std::runtime_error("Femto Mega aligned color frame is missing");
        if(aligned_color->getFormat() != OB_FORMAT_RGB) {
            if(aligned_color->getFormat() == OB_FORMAT_MJPG) format_converter->setFormatConvertType(FORMAT_MJPG_TO_RGB);
            else if(aligned_color->getFormat() == OB_FORMAT_UYVY) format_converter->setFormatConvertType(FORMAT_UYVY_TO_RGB);
            else if(aligned_color->getFormat() == OB_FORMAT_YUYV) format_converter->setFormatConvertType(FORMAT_YUYV_TO_RGB);
            else throw std::runtime_error("Femto Mega aligned color has an unsupported format");
            aligned_color = format_converter->process(aligned_color)->as<ob::ColorFrame>();
        }
        if(static_cast<int>(aligned_color->getWidth()) != camera.width
            || static_cast<int>(aligned_color->getHeight()) != camera.height) {
            throw std::runtime_error("Femto Mega C2D output does not match native depth geometry");
        }
        const auto *aligned_data = aligned_color->getData();
        std::vector<std::uint8_t> native_aligned_rgb(
            aligned_data,
            aligned_data + camera.width * camera.height * 3);
        auto depth = depth_rectifier.rectify_depth(native_depth);
        auto rgb = depth_rectifier.rectify_color(native_aligned_rgb);
        if(options.depth_fov == "narrow") {
            complete_orbbec_narrow_color(
                camera,
                color_intrinsic,
                color_distortion,
                depth_to_rgb,
                rigid_depth_to_rgb,
                depth,
                native_rgb,
                static_cast<int>(color_frame->getWidth()),
                static_cast<int>(color_frame->getHeight()),
                rgb);
        }
        const std::uint64_t rgb_timestamp_us = color_frame->getTimeStampUs();
        if(rgbd_stream) {
            const auto gyro_delta = imu_active ? gyro_integrator.take() : std::nullopt;
            rgbd_stream->publish(
                source_frames - 1,
                timestamp_us,
                rgb_timestamp_us,
                static_cast<std::uint32_t>(camera.width),
                static_cast<std::uint32_t>(camera.height),
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                1000.0F,
                0.25F,
                options.max_depth_m,
                depth,
                rgb,
                gyro_delta ? &*gyro_delta : nullptr);
        }
        if(save_source) {
            scanlan::RgbdArchiveFrame archive_frame;
            archive_frame.source_sequence = source_frames - 1;
            archive_frame.depth_timestamp_us = timestamp_us;
            archive_frame.color_timestamp_us = rgb_timestamp_us;
            archive_frame.depth = std::move(depth);
            archive_frame.aligned_color = std::move(rgb);
            archive_frame.native_color_width =
                static_cast<std::uint32_t>(color_frame->getWidth());
            archive_frame.native_color_height =
                static_cast<std::uint32_t>(color_frame->getHeight());
            archive_frame.native_color = std::move(native_rgb);
            archive.submit(std::move(archive_frame));
        }
        if(source_frames == 1 || source_frames % 3 == 0) {
            write_live_status(options, sensor, timestamp_us, archive.saved(),
                              stream_fps, imu_active, imu ? imu->rate_hz() : 0.0F,
                              recording.load()
                                  ? "Full-rate RGB-D tracking and bounded background recording"
                                  : "Live RGB-D preview ready; press Capture to begin recording");
        }
        frame_set.reset();
    }
    if(imu_active) {
        accel_sensor->stop();
        gyro_sensor->stop();
    }
    if(rgbd_stream) rgbd_stream->close();
    const auto duration = recording_frames == 0 ? 0U : static_cast<std::uint32_t>(std::max<std::int64_t>(1,
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - recording_started).count()));
    archive.close();
    const auto saved_frames = archive.saved();
    write_manifest(options, sensor, camera, rgb_camera, created_at, saved_frames, duration, imu_active,
                   archive.dropped() + archive.failed());
    pipeline->stop();
    return saved_frames > 0 || options.preview ? 0 : 2;
}
#else
int run_orbbec(const Options &) {
    throw std::runtime_error("Orbbec Femto Mega support was not compiled; install Orbbec SDK v2, set ORBBEC_SDK_ROOT, and rebuild");
}
#endif

int print_capabilities() {
    std::cout << '[';
#ifdef SCANLAN_HAS_AZURE_KINECT
    std::cout << "\"azure_kinect\"";
#endif
#ifdef SCANLAN_HAS_ORBBEC
#ifdef SCANLAN_HAS_AZURE_KINECT
    std::cout << ',';
#endif
    std::cout << "\"femto_mega\"";
#endif
    std::cout << "]\n";
    return 0;
}

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        if(options.capabilities) return print_capabilities();
        if(options.list) return list_sensors();
        if(options.stream_rgbd) _setmode(_fileno(stdout), _O_BINARY);
        return options.sensor == "femto_mega" ? run_orbbec(options) : run_azure(options);
#ifdef SCANLAN_HAS_ORBBEC
    } catch(const ob::Error &error) {
        std::cerr << "rgbd-capture-worker: " << error.what() << " (" << error.getFunction() << ")\n";
        return 1;
#endif
    } catch(const std::exception &error) {
        std::cerr << "rgbd-capture-worker: " << error.what() << '\n';
        return 1;
    }
}
