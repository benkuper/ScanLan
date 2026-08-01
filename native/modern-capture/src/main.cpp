#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <io.h>

#ifdef SCANLAN_HAS_AZURE_KINECT
#include <k4a/k4a.h>
#endif

#ifdef SCANLAN_HAS_ORBBEC
#include <libobsensor/ObSensor.hpp>
#endif

namespace fs = std::filesystem;

struct Options {
    bool probe = false;
    bool list = false;
    bool preview = false;
    bool use_imu = false;
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

bool should_save_source_frame(std::uint64_t source_frame, int native_fps, int requested_fps) {
    if(source_frame <= 1) return true;
    const auto target_fps = std::min(native_fps, requested_fps);
    const auto current_bucket = (source_frame - 1) * static_cast<std::uint64_t>(target_fps)
        / static_cast<std::uint64_t>(native_fps);
    const auto previous_bucket = (source_frame - 2) * static_cast<std::uint64_t>(target_fps)
        / static_cast<std::uint64_t>(native_fps);
    return current_bucket != previous_bucket;
}

struct CameraInfo {
    int width = 0;
    int height = 0;
    float fx = 0;
    float fy = 0;
    float cx = 0;
    float cy = 0;
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
        if(argument == "--probe") options.probe = true;
        else if(argument == "--list") options.list = true;
        else if(argument == "--preview") { options.preview = true; options.root = value(); }
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
        else if(argument == "--imu") options.use_imu = true;
        else throw std::runtime_error("Unknown argument: " + argument);
    }
    if(!options.probe && !options.list && options.root.empty()) throw std::runtime_error("Pass --phase PATH or --preview PATH");
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
                  << "\",\"supportsImu\":" << (sensor.supports_imu ? "true" : "false") << '}';
    }
    std::cout << "]\n";
}

int list_sensors() {
    std::vector<AvailableSensorInfo> sensors;
#ifdef SCANLAN_HAS_AZURE_KINECT
    const auto azure_count = k4a_device_get_installed_count();
    for(std::uint32_t index = 0; index < azure_count; ++index) {
        std::string serial;
        k4a_device_t device = nullptr;
        if(k4a_device_open(index, &device) == K4A_RESULT_SUCCEEDED) {
            std::size_t serial_size = 0;
            if(k4a_device_get_serialnum(device, nullptr, &serial_size) == K4A_BUFFER_RESULT_TOO_SMALL && serial_size > 0) {
                std::vector<char> buffer(serial_size);
                if(k4a_device_get_serialnum(device, buffer.data(), &serial_size) == K4A_BUFFER_RESULT_SUCCEEDED) {
                    serial = buffer.data();
                }
            }
            k4a_device_close(device);
        }
        sensors.push_back({azure_device_id(serial, index), "azure_kinect", "Azure Kinect DK", "usb", "", serial, true});
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
    }
    std::error_code error;
    fs::remove(path, error);
    for(int attempt = 0; attempt < 40; ++attempt) {
        error.clear();
        fs::rename(temporary, path, error);
        if(!error) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(5 + attempt));
    }
    throw std::runtime_error("Could not publish " + path.string() + ": " + error.message());
}

template<typename T>
void write_binary(const fs::path &path, const std::vector<T> &values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if(!output) throw std::runtime_error("Could not write " + path.string());
    output.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
}

void write_manifest(const Options &options, const SensorInfo &sensor, const CameraInfo &camera,
                    const std::string &created_at, std::uint32_t frame_count, std::uint32_t duration_seconds,
                    bool imu_active) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"schemaVersion\": 2,\n"
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
           << "  }\n"
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

void append_packet_float(std::vector<std::uint8_t> &packet, float value) {
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(&value);
    packet.insert(packet.end(), bytes, bytes + sizeof(value));
}

void write_preview_packet(const std::vector<std::uint16_t> &depth, const CameraInfo &camera,
                          std::uint32_t frame_count, std::uint64_t timestamp_us, float stream_fps) {
    const int stride = std::max(2, static_cast<int>(std::ceil(std::sqrt(
        static_cast<double>(camera.width * camera.height) / 12000.0))));
    std::vector<std::uint8_t> packet(24, 0);
    std::uint32_t count = 0;
    for(int y = 0; y < camera.height; y += stride) {
        for(int x = 0; x < camera.width; x += stride) {
            const std::uint16_t depth_mm = depth[static_cast<std::size_t>(y * camera.width + x)];
            if(depth_mm == 0) continue;
            const float z = depth_mm / 1000.0F;
            // Azure Kinect and Femto Mega depth images already use the expected
            // left/right orientation. Only the Kinect v2 backend needs an X mirror.
            append_packet_float(packet, (x - camera.cx) * z / camera.fx);
            append_packet_float(packet, -(y - camera.cy) * z / camera.fy);
            append_packet_float(packet, -z);
            const float normalized = std::clamp((z - 0.5F) / 4.0F, 0.0F, 1.0F);
            packet.push_back(static_cast<std::uint8_t>(72 + normalized * 58));
            packet.push_back(static_cast<std::uint8_t>(190 - normalized * 52));
            packet.push_back(static_cast<std::uint8_t>(220 - normalized * 35));
            ++count;
        }
    }
    std::memcpy(packet.data(), "K2P1", 4);
    std::memcpy(packet.data() + 4, &frame_count, sizeof(frame_count));
    std::memcpy(packet.data() + 8, &timestamp_us, sizeof(timestamp_us));
    std::memcpy(packet.data() + 16, &stream_fps, sizeof(stream_fps));
    std::memcpy(packet.data() + 20, &count, sizeof(count));
    std::cout.write(reinterpret_cast<const char *>(packet.data()), static_cast<std::streamsize>(packet.size()));
    std::cout.flush();
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
    fs::remove(options.root / "live.json");
    if(!options.preview) {
        fs::create_directories(options.root / "depth");
        fs::create_directories(options.root / "color");
    }
}

std::string csv_header() {
    std::ostringstream output;
    output << "index,timestamp_us,depth_path,color_path";
    for(int row = 0; row < 4; ++row) for(int column = 0; column < 4; ++column) output << ",m" << row << column;
    output << '\n';
    return output.str();
}

void save_frame(const Options &options, std::ofstream &frames, std::uint32_t index, std::uint64_t timestamp_us,
                const std::vector<std::uint16_t> &depth, const std::vector<std::uint8_t> &rgb) {
    std::ostringstream stem;
    stem << std::setw(6) << std::setfill('0') << index;
    const std::string depth_name = stem.str() + ".u16";
    const std::string color_name = stem.str() + ".rgb";
    write_binary(options.root / "depth" / depth_name, depth);
    write_binary(options.root / "color" / color_name, rgb);
    frames << index << ',' << timestamp_us << ",depth/" << depth_name << ",color/" << color_name;
    for(int value = 0; value < 16; ++value) frames << ',';
    frames << '\n';
    frames.flush();
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
        if(options.device_id.empty() || options.device_id == candidate_id) {
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
        config.synchronized_images_only = !options.preview && !options.probe;
        if(!options.preview) {
            config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
            config.color_resolution = K4A_COLOR_RESOLUTION_720P;
        }
        k4a_calibration_t calibration{};
        if(k4a_device_get_calibration(device, config.depth_mode, config.color_resolution, &calibration) != K4A_RESULT_SUCCEEDED) {
            throw std::runtime_error("Azure Kinect calibration is unavailable");
        }
        const auto &intrinsics = calibration.depth_camera_calibration.intrinsics.parameters.param;
        CameraInfo camera{calibration.depth_camera_calibration.resolution_width,
                          calibration.depth_camera_calibration.resolution_height,
                          intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy};
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
        const std::string created_at = utc_now();
        std::ofstream frames;
        if(!options.preview) {
            frames.open(options.root / "frames.csv", std::ios::trunc);
            frames << csv_header();
        }
        std::unique_ptr<ImuCsv> imu;
        bool imu_active = false;
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
                imu->write(sample.acc_timestamp_usec, "accel", accel[0], accel[1], accel[2], sample.temperature);
                imu->write(sample.gyro_timestamp_usec, "gyro", gyro[0], gyro[1], gyro[2], sample.temperature);
            }
        };
        write_manifest(options, sensor, camera, created_at, 0, 0, imu_active);

        k4a_transformation_t transformation = options.preview ? nullptr : k4a_transformation_create(&calibration);
        std::uint64_t source_frames = 0;
        std::uint32_t saved_frames = 0;
        const auto started = std::chrono::steady_clock::now();
        while(!fs::exists(options.root / "stop.flag")) {
            k4a_capture_t capture = nullptr;
            const auto wait = k4a_device_get_capture(device, &capture, 1000);
            if(wait == K4A_WAIT_RESULT_TIMEOUT) continue;
            if(wait != K4A_WAIT_RESULT_SUCCEEDED || !capture) throw std::runtime_error("Azure Kinect stopped delivering frames");
            drain_imu();
            k4a_image_t depth_image = k4a_capture_get_depth_image(capture);
            if(!depth_image) { k4a_capture_release(capture); continue; }
            ++source_frames;
            if(!should_save_source_frame(source_frames, depth_mode.native_fps, options.fps)) {
                k4a_image_release(depth_image);
                k4a_capture_release(capture);
                continue;
            }
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
            ++saved_frames;
            const double elapsed = std::max(0.001, std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
            const float stream_fps = static_cast<float>(saved_frames / elapsed);
            if(options.preview) {
                write_preview_packet(depth, camera, saved_frames, timestamp_us, stream_fps);
            } else {
                k4a_image_t color_image = k4a_capture_get_color_image(capture);
                if(!color_image) {
                    --saved_frames;
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
                save_frame(options, frames, saved_frames - 1, timestamp_us, depth, rgb);
                k4a_image_release(aligned);
                k4a_image_release(color_image);
            }
            write_live_status(options, sensor, timestamp_us, saved_frames, stream_fps, imu_active,
                              imu ? imu->rate_hz() : 0.0F,
                              imu_active ? "Recording calibrated IMU; RGB-D odometry runs during Build" : "RGB-D odometry runs during Build");
            k4a_image_release(depth_image);
            k4a_capture_release(capture);
        }
        if(transformation) k4a_transformation_destroy(transformation);
        if(imu_active) {
            drain_imu();
            k4a_device_stop_imu(device);
        }
        const auto duration = static_cast<std::uint32_t>(std::max<std::int64_t>(1,
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - started).count()));
        write_manifest(options, sensor, camera, created_at, saved_frames, duration, imu_active);
        k4a_device_stop_cameras(device);
        k4a_device_close(device);
        return options.preview || saved_frames > 0 ? 0 : 2;
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

std::shared_ptr<ob::Config> femto_video_config(const std::shared_ptr<ob::Pipeline> &pipeline,
                                               const Options &options) {
    auto config = std::make_shared<ob::Config>();
    const auto requested = requested_depth_mode(options);
    if(options.preview) {
        auto depth_profiles = pipeline->getStreamProfileList(OB_SENSOR_DEPTH);
        std::shared_ptr<ob::StreamProfile> best_depth;
        long best_score = LONG_MAX;
        for(std::uint32_t index = 0; index < depth_profiles->getCount(); ++index) {
            auto depth = depth_profiles->getProfile(index);
            auto video = depth->as<ob::VideoStreamProfile>();
            if(static_cast<int>(video->getWidth()) != requested.width
                || static_cast<int>(video->getHeight()) != requested.height) continue;
            const long score = std::labs(static_cast<long>(video->getFps()) - requested.native_fps);
            if(score < best_score) {
                best_score = score;
                best_depth = depth;
            }
        }
        if(!best_depth) {
            throw std::runtime_error("Femto Mega does not expose the selected "
                + std::to_string(requested.width) + "x" + std::to_string(requested.height) + " depth mode");
        }
        config->enableStream(best_depth);
        config->setDepthScaleRequire(true);
        return config;
    }

    auto color_profiles = pipeline->getStreamProfileList(OB_SENSOR_COLOR);
    std::shared_ptr<ob::StreamProfile> best_color;
    std::shared_ptr<ob::StreamProfile> best_depth;
    long best_score = LONG_MAX;
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
            const long score = std::labs(static_cast<long>(color_video->getWidth()) - 1280L)
                + std::labs(static_cast<long>(color_video->getHeight()) - 720L)
                + std::labs(static_cast<long>(color_video->getFps()) - requested.native_fps) * 100L;
            if(score < best_score) {
                best_score = score;
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
    config->setAlignMode(ALIGN_D2C_HW_MODE);
    config->setDepthScaleRequire(true);
    config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
    pipeline->enableFrameSync();
    return config;
}

int run_orbbec(const Options &options) {
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
    const auto intrinsic = video_profile->getIntrinsic();
    CameraInfo camera{static_cast<int>(first_depth->getWidth()), static_cast<int>(first_depth->getHeight()),
                      intrinsic.fx, intrinsic.fy, intrinsic.cx, intrinsic.cy};
    const std::string created_at = utc_now();
    std::ofstream frames;
    if(!options.preview) {
        frames.open(options.root / "frames.csv", std::ios::trunc);
        frames << csv_header();
    }

    std::unique_ptr<ImuCsv> imu;
    std::shared_ptr<ob::Sensor> accel_sensor;
    std::shared_ptr<ob::Sensor> gyro_sensor;
    bool imu_active = false;
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
            accel_sensor->start(accel_profile, [writer, target_profile](std::shared_ptr<ob::Frame> raw) {
                auto frame = raw->as<ob::AccelFrame>();
                const auto value = rotate_ob(frame->getStreamProfile()->getExtrinsicTo(target_profile), frame->getValue());
                writer->write(frame->getTimeStampUs(), "accel", value[0], value[1], value[2], frame->getTemperature());
            });
            gyro_sensor->start(gyro_profile, [writer, target_profile](std::shared_ptr<ob::Frame> raw) {
                auto frame = raw->as<ob::GyroFrame>();
                const auto value = rotate_ob(frame->getStreamProfile()->getExtrinsicTo(target_profile), frame->getValue());
                writer->write(frame->getTimeStampUs(), "gyro", value[0], value[1], value[2], frame->getTemperature());
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
    write_manifest(options, sensor, camera, created_at, 0, 0, imu_active);

    auto format_converter = std::make_shared<ob::FormatConvertFilter>();
    const int native_fps = std::max(1, static_cast<int>(video_profile->getFps()));
    std::uint64_t source_frames = 0;
    std::uint32_t saved_frames = 0;
    const auto started = std::chrono::steady_clock::now();
    auto frame_set = aligned_first;
    while(!fs::exists(options.root / "stop.flag")) {
        if(!frame_set) frame_set = pipeline->waitForFrameset(1000);
        if(!frame_set) continue;
        auto depth_frame = frame_set->getDepthFrame();
        if(!depth_frame) { frame_set.reset(); continue; }
        ++source_frames;
        if(!should_save_source_frame(source_frames, native_fps, options.fps)) { frame_set.reset(); continue; }
        const float scale = depth_frame->getValueScale();
        const auto *raw_depth = reinterpret_cast<const std::uint16_t *>(depth_frame->getData());
        std::vector<std::uint16_t> depth(static_cast<std::size_t>(camera.width * camera.height));
        for(std::size_t index = 0; index < depth.size(); ++index) {
            const float millimetres = raw_depth[index] * scale;
            depth[index] = millimetres > 0 && millimetres <= options.max_depth_m * 1000.0F
                ? static_cast<std::uint16_t>(std::lround(millimetres)) : 0;
        }
        const std::uint64_t timestamp_us = depth_frame->getTimeStampUs();
        ++saved_frames;
        const double elapsed = std::max(0.001, std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        const float stream_fps = static_cast<float>(saved_frames / elapsed);
        if(options.preview) {
            write_preview_packet(depth, camera, saved_frames, timestamp_us, stream_fps);
        } else {
            auto color_frame = frame_set->getColorFrame();
            if(!color_frame) { --saved_frames; frame_set.reset(); continue; }
            if(color_frame->getFormat() != OB_FORMAT_RGB) {
                if(color_frame->getFormat() == OB_FORMAT_MJPG) format_converter->setFormatConvertType(FORMAT_MJPG_TO_RGB);
                else if(color_frame->getFormat() == OB_FORMAT_UYVY) format_converter->setFormatConvertType(FORMAT_UYVY_TO_RGB);
                else if(color_frame->getFormat() == OB_FORMAT_YUYV) format_converter->setFormatConvertType(FORMAT_YUYV_TO_RGB);
                else throw std::runtime_error("Femto Mega returned an unsupported color format");
                color_frame = format_converter->process(color_frame)->as<ob::ColorFrame>();
            }
            if(static_cast<int>(color_frame->getWidth()) != camera.width || static_cast<int>(color_frame->getHeight()) != camera.height) {
                throw std::runtime_error("Femto Mega aligned depth/color dimensions do not match");
            }
            const auto *rgb_data = color_frame->getData();
            std::vector<std::uint8_t> rgb(rgb_data, rgb_data + camera.width * camera.height * 3);
            save_frame(options, frames, saved_frames - 1, timestamp_us, depth, rgb);
        }
        write_live_status(options, sensor, timestamp_us, saved_frames, stream_fps, imu_active,
                          imu ? imu->rate_hz() : 0.0F,
                          imu_active ? "Recording calibrated IMU; RGB-D odometry runs during Build" : "RGB-D odometry runs during Build");
        frame_set.reset();
    }
    if(imu_active) {
        accel_sensor->stop();
        gyro_sensor->stop();
    }
    const auto duration = static_cast<std::uint32_t>(std::max<std::int64_t>(1,
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - started).count()));
    write_manifest(options, sensor, camera, created_at, saved_frames, duration, imu_active);
    pipeline->stop();
    return options.preview || saved_frames > 0 ? 0 : 2;
}
#else
int run_orbbec(const Options &) {
    throw std::runtime_error("Orbbec Femto Mega support was not compiled; install Orbbec SDK v2, set ORBBEC_SDK_ROOT, and rebuild");
}
#endif

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        if(options.list) return list_sensors();
        if(options.preview) _setmode(_fileno(stdout), _O_BINARY);
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
