#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace scanlan {

inline constexpr std::array<char, 8> RgbdStreamMagic{'S', 'C', 'A', 'N', 'R', 'G', 'B', 'D'};
inline constexpr std::uint16_t RgbdStreamVersion = 1;

enum RgbdFrameFlags : std::uint32_t {
    RgbdHasColor = 1U << 0U,
    RgbdHasImuDelta = 1U << 1U,
    RgbdHasCameraPose = 1U << 2U,
    RgbdMirrorX = 1U << 3U,
};

#pragma pack(push, 1)
struct RgbdFrameHeader {
    char magic[8];
    std::uint16_t version;
    std::uint16_t header_bytes;
    std::uint32_t flags;
    std::uint64_t sequence;
    std::uint64_t depth_timestamp_us;
    std::uint64_t color_timestamp_us;
    std::uint32_t width;
    std::uint32_t height;
    float fx;
    float fy;
    float cx;
    float cy;
    float depth_scale;
    float min_depth_m;
    float max_depth_m;
    float gyro_delta_xyzw[4];
    float camera_to_world[16];
    std::uint32_t depth_bytes;
    std::uint32_t color_bytes;
};
#pragma pack(pop)

static_assert(sizeof(RgbdFrameHeader) == 164, "RGB-D stream header layout changed");

struct RgbdStreamFrame {
    RgbdFrameHeader header{};
    std::vector<std::uint16_t> depth;
    std::vector<std::uint8_t> color;
};

inline std::array<float, 16> identity_pose() {
    return {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
}

// Writes full-rate RGB-D frames without ever blocking the sensor thread. If the
// consumer falls behind, stale unpublished frames are discarded and the newest
// frame wins. Tracking can report these sequence gaps explicitly instead of
// silently accumulating seconds of latency.
class RgbdStreamWriter {
public:
    explicit RgbdStreamWriter(std::ostream &output, std::size_t capacity = 3)
        : output_(output), capacity_(capacity) {
        if(capacity_ == 0) throw std::invalid_argument("RGB-D stream capacity must be positive");
        thread_ = std::thread([this] { run(); });
    }

    RgbdStreamWriter(const RgbdStreamWriter &) = delete;
    RgbdStreamWriter &operator=(const RgbdStreamWriter &) = delete;

    ~RgbdStreamWriter() { close(); }

    void publish(
        std::uint64_t sequence,
        std::uint64_t depth_timestamp_us,
        std::uint64_t color_timestamp_us,
        std::uint32_t width,
        std::uint32_t height,
        float fx,
        float fy,
        float cx,
        float cy,
        float depth_scale,
        float min_depth_m,
        float max_depth_m,
        std::vector<std::uint16_t> depth,
        std::vector<std::uint8_t> color,
        const std::array<float, 4> *gyro_delta_xyzw = nullptr,
        const std::array<float, 16> *camera_to_world = nullptr,
        bool mirror_x = false) {
        const auto pixel_count = static_cast<std::size_t>(width) * height;
        if(width == 0 || height == 0 || depth.size() != pixel_count) {
            throw std::invalid_argument("RGB-D stream depth dimensions are invalid");
        }
        if(!color.empty() && color.size() != pixel_count * 3) {
            throw std::invalid_argument("RGB-D stream color dimensions are invalid");
        }

        RgbdStreamFrame frame;
        std::memcpy(frame.header.magic, RgbdStreamMagic.data(), RgbdStreamMagic.size());
        frame.header.version = RgbdStreamVersion;
        frame.header.header_bytes = sizeof(RgbdFrameHeader);
        frame.header.flags = color.empty() ? 0U : RgbdHasColor;
        frame.header.sequence = sequence;
        frame.header.depth_timestamp_us = depth_timestamp_us;
        frame.header.color_timestamp_us = color_timestamp_us;
        frame.header.width = width;
        frame.header.height = height;
        frame.header.fx = fx;
        frame.header.fy = fy;
        frame.header.cx = cx;
        frame.header.cy = cy;
        frame.header.depth_scale = depth_scale;
        frame.header.min_depth_m = min_depth_m;
        frame.header.max_depth_m = max_depth_m;
        const std::array<float, 4> identity_quaternion{0, 0, 0, 1};
        const auto &quaternion = gyro_delta_xyzw ? *gyro_delta_xyzw : identity_quaternion;
        std::copy(quaternion.begin(), quaternion.end(), frame.header.gyro_delta_xyzw);
        if(gyro_delta_xyzw) frame.header.flags |= RgbdHasImuDelta;
        const auto identity = identity_pose();
        const auto &pose = camera_to_world ? *camera_to_world : identity;
        std::copy(pose.begin(), pose.end(), frame.header.camera_to_world);
        if(camera_to_world) frame.header.flags |= RgbdHasCameraPose;
        if(mirror_x) frame.header.flags |= RgbdMirrorX;
        frame.header.depth_bytes = static_cast<std::uint32_t>(depth.size() * sizeof(std::uint16_t));
        frame.header.color_bytes = static_cast<std::uint32_t>(color.size());
        frame.depth = std::move(depth);
        frame.color = std::move(color);

        {
            std::lock_guard lock(mutex_);
            if(closed_) return;
            if(queue_.size() == capacity_) {
                queue_.pop_front();
                ++dropped_;
            }
            queue_.push_back(std::move(frame));
        }
        ready_.notify_one();
    }

    void close() {
        bool notify = false;
        {
            std::lock_guard lock(mutex_);
            if(!closed_) {
                closed_ = true;
                notify = true;
            }
        }
        if(notify) ready_.notify_all();
        // A broken consumer pipe marks the writer closed from run(). The
        // worker is still joinable after it exits, so close() must always reap
        // it instead of returning early and letting std::thread terminate the
        // capture process during destruction.
        if(thread_.joinable()) thread_.join();
    }

    std::uint64_t dropped() const { return dropped_.load(); }
    bool failed() const { return failed_.load(); }

private:
    void run() {
        for(;;) {
            RgbdStreamFrame frame;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [this] { return closed_ || !queue_.empty(); });
                if(queue_.empty()) break;
                frame = std::move(queue_.front());
                queue_.pop_front();
            }
            output_.write(reinterpret_cast<const char *>(&frame.header), sizeof(frame.header));
            output_.write(
                reinterpret_cast<const char *>(frame.depth.data()),
                static_cast<std::streamsize>(frame.header.depth_bytes));
            if(!frame.color.empty()) {
                output_.write(
                    reinterpret_cast<const char *>(frame.color.data()),
                    static_cast<std::streamsize>(frame.header.color_bytes));
            }
            output_.flush();
            if(!output_) {
                failed_ = true;
                std::lock_guard lock(mutex_);
                queue_.clear();
                closed_ = true;
                break;
            }
        }
    }

    std::ostream &output_;
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<RgbdStreamFrame> queue_;
    std::thread thread_;
    std::atomic<std::uint64_t> dropped_{0};
    std::atomic<bool> failed_{false};
    bool closed_ = false;
};

} // namespace scanlan
