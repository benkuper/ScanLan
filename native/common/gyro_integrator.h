#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <optional>

namespace scanlan {

// Integrates calibrated camera-frame angular velocity into the point transform
// from the previous depth frame to the current one. take() is called at each
// depth frame boundary; malformed timing windows are discarded instead of
// poisoning visual odometry.
class GyroDeltaIntegrator {
public:
    void add(std::uint64_t timestamp_us, float x, float y, float z) {
        std::lock_guard lock(mutex_);
        if(last_timestamp_us_ == 0) {
            last_timestamp_us_ = timestamp_us;
            previous_rate_ = {x, y, z};
            return;
        }
        if(timestamp_us <= last_timestamp_us_) {
            reset_locked(timestamp_us, x, y, z);
            return;
        }
        const double dt = static_cast<double>(timestamp_us - last_timestamp_us_) / 1'000'000.0;
        if(dt > 0.08) {
            reset_locked(timestamp_us, x, y, z);
            return;
        }
        const std::array<double, 3> mean{
            0.5 * (previous_rate_[0] + x),
            0.5 * (previous_rate_[1] + y),
            0.5 * (previous_rate_[2] + z),
        };
        const double magnitude = std::sqrt(
            mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]);
        const double angle = magnitude * dt;
        if(std::isfinite(angle) && angle > 1e-9 && angle <= 0.7) {
            const double scale = std::sin(angle * 0.5) / magnitude;
            // Gyroscope integration gives the camera's body rotation. RGB-D
            // odometry needs the inverse point transform (previous -> current).
            const std::array<double, 4> delta{
                -mean[0] * scale,
                -mean[1] * scale,
                -mean[2] * scale,
                std::cos(angle * 0.5),
            };
            quaternion_ = multiply(delta, quaternion_);
            has_delta_ = true;
        }
        last_timestamp_us_ = timestamp_us;
        previous_rate_ = {x, y, z};
    }

    std::optional<std::array<float, 4>> take() {
        std::lock_guard lock(mutex_);
        if(!has_delta_) return std::nullopt;
        const double norm = std::sqrt(
            quaternion_[0] * quaternion_[0] + quaternion_[1] * quaternion_[1]
            + quaternion_[2] * quaternion_[2] + quaternion_[3] * quaternion_[3]);
        std::optional<std::array<float, 4>> result;
        if(std::isfinite(norm) && norm > 1e-9) {
            result = std::array<float, 4>{
                static_cast<float>(quaternion_[0] / norm),
                static_cast<float>(quaternion_[1] / norm),
                static_cast<float>(quaternion_[2] / norm),
                static_cast<float>(quaternion_[3] / norm),
            };
        }
        quaternion_ = {0, 0, 0, 1};
        has_delta_ = false;
        return result;
    }

private:
    static std::array<double, 4> multiply(
        const std::array<double, 4>& left,
        const std::array<double, 4>& right) {
        const auto [lx, ly, lz, lw] = left;
        const auto [rx, ry, rz, rw] = right;
        return {
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        };
    }

    void reset_locked(std::uint64_t timestamp_us, float x, float y, float z) {
        last_timestamp_us_ = timestamp_us;
        previous_rate_ = {x, y, z};
        quaternion_ = {0, 0, 0, 1};
        has_delta_ = false;
    }

    std::mutex mutex_;
    std::uint64_t last_timestamp_us_ = 0;
    std::array<double, 3> previous_rate_{0, 0, 0};
    std::array<double, 4> quaternion_{0, 0, 0, 1};
    bool has_delta_ = false;
};

} // namespace scanlan
