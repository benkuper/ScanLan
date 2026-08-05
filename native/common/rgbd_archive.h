#pragma once

#include "jpeg_writer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace scanlan {

inline bool archive_frame_due(
    std::uint64_t source_frame,
    int native_fps,
    int requested_fps) {
    if(source_frame <= 1) return true;
    const auto source_rate = std::max(1, native_fps);
    const auto target_rate = std::clamp(requested_fps, 1, source_rate);
    const auto current_bucket = (source_frame - 1) * static_cast<std::uint64_t>(target_rate)
        / static_cast<std::uint64_t>(source_rate);
    const auto previous_bucket = (source_frame - 2) * static_cast<std::uint64_t>(target_rate)
        / static_cast<std::uint64_t>(source_rate);
    return current_bucket != previous_bucket;
}

struct RgbdArchiveFrame {
    std::uint64_t source_sequence = 0;
    std::uint64_t depth_timestamp_us = 0;
    std::uint64_t color_timestamp_us = 0;
    std::vector<std::uint16_t> depth;
    std::vector<std::uint8_t> aligned_color;
    std::uint32_t native_color_width = 0;
    std::uint32_t native_color_height = 0;
    std::vector<std::uint8_t> native_color;
    std::optional<std::array<float, 16>> camera_to_world;
};

// Bounded, latest-wins RGB-D recording. The camera thread only moves buffers
// into this queue; depth/color files, JPEG compression, and CSV publication are
// handled in the background. An overloaded disk therefore produces observable
// source gaps instead of delayed tracking input.
class RgbdArchiveWriter {
public:
    RgbdArchiveWriter(
        fs::path root,
        int jpeg_quality,
        std::uint32_t max_rgb_dimension,
        std::size_t capacity = 8)
        : root_(std::move(root)),
          capacity_(std::max<std::size_t>(1, capacity)),
          jpeg_(jpeg_quality, max_rgb_dimension, capacity_),
          frames_(root_ / "frames.csv", std::ios::trunc) {
        fs::create_directories(root_ / "depth");
        fs::create_directories(root_ / "color");
        fs::create_directories(root_ / "rgb");
        if(!frames_) throw std::runtime_error("Could not create the RGB-D frame index");
        frames_ << "index,source_sequence,timestamp_us,depth_path,color_path,rgb_path,rgb_timestamp_us";
        for(int row = 0; row < 4; ++row) {
            for(int column = 0; column < 4; ++column) frames_ << ",m" << row << column;
        }
        frames_ << '\n';
        worker_ = std::thread([this] { run(); });
    }

    RgbdArchiveWriter(const RgbdArchiveWriter&) = delete;
    RgbdArchiveWriter& operator=(const RgbdArchiveWriter&) = delete;

    ~RgbdArchiveWriter() { close(); }

    bool submit(RgbdArchiveFrame frame) {
        {
            std::lock_guard lock(mutex_);
            if(closing_) return false;
            if(queue_.size() == capacity_) {
                queue_.pop_front();
                ++dropped_;
            }
            queue_.push_back(std::move(frame));
        }
        available_.notify_one();
        return true;
    }

    void close() {
        {
            std::lock_guard lock(mutex_);
            if(closing_) return;
            closing_ = true;
        }
        available_.notify_all();
        if(worker_.joinable()) worker_.join();
        jpeg_.close();
        frames_.flush();
    }

    [[nodiscard]] std::uint32_t saved() const { return saved_.load(); }
    [[nodiscard]] std::uint32_t dropped() const {
        return dropped_.load() + jpeg_.dropped();
    }
    [[nodiscard]] std::uint32_t failed() const {
        return failed_.load() + jpeg_.failed();
    }

private:
    template <typename T>
    static void write_binary(const fs::path& path, const std::vector<T>& values) {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if(!output) throw std::runtime_error("Could not create " + path.string());
        output.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
        if(!output) throw std::runtime_error("Could not write " + path.string());
    }

    void write(RgbdArchiveFrame frame) {
        const std::uint32_t index = saved_.load();
        std::ostringstream stem;
        stem << std::setw(6) << std::setfill('0') << index;
        const std::string depth_name = stem.str() + ".u16";
        const std::string color_name = stem.str() + ".rgb";
        const std::string rgb_name = stem.str() + ".jpg";
        write_binary(root_ / "depth" / depth_name, frame.depth);
        write_binary(root_ / "color" / color_name, frame.aligned_color);
        const bool rgb_queued = !frame.native_color.empty()
            && jpeg_.enqueue(
                root_ / "rgb" / rgb_name,
                frame.native_color_width,
                frame.native_color_height,
                std::move(frame.native_color));

        frames_ << index << ',' << frame.source_sequence << ',' << frame.depth_timestamp_us
                << ",depth/" << depth_name << ",color/" << color_name << ',';
        if(rgb_queued) frames_ << "rgb/" << rgb_name;
        frames_ << ',';
        if(rgb_queued) frames_ << frame.color_timestamp_us;
        if(frame.camera_to_world) {
            for(float value : *frame.camera_to_world) frames_ << ',' << value;
        } else {
            for(int value = 0; value < 16; ++value) frames_ << ',';
        }
        frames_ << '\n';
        frames_.flush();
        ++saved_;
    }

    void run() {
        for(;;) {
            RgbdArchiveFrame frame;
            {
                std::unique_lock lock(mutex_);
                available_.wait(lock, [this] { return closing_ || !queue_.empty(); });
                if(queue_.empty()) break;
                frame = std::move(queue_.front());
                queue_.pop_front();
            }
            try {
                write(std::move(frame));
            } catch(...) {
                ++failed_;
            }
        }
    }

    fs::path root_;
    std::size_t capacity_;
    JpegWriterQueue jpeg_;
    std::ofstream frames_;
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::deque<RgbdArchiveFrame> queue_;
    std::thread worker_;
    std::atomic<std::uint32_t> saved_{0};
    std::atomic<std::uint32_t> dropped_{0};
    std::atomic<std::uint32_t> failed_{0};
    bool closing_ = false;
};

} // namespace scanlan
