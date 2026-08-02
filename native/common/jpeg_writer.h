#pragma once

#include <Windows.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace scanlan {

namespace fs = std::filesystem;
using Microsoft::WRL::ComPtr;

class JpegWriterQueue {
public:
    explicit JpegWriterQueue(int quality = 92, std::uint32_t max_dimension = 0, std::size_t capacity = 6)
        : quality_(std::clamp(quality, 60, 100)),
          max_dimension_(max_dimension),
          capacity_(std::max<std::size_t>(1, capacity)),
          worker_([this] { run(); }) {}

    JpegWriterQueue(const JpegWriterQueue&) = delete;
    JpegWriterQueue& operator=(const JpegWriterQueue&) = delete;

    ~JpegWriterQueue() { close(); }

    bool enqueue(fs::path path, std::uint32_t width, std::uint32_t height,
                 std::vector<std::uint8_t> rgb) {
        std::lock_guard lock(mutex_);
        if (closing_ || queue_.size() >= capacity_) {
            ++dropped_;
            return false;
        }
        queue_.push_back(Item{std::move(path), width, height, std::move(rgb)});
        available_.notify_one();
        return true;
    }

    void close() {
        {
            std::lock_guard lock(mutex_);
            if (closing_) return;
            closing_ = true;
        }
        available_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    [[nodiscard]] std::uint32_t dropped() const { return dropped_.load(); }
    [[nodiscard]] std::uint32_t failed() const { return failed_.load(); }

private:
    struct Item {
        fs::path path;
        std::uint32_t width;
        std::uint32_t height;
        std::vector<std::uint8_t> rgb;
    };

    static void check(HRESULT result, const char* detail) {
        if (FAILED(result)) throw std::runtime_error(detail);
    }

    void encode(const Item& item, IWICImagingFactory* factory) const {
        fs::create_directories(item.path.parent_path());
        ComPtr<IWICStream> stream;
        check(factory->CreateStream(&stream), "WIC could not create a JPEG stream");
        check(stream->InitializeFromFilename(item.path.c_str(), GENERIC_WRITE),
              "WIC could not open the JPEG destination");

        ComPtr<IWICBitmapEncoder> encoder;
        check(factory->CreateEncoder(GUID_ContainerFormatJpeg, nullptr, &encoder),
              "WIC JPEG encoder is unavailable");
        check(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache),
              "WIC could not initialize the JPEG encoder");
        ComPtr<IWICBitmapFrameEncode> frame;
        ComPtr<IPropertyBag2> properties;
        check(encoder->CreateNewFrame(&frame, &properties), "WIC could not create a JPEG frame");
        PROPBAG2 option{};
        option.pstrName = const_cast<LPOLESTR>(L"ImageQuality");
        VARIANT value;
        VariantInit(&value);
        value.vt = VT_R4;
        value.fltVal = static_cast<float>(quality_) / 100.0F;
        properties->Write(1, &option, &value);
        VariantClear(&value);
        check(frame->Initialize(properties.Get()), "WIC could not initialize a JPEG frame");

        ComPtr<IWICBitmap> bitmap;
        check(factory->CreateBitmapFromMemory(
                  item.width, item.height, GUID_WICPixelFormat24bppRGB, item.width * 3,
                  static_cast<UINT>(item.rgb.size()),
                  const_cast<BYTE*>(reinterpret_cast<const BYTE*>(item.rgb.data())), &bitmap),
              "WIC could not wrap the RGB image");
        std::uint32_t output_width = item.width;
        std::uint32_t output_height = item.height;
        if (max_dimension_ > 0 && std::max(item.width, item.height) > max_dimension_) {
            const double scale = static_cast<double>(max_dimension_) / std::max(item.width, item.height);
            output_width = std::max(1U, static_cast<std::uint32_t>(item.width * scale + 0.5));
            output_height = std::max(1U, static_cast<std::uint32_t>(item.height * scale + 0.5));
        }
        check(frame->SetSize(output_width, output_height), "WIC could not set JPEG dimensions");
        WICPixelFormatGUID format = GUID_WICPixelFormat24bppRGB;
        check(frame->SetPixelFormat(&format), "WIC could not set JPEG RGB format");
        if (output_width == item.width && output_height == item.height) {
            check(frame->WriteSource(bitmap.Get(), nullptr), "WIC could not write JPEG pixels");
        } else {
            ComPtr<IWICBitmapScaler> scaler;
            check(factory->CreateBitmapScaler(&scaler), "WIC could not create an RGB scaler");
            check(scaler->Initialize(bitmap.Get(), output_width, output_height,
                                     WICBitmapInterpolationModeFant),
                  "WIC could not scale native RGB");
            check(frame->WriteSource(scaler.Get(), nullptr), "WIC could not write scaled JPEG pixels");
        }
        check(frame->Commit(), "WIC could not finalize the JPEG frame");
        check(encoder->Commit(), "WIC could not finalize the JPEG file");
    }

    void run() {
        const HRESULT initialized = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        ComPtr<IWICImagingFactory> factory;
        if (SUCCEEDED(initialized)) {
            CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                             IID_PPV_ARGS(&factory));
        }
        while (true) {
            Item item;
            {
                std::unique_lock lock(mutex_);
                available_.wait(lock, [this] { return closing_ || !queue_.empty(); });
                if (queue_.empty()) {
                    if (closing_) break;
                    continue;
                }
                item = std::move(queue_.front());
                queue_.pop_front();
            }
            try {
                if (!factory) throw std::runtime_error("WIC imaging factory is unavailable");
                encode(item, factory.Get());
            } catch (...) {
                ++failed_;
                std::error_code ignored;
                fs::remove(item.path, ignored);
            }
        }
        if (SUCCEEDED(initialized)) CoUninitialize();
    }

    int quality_;
    std::uint32_t max_dimension_;
    std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::deque<Item> queue_;
    bool closing_ = false;
    std::atomic<std::uint32_t> dropped_{0};
    std::atomic<std::uint32_t> failed_{0};
    std::thread worker_;
};

}  // namespace scanlan
