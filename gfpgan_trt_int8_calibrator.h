#pragma once

#include "NvInfer.h"
#include "gfpgan_trt_int8_stream.h"

#include <string>
#include <vector>

class EntropyCalibratorImpl
{
public:

    EntropyCalibratorImpl(std::string const& networkName,
        const char* inputBlobName, bool readCache = true)
        : mCalibrationTableName("CalibrationTable" + networkName)
        , mInputBlobName(inputBlobName)
        , mReadCache(readCache)
    {
        mInputCount = CALIB_STEP * 3 * 512 * 512;
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(INPUT_DATA_TYPE)));
    }

    virtual ~EntropyCalibratorImpl()
    {
        //CHECK(nvinfer1::cudaFree(mDeviceInput));
    }

    int getBatchSize() const noexcept
    {
        return mStream.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
    {
        if (!mStream.next())
        {
            return false;
        }
        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(MODEL_IN_DATA_TYPE), cudaMemcpyHostToDevice));
        ASSERT(!strcmp(names[0], mInputBlobName));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length) noexcept
    {
        mCalibrationCache.clear();
        std::ifstream input(mCalibrationTableName, std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept
    {
        std::ofstream output(mCalibrationTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    GfpGanInt8Stream mStream;
    size_t mInputCount;
    std::string mCalibrationTableName;
    const char* mInputBlobName;
    bool mReadCache{ true };
    void* mDeviceInput{ nullptr };
    std::vector<char> mCalibrationCache;
};

class GfpGanInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    GfpGanInt8Calibrator() : p_entropy_calibrator_impl("GFPGAN", "input", true)
    {

    };

    int getBatchSize() const noexcept override
    {
        return p_entropy_calibrator_impl.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
    {
        return p_entropy_calibrator_impl.getBatch(bindings, names, nbBindings);
    }

    const void* readCalibrationCache(size_t& length) noexcept override
    {
        return p_entropy_calibrator_impl.readCalibrationCache(length);
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override
    {
        p_entropy_calibrator_impl.writeCalibrationCache(cache, length);
    }

private:
    EntropyCalibratorImpl p_entropy_calibrator_impl;
};