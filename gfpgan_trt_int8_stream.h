#pragma once

#include "NvInfer.h"
#include "run_constant.h"
#include "utils.h"

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

#define MODEL_IN_DATA_TYPE cv::float16_t

class IBatchStream
{
public:
    virtual void reset(int firstBatch) = 0;
    virtual bool next() = 0;
    virtual void skip(int skipCount) = 0;
    virtual MODEL_IN_DATA_TYPE* getBatch() = 0;
    virtual float* getLabels() = 0;
    virtual int getBatchesRead() const = 0;
    virtual int getBatchSize() const = 0;
    virtual nvinfer1::Dims getDims() const = 0;
};

class GfpGanInt8Stream : public IBatchStream
{
public:
    GfpGanInt8Stream()
    {
        p_img_read_step = 25;
        mBatchCount = 0;

        if (!pPrepareImagesPath())
        {
            std::cerr << "[GfpGanInt8Stream] Construct stop! pPrepareImagesPath() Failed!!!" << std::endl;
            return;
        }

        p_batch_data = new cv::float16_t[25 * 3 * 512 * 512];

    }

    bool next() override
    {
        if (mBatchCount + p_img_read_step > p_img_count)
        {
            return false;
        }
        std::cout << "[GfpGanInt8Stream] Current: " << mBatchCount << " batch data readed!" << std::endl;
        this->mBatch.clear();
        this->mBatch.resize(3 * 512 * 512 * p_img_read_step);
        size_t index(0);
        cv::Mat temp(512, 512, CV_16FC3);
        for (; index < p_img_read_step; ++index)
        {
            temp = cv::imread(p_vec_imgs_path[index + mBatchCount], cv::IMREAD_COLOR);
            temp.convertTo(temp, CV_16FC3, 1.0 / 255.0, 0);
            cv::Mat input_data = (temp - 0.5) / 0.5;
            cv::split(input_data, p_vec_split_res);

            std::memcpy(p_batch_data + index * 3 * 512 * 512 + 0 * 512 * 512, p_vec_split_res[2].data, 512 * 512);
            std::memcpy(p_batch_data + index * 3 * 512 * 512 + 0 * 512 * 512, p_vec_split_res[1].data, 512 * 512);
            std::memcpy(p_batch_data + index * 3 * 512 * 512 + 0 * 512 * 512, p_vec_split_res[0].data, 512 * 512);
        }
        std::cout << "[GfpGanInt8Stream] Current batch data size: " << 25 * 3 * 512 * 512 << " loaded!" << std::endl;
        mBatchCount = mBatchCount + p_img_read_step;
        return true;
    }

    MODEL_IN_DATA_TYPE* getBatch() override
    {
        return p_batch_data;
        //return mBatch.data();
    }

    int getBatchSize() const override
    {
        return p_img_count / p_img_read_step;
    }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    void skip(int skipCount) override
    {
    }

    float* getLabels() override
    {
        return nullptr;
    }

    void reset(int firstBatch) override
    {
        mBatchCount = 0;
    }

    nvinfer1::Dims getDims() const override
    {
        nvinfer1::Dims ret;
        ret.nbDims = 4;
        ret.d[0] = 25;
        ret.d[1] = 3;
        ret.d[2] = 512;
        ret.d[3] = 512;
        return ret;
    }
private:
    bool pPrepareImagesPath()
    {
        p_vec_imgs_path = LqhUtil::GetFilesPath("D:\\data\\SF_dataset\\gt", ".png");
        if (p_vec_imgs_path.size() == 0)
        {
            std::cerr << "[GfpGanInt8Stream] pPrepareImagesPath() failed! p_imgs_path.size() == 0 !" << std::endl;
            return false;
        }

        if (p_vec_imgs_path.size() < p_img_read_step)
        {
            std::cerr << "[GfpGanInt8Stream] pPrepareImagesPath() failed! p_imgs_path.size() < p_img_read_step: "
                << p_vec_imgs_path.size() << " < " << p_img_read_step << std::endl;
            return false;
        }

        p_img_count = p_vec_imgs_path.size();
        return true;
    }

    std::vector<std::string>        p_vec_imgs_path;    // img names in img_path
    std::vector<cv::Mat>            p_vec_split_res;    // [B, G, R]
    std::vector<MODEL_IN_DATA_TYPE> mBatch;         //!< Data for the batch
    MODEL_IN_DATA_TYPE* p_batch_data;
    size_t mBatchCount;                             // current index
    size_t p_img_read_step;                         // img read step 
    size_t p_img_count;                             // img count

};