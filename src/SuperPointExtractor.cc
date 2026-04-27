/**
 * This file is part of ORB-SLAM3 SuperPoint integration.
 *
 * SuperPointExtractor implementation using ONNX Runtime.
 */

#include "SuperPointExtractor.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;

namespace ORB_SLAM3
{

// Default ONNX model path
static const std::string DEFAULT_MODEL_PATH = "./Models/superpoint_v1.onnx";

SuperPointExtractor::SuperPointExtractor(int nfeatures, float scaleFactor,
    int nlevels, float iniThFAST, float minThFAST,
    const std::string &modelPath)
  : nfeatures(nfeatures), scaleFactor(scaleFactor), nlevels(nlevels),
    mfConfThreshold(iniThFAST > 0 ? iniThFAST : 0.015f),
    mfNmsRadius(minThFAST > 0 ? minThFAST : 4.0f),
    mnDescriptorDim(256)
{
    // Initialize scale pyramid (for compatibility with Frame)
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i = 0; i < nlevels; i++)
    {
        mvScaleFactor[i] = pow(scaleFactor, i);
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
        mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
    }

    // Initialize ONNX Runtime
    mpEnv = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");

    mSessionOptions.SetIntraOpNumThreads(4);
    mSessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Load model
    const std::string &modelFile = modelPath.empty() ? DEFAULT_MODEL_PATH : modelPath;
    mpSession = new Ort::Session(*mpEnv, modelFile.c_str(), mSessionOptions);
    cerr << "[SuperPoint] Model loaded: " << modelFile << endl;

    // Get input name
    mInputName = mpSession->GetInputNameAllocated(0, mAllocator).get();

    // Get output names
    size_t numOutputs = mpSession->GetOutputCount();
    mOutputNameStrings.resize(numOutputs);
    mOutputNames.resize(numOutputs);
    for(size_t i = 0; i < numOutputs; i++)
    {
        auto nameAllocated = mpSession->GetOutputNameAllocated(i, mAllocator);
        mOutputNameStrings[i] = nameAllocated.get();
        mOutputNames[i] = mOutputNameStrings[i].c_str();
    }
}

SuperPointExtractor::~SuperPointExtractor()
{
    if(mpSession) delete mpSession;
    if(mpEnv) delete mpEnv;
}

void SuperPointExtractor::ComputePyramid(cv::Mat image)
{
    mvImagePyramid.resize(nlevels);
    mvImagePyramid[0] = image.clone();
    for(int level = 1; level < nlevels; level++)
    {
        cv::Size sz(image.cols / mvScaleFactor[level],
                     image.rows / mvScaleFactor[level]);
        cv::resize(mvImagePyramid[level-1], mvImagePyramid[level], sz);
    }
}

int SuperPointExtractor::operator()(cv::InputArray _image, cv::InputArray _mask,
    std::vector<cv::KeyPoint> &_keypoints, cv::OutputArray _descriptors,
    std::vector<int> &vLappingArea)
{
    cv::Mat image = _image.getMat();
    if(image.empty())
    {
        _keypoints.clear();
        _descriptors.create(0, mnDescriptorDim, CV_32F);
        return 0;
    }

    // Convert to grayscale float32 [0,1]
    cv::Mat processed;
    if(image.channels() == 3)
        cv::cvtColor(image, processed, cv::COLOR_BGR2GRAY);
    else
        processed = image.clone();
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);

    int H = processed.rows;
    int W = processed.cols;

    // Prepare input tensor (1, 1, H, W)
    vector<int64_t> inputShape = {1, 1, H, W};
    size_t inputSize = 1 * 1 * H * W * sizeof(float);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, processed.ptr<float>(), inputSize,
        inputShape.data(), inputShape.size());

    // Run inference
    const char* inputNames[] = { mInputName.c_str() };
    auto outputTensors = mpSession->Run(
        Ort::RunOptions{nullptr},
        inputNames, &inputTensor, 1,
        mOutputNames.data(), mOutputNames.size());

    // Get outputs: heatmap (1, H, W), desc (1, 256, H/8, W/8)
    float* heatmap_data = outputTensors[0].GetTensorMutableData<float>();
    float* desc = outputTensors[1].GetTensorMutableData<float>();

    auto heatmapInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    auto descInfo = outputTensors[1].GetTensorTypeAndShapeInfo();
    auto heatmapShape = heatmapInfo.GetShape();
    auto descShape = descInfo.GetShape();

    int H8 = static_cast<int>(heatmapShape[1]); // H (full resolution)
    int W8 = static_cast<int>(heatmapShape[2]); // W (full resolution)
    int featH = static_cast<int>(descShape[2]); // H/8
    int featW = static_cast<int>(descShape[3]); // W/8

    // --- cv::dilate NMS (replaces O(N^2) brute-force) ---
    // Wrap ONNX output tensor as cv::Mat (zero-copy, safe: inference is done)
    cv::Mat heatmap(H8, W8, CV_32F, heatmap_data);

    int nmsRadius = static_cast<int>(mfNmsRadius);
    int ksize = nmsRadius * 2 + 1;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize, ksize));
    cv::Mat local_max;
    cv::dilate(heatmap, local_max, kernel);
    cv::Mat nms_mask = (heatmap >= local_max);
    heatmap.setTo(0, ~nms_mask);

    // Border removal + threshold filtering
    int border = 4; // matches remove_borders=4 in superpoint.py
    int x0 = (vLappingArea.size() >= 2) ? vLappingArea[0] : 0;
    int x1 = (vLappingArea.size() >= 2) ? vLappingArea[1] : W8;

    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(nfeatures);
    for(int y = border; y < H8 - border; y++)
    {
        const float* row = heatmap.ptr<float>(y);
        for(int x = std::max(x0, border); x < std::min(x1, W8 - border); x++)
        {
            if(row[x] > mfConfThreshold)
                keypoints.push_back(cv::KeyPoint((float)x, (float)y, 31.0f, -1.0f, row[x], 0, -1));
        }
    }

    // Sort by score descending, take top nfeatures
    std::sort(keypoints.begin(), keypoints.end(),
        [](const cv::KeyPoint& a, const cv::KeyPoint& b) { return a.response > b.response; });
    if((int)keypoints.size() > nfeatures)
        keypoints.resize(nfeatures);

    int N = keypoints.size();
    _keypoints = keypoints;

    // --- Descriptor extraction (bilinear interpolation) ---
    cv::Mat descriptors(N, mnDescriptorDim, CV_32F);
    int planeSize = featH * featW;

    for(int i = 0; i < N; i++)
    {
        float fx = keypoints[i].pt.x - 4.0f + 0.5f;
        float fy = keypoints[i].pt.y - 4.0f + 0.5f;
        fx = fx / (featW * 8.0f - 4.0f - 0.5f) * 2.0f - 1.0f;
        fy = fy / (featH * 8.0f - 4.0f - 0.5f) * 2.0f - 1.0f;
        fx = (fx + 1.0f) * (featW - 1.0f) / 2.0f;
        fy = (fy + 1.0f) * (featH - 1.0f) / 2.0f;

        int bx0 = (int)floor(fx);
        int by0 = (int)floor(fy);
        int bx1 = bx0 + 1;
        int by1 = by0 + 1;
        float wa = (fx - bx0);
        float ha = (fy - by0);
        bx0 = std::max(0, std::min(bx0, featW - 1));
        bx1 = std::max(0, std::min(bx1, featW - 1));
        by0 = std::max(0, std::min(by0, featH - 1));
        by1 = std::max(0, std::min(by1, featH - 1));

        float* dPtr = descriptors.ptr<float>(i);
        for(int c = 0; c < mnDescriptorDim; c++)
        {
            float v00 = desc[c * planeSize + by0 * featW + bx0];
            float v01 = desc[c * planeSize + by0 * featW + bx1];
            float v10 = desc[c * planeSize + by1 * featW + bx0];
            float v11 = desc[c * planeSize + by1 * featW + bx1];
            dPtr[c] = (1-ha)*(1-wa)*v00 + (1-ha)*wa*v01 + ha*(1-wa)*v10 + ha*wa*v11;
        }
        // Re-normalize after bilinear interpolation (interpolation breaks unit norm)
        float norm = 0;
        for(int c = 0; c < mnDescriptorDim; c++)
            norm += dPtr[c] * dPtr[c];
        norm = sqrt(norm);
        if(norm > 1e-6)
            for(int c = 0; c < mnDescriptorDim; c++)
                dPtr[c] /= norm;
    }

    _descriptors.create(N, mnDescriptorDim, CV_32F);
    descriptors.copyTo(_descriptors.getMat());

    // Compute image pyramid for stereo matching compatibility
    cv::Mat grayImage;
    if(image.channels() == 3)
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    else
        grayImage = image.clone();
    ComputePyramid(grayImage);

    return N;
}

} // namespace ORB_SLAM3
