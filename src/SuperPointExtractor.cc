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

    mSessionOptions.SetIntraOpNumThreads(1);
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

    cerr << "[SuperPoint] Model loaded: " << DEFAULT_MODEL_PATH
         << " | Features: " << nfeatures
         << " | ConfThreshold: " << mfConfThreshold
         << " | NMSRadius: " << mfNmsRadius << endl;
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

    cerr << "[SP] Image: " << image.cols << "x" << image.rows << endl;

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

    // Get outputs: semi (1, 65, H/8, W/8), desc (1, 256, H/8, W/8)
    float* semi = outputTensors[0].GetTensorMutableData<float>();
    float* desc = outputTensors[1].GetTensorMutableData<float>();

    auto semiInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    auto descInfo = outputTensors[1].GetTensorTypeAndShapeInfo();
    auto semiShape = semiInfo.GetShape();
    auto descShape = descInfo.GetShape();

    int semiChannels = static_cast<int>(semiShape[1]); // 65
    int featH = static_cast<int>(semiShape[2]); // H/8
    int featW = static_cast<int>(semiShape[3]); // W/8

    // Apply dustbin-free softmax to semi along channel dimension, then reshape
    // SuperPoint reference (superpoint.py L163-166):
    //   scores = softmax(scores, 1)[:, :-1]        # (B, 64, H/8, W/8)
    //   scores = scores.permute(0,2,3,1).reshape(B, H/8, W/8, 8, 8)
    //   scores = scores.permute(0,1,3,2,4).reshape(B, H, W)
    //
    // This expands each cell's 64-channel probability into an 8x8 sub-pixel grid,
    // producing a full-resolution (H x W) heatmap where each pixel has a probability
    // of being a keypoint.

    int H8 = featH * 8; // = H (original image height)
    int W8 = featW * 8; // = W (original image width)
    cv::Mat heatmap(H8, W8, CV_32F, 0.0f);

    for(int cy = 0; cy < featH; cy++)
    {
        for(int cx = 0; cx < featW; cx++)
        {
            // Softmax over 64 channels (exclude dustbin ch 64) for numerical stability
            float maxVal = -FLT_MAX;
            for(int c = 0; c < 64; c++)
            {
                float v = semi[c * featH * featW + cy * featW + cx];
                if(v > maxVal) maxVal = v;
            }
            float sumExp = 0;
            for(int c = 0; c < 64; c++)
            {
                float v = semi[c * featH * featW + cy * featW + cx];
                sumExp += exp(v - maxVal);
            }

            // Reshape: 64 channels -> 8x8 grid, mapped to sub-pixel positions
            // Channel c = dy*8 + dx corresponds to sub-pixel offset (dx, dy) within cell
            // Output heatmap pixel at (cy*8 + dy, cx*8 + dx)
            for(int dy = 0; dy < 8; dy++)
            {
                for(int dx = 0; dx < 8; dx++)
                {
                    int c = dy * 8 + dx;
                    float v = semi[c * featH * featW + cy * featW + cx];
                    float prob = exp(v - maxVal) / sumExp;
                    int hy = cy * 8 + dy;
                    int hx = cx * 8 + dx;
                    if(hy < H8 && hx < W8)
                        heatmap.at<float>(hy, hx) = prob;
                }
            }
        }
    }

    // NMS (simple_nms from superpoint.py: iterative max-pool suppression)
    // Using a single-pass approach with configurable radius (in original image pixels)
    int nmsRadius = static_cast<int>(mfNmsRadius);
    int x0 = (vLappingArea.size() >= 2) ? vLappingArea[0] : 0;
    int x1 = (vLappingArea.size() >= 2) ? vLappingArea[1] : W8;

    // Collect candidate keypoints above threshold (in full-resolution heatmap)
    struct Candidate {
        float score;
        int x, y;
    };
    vector<Candidate> candidates;
    candidates.reserve(H8 * W8 / 10); // estimate

    for(int y = 4; y < H8 - 4; y++) // remove_borders=4
    {
        const float* row = heatmap.ptr<float>(y);
        for(int x = max(x0, 4); x < min(x1, W8 - 4); x++)
        {
            if(row[x] > mfConfThreshold)
            {
                candidates.push_back({row[x], x, y});
            }
        }
    }

    // Sort by score descending
    sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    // NMS: suppress candidates within nmsRadius pixels of a higher-scoring candidate
    vector<bool> suppressed(candidates.size(), false);
    vector<cv::KeyPoint> selectedKps;
    vector<int> selectedIndices;

    int maxKP = nfeatures;
    int nmsR2 = nmsRadius * nmsRadius;
    for(size_t i = 0; i < candidates.size() && (int)selectedKps.size() < maxKP; i++)
    {
        if(suppressed[i]) continue;
        selectedKps.push_back(cv::KeyPoint(
            (float)candidates[i].x, (float)candidates[i].y,
            31.0f, -1.0f, candidates[i].score, 0, -1));
        selectedIndices.push_back(i);

        // Suppress nearby (in full-resolution coordinates)
        for(size_t j = i + 1; j < candidates.size(); j++)
        {
            if(suppressed[j]) continue;
            int ddx = candidates[i].x - candidates[j].x;
            int ddy = candidates[i].y - candidates[j].y;
            if(ddx*ddx + ddy*ddy <= nmsR2)
            {
                suppressed[j] = true;
            }
        }
    }

    int N = selectedKps.size();
    _keypoints = selectedKps;

    // Extract descriptors from the dense descriptor map using bilinear interpolation
    // Following superpoint.py sample_descriptors: grid_sample with bilinear mode
    cv::Mat descriptors(N, mnDescriptorDim, CV_32F);
    int planeSize = featH * featW;

    for(int i = 0; i < N; i++)
    {
        // Map keypoint from image coords to feature-map coords
        // superpoint.py: keypoints = keypoints - s/2 + 0.5; then normalize to [-1,1]
        float fx = selectedKps[i].pt.x - 4.0f + 0.5f; // s=8, s/2=4
        float fy = selectedKps[i].pt.y - 4.0f + 0.5f;
        // Normalize to feature map range
        fx = fx / (featW * 8.0f - 4.0f - 0.5f) * 2.0f - 1.0f;
        fy = fy / (featH * 8.0f - 4.0f - 0.5f) * 2.0f - 1.0f;
        // Convert from [-1,1] to feature-map pixel coords
        fx = (fx + 1.0f) * (featW - 1.0f) / 2.0f;
        fy = (fy + 1.0f) * (featH - 1.0f) / 2.0f;

        // Bilinear interpolation in the feature map
        int x0 = (int)floor(fx);
        int y0 = (int)floor(fy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float wa = (fx - x0);
        float ha = (fy - y0);
        x0 = max(0, min(x0, featW - 1));
        x1 = max(0, min(x1, featW - 1));
        y0 = max(0, min(y0, featH - 1));
        y1 = max(0, min(y1, featH - 1));

        float* dPtr = descriptors.ptr<float>(i);
        for(int c = 0; c < mnDescriptorDim; c++)
        {
            float v00 = desc[c * planeSize + y0 * featW + x0];
            float v01 = desc[c * planeSize + y0 * featW + x1];
            float v10 = desc[c * planeSize + y1 * featW + x0];
            float v11 = desc[c * planeSize + y1 * featW + x1];
            dPtr[c] = (1-ha)*(1-wa)*v00 + (1-ha)*wa*v01 + ha*(1-wa)*v10 + ha*wa*v11;
        }

        // L2 normalize
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

    cv::Mat descMat = _descriptors.getMat();
    cerr << "[SP] Extracted " << N << " keypoints, descriptors: " << descMat.rows << "x" << descMat.cols << " type=" << descMat.type() << endl;

    return N;
}

} // namespace ORB_SLAM3
