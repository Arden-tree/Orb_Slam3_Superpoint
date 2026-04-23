/**
 * This file is part of ORB-SLAM3 SuperPoint integration.
 *
 * SuperPointExtractor: Replaces ORBextractor with ONNX Runtime inference.
 * Outputs 256-dim L2-normalized float descriptors (CV_32F).
 */

#ifndef SUPERPOINTEXTRACTOR_H
#define SUPERPOINTEXTRACTOR_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace ORB_SLAM3
{

class SuperPointExtractor
{
public:

    SuperPointExtractor(int nfeatures, float scaleFactor, int nlevels,
                        float iniThFAST, float minThFAST,
                        const std::string &modelPath = "./Models/superpoint_v1.onnx");

    ~SuperPointExtractor();

    // Drop-in replacement for ORBextractor::operator()
    // nfeatures: max keypoints, scaleFactor: unused, nlevels: pyramid levels
    // iniThFAST: repurposed as confidence threshold, minThFAST: repurposed as NMS radius
    int operator()(cv::InputArray _image, cv::InputArray _mask,
                   std::vector<cv::KeyPoint> &_keypoints,
                   cv::OutputArray _descriptors,
                   std::vector<int> &vLappingArea);

    int inline GetLevels(){ return nlevels; }
    float inline GetScaleFactor(){ return scaleFactor; }
    std::vector<float> inline GetScaleFactors(){ return mvScaleFactor; }
    std::vector<float> inline GetInverseScaleFactors(){ return mvInvScaleFactor; }
    std::vector<float> inline GetScaleSigmaSquares(){ return mvLevelSigma2; }
    std::vector<float> inline GetInverseScaleSigmaSquares(){ return mvInvLevelSigma2; }

    // Image pyramid for stereo sub-pixel matching compatibility
    std::vector<cv::Mat> mvImagePyramid;

protected:

    void ComputePyramid(cv::Mat image);

    int nfeatures;
    double scaleFactor;
    int nlevels;

    // SuperPoint-specific parameters
    float mfConfThreshold;   // keypoint confidence threshold (from iniThFAST)
    float mfNmsRadius;       // NMS radius (from minThFAST)
    int mnDescriptorDim;     // 256

    // Scale pyramid compatibility
    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    // ONNX Runtime
    Ort::Env *mpEnv;
    Ort::Session *mpSession;
    Ort::SessionOptions mSessionOptions;
    Ort::AllocatorWithDefaultOptions mAllocator;

    std::string mInputName;
    std::vector<const char*> mOutputNames;
    std::vector<std::string> mOutputNameStrings;
};

} // namespace ORB_SLAM3

#endif
