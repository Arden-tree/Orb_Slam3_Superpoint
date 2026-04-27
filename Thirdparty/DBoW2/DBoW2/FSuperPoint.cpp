/**
 * FSuperPoint.cpp
 * DBoW2 descriptor class for SuperPoint 256-dim float L2-normalized descriptors.
 */

#include "FSuperPoint.h"
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace std;

namespace DBoW2 {

void FSuperPoint::meanValue(const std::vector<pDescriptor> &descriptors,
  TDescriptor &mean)
{
  // When cluster has no members, keep existing cluster center unchanged.
  // This prevents empty Mat from destroying a valid cluster center,
  // which would cause distance() to crash with size mismatch.
  if(descriptors.empty())
  {
    return;
  }

  mean = cv::Mat::zeros(1, L, CV_32F);
  float *meanPtr = mean.ptr<float>();

  for(size_t i = 0; i < descriptors.size(); i++)
  {
    const cv::Mat &desc = *descriptors[i];

    // Handle row descriptors: desc can be 1x256 or Nx256, use first row
    int cols = desc.cols;
    const float *rowPtr = desc.ptr<float>(0);
    for(int j = 0; j < L && j < cols; j++)
    {
      meanPtr[j] += rowPtr[j];
    }
  }

  // Divide by N
  for(int j = 0; j < L; j++)
  {
    meanPtr[j] /= descriptors.size();
  }

  // L2-normalize the mean
  float norm = 0;
  for(int j = 0; j < L; j++)
  {
    norm += meanPtr[j] * meanPtr[j];
  }
  norm = sqrt(norm);
  if(norm > 1e-6)
  {
    for(int j = 0; j < L; j++)
    {
      meanPtr[j] /= norm;
    }
  }
}

float FSuperPoint::distance(const TDescriptor &a, const TDescriptor &b)
{
  // For L2-normalized vectors: L2^2 = 2 - 2*dot(a,b)
  // Raw pointer loop avoids cv::Mat::dot() overhead:
  //   - no type inference / size check / boundary assertion
  //   - no temporary matrix allocation
  //   - tight loop, compiler-friendly for auto-vectorization (NEON on ARM)
  if(a.empty() || b.empty() || a.size() != b.size())
    return 1.414f;  // max L2 distance for normalized vectors

  const float *pa = a.ptr<float>(0);
  const float *pb = b.ptr<float>(0);
  float dot = 0;
  for(int i = 0; i < L; i++)
  {
    dot += pa[i] * pb[i];
  }

  float dist = 2.0f - 2.0f * dot;
  if(dist < 0) dist = 0;  // numerical stability
  return sqrtf(dist);
}

string FSuperPoint::toString(const TDescriptor &a)
{
  stringstream ss;
  const float *p = a.ptr<float>();
  for(int i = 0; i < L; i++)
  {
    ss << p[i];
    if(i < L - 1) ss << " ";
  }
  return ss.str();
}

void FSuperPoint::fromString(TDescriptor &a, const string &s)
{
  a.create(1, L, CV_32F);
  float *p = a.ptr<float>();
  stringstream ss(s);
  for(int i = 0; i < L; i++)
  {
    ss >> p[i];
  }
}

void FSuperPoint::toMat32F(const std::vector<TDescriptor> &descriptors,
  cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }

  const int N = descriptors.size();
  mat.create(N, L, CV_32F);

  for(int i = 0; i < N; i++)
  {
    descriptors[i].row(0).copyTo(mat.row(i));
  }
}

void FSuperPoint::toMat8U(const std::vector<TDescriptor> &descriptors,
  cv::Mat &mat)
{
  // Cannot losslessly convert 256 float values to 8U.
  // Scale each float from [0,1] (L2-normalized range per element) to [0,255]
  if(descriptors.empty())
  {
    mat.release();
    return;
  }

  const int N = descriptors.size();
  mat.create(N, L, CV_8U);

  for(int i = 0; i < N; i++)
  {
    const float *pSrc = descriptors[i].ptr<float>();
    uchar *pDst = mat.ptr<uchar>(i);
    for(int j = 0; j < L; j++)
    {
      float v = pSrc[j];
      // L2-normalized vectors can have negative values
      // Map from approx [-0.2, 0.2] -> [0, 255]
      v = (v + 0.2f) / 0.4f;  // normalize to ~[0,1]
      v = v < 0 ? 0 : (v > 1 ? 1 : v);
      pDst[j] = (uchar)(v * 255.0f);
    }
  }
}

} // namespace DBoW2
