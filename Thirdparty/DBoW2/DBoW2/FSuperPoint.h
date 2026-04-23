/**
 * FSuperPoint.h
 * DBoW2 descriptor class for SuperPoint 256-dim float L2-normalized descriptors.
 */

#ifndef __D_T_F_SUPERPOINT__
#define __D_T_F_SUPERPOINT__

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdlib>

#include "FClass.h"

namespace DBoW2 {

class FSuperPoint: protected FClass
{
public:

  typedef cv::Mat TDescriptor;  // CV_32F, 1 row x 256 cols, L2-normalized
  typedef const TDescriptor *pDescriptor;
  static const int L = 256;

  static void meanValue(const std::vector<pDescriptor> &descriptors,
    TDescriptor &mean);

  static float distance(const TDescriptor &a, const TDescriptor &b);

  static std::string toString(const TDescriptor &a);

  static void fromString(TDescriptor &a, const std::string &s);

  static void toMat32F(const std::vector<TDescriptor> &descriptors,
    cv::Mat &mat);

  static void toMat8U(const std::vector<TDescriptor> &descriptors,
    cv::Mat &mat);
};

} // namespace DBoW2

#endif
