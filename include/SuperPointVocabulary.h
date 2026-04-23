#ifndef SUPERPOINT_VOCABULARY_H
#define SUPERPOINT_VOCABULARY_H

#include "Thirdparty/DBoW2/DBoW2/FSuperPoint.h"
#include "Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h"

namespace ORB_SLAM3
{

typedef DBoW2::TemplatedVocabulary<DBoW2::FSuperPoint::TDescriptor, DBoW2::FSuperPoint>
  SuperPointVocabulary;

} // namespace ORB_SLAM3

#endif // SUPERPOINT_VOCABULARY_H
