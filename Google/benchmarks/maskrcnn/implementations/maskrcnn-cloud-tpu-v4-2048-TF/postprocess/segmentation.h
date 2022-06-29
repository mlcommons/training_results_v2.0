// This is a C++ rewrite of the segmentation postprocessing pipeline for
// COCO results.
//
// This implements the logic in C++ and takes advantage of C++ threading to
// bypass the Python GIL for processing masks concurrently.
#ifndef THIRD_PARTY_TENSORFLOW_MODELS_MLPERF_MODELS_ROUGH_MASK_RCNN_SEGMENTATION_SEGMENTATION_H_
#define THIRD_PARTY_TENSORFLOW_MODELS_MLPERF_MODELS_ROUGH_MASK_RCNN_SEGMENTATION_SEGMENTATION_H_

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "/usr/include/opencv4/opencv2/core.hpp"
#include "/usr/include/opencv4/opencv2/imgproc.hpp"
#include "/usr/local/lib/python3.7/dist-packages/numpy/core/include/numpy/arrayobject.h"
#include "postprocess/segmentation_results.h"

namespace segmentation {

class MaskProcessor {
  // This class encapsulates functionality of computing the RLE of a single mask
  // in pure C++. This allows us to enable multithreading and avoid the GIL.
 public:
  MaskProcessor(
      int maskIndex, float* maskData, int* refBoxesData,
      int numMasks, int maskSize, int imgHeight, int imgWidth);

  template<class T>
  T* allocateMatrix(size_t w, size_t h);

  std::shared_ptr<RLEType> ProcessMask();

 protected:
  void CreatePaddedMask();
  void ResizeMask(int resizedWidth, int resizedHeight);

  int GetBoxValue(int position);

  void CreateImageMask(
      int resizedWidth, int resizedHeight,
      int x, int y, int x_0, int x_1, int y_0, int y_1);

  int maskIndex_;
  int maskSize_;

  int imgHeight_;
  int imgWidth_;
  int numMasks_;

  int* refBoxes_;
  float* masks_;

  float* paddedMask_;
  float* resizedMask_;
  uint8_t* imageMask_;
  cv::Mat resizedMat_;
};


class MaskProcessorNoMalloc: MaskProcessor {
 public:
  MaskProcessorNoMalloc(int maskIndex, float* maskData, int* refBoxesData,
                        int numMasks, int maskSize, int imgHeight, int imgWidth,
                        uint8_t* imageMaskBuffer, float* paddedMaskBuffer,
                        unsigned int* rleCntsBuffer);

  std::shared_ptr<RLEType> ProcessMask();

 protected:
  unsigned int* rleCntsBuffer_;
};

class SegmentationProcessor {
 public:
  PyObject* ProcessIndividualMask(
      PyObject* maskObject, int maskIndex, PyObject* refBoxesObject,
      int imgHeight, int imgWidth);
};

}  // namespace segmentation


#endif  // THIRD_PARTY_TENSORFLOW_MODELS_MLPERF_MODELS_ROUGH_MASK_RCNN_SEGMENTATION_SEGMENTATION_H_

