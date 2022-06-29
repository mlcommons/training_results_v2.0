#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL _postprocess_ARRAY_API
#include "/usr/local/lib/python3.7/dist-packages/numpy/core/include/numpy/arrayobject.h"

#include "postprocess/segmentation.h"
#include <memory>
#include <iostream>
#include <string>
#include "/usr/include/opencv4/opencv2/core.hpp"
#include "/usr/include/opencv4/opencv2/imgproc.hpp"
#include "/usr/local/lib/python3.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h"
// #include "third_party/py/numpy/core/include/numpy/ndarraytypes.h"
#include "/usr/local/lib/python3.7/dist-packages/numpy/core/include/numpy/npy_common.h"
//#include "third_party/py/numpy/core/include/numpy/npy_common.h"
#include "postprocess/segmentation_results.h"
// #include "thread/fiber/bundle.h"
// #include "third_party/tensorflow/core/profiler/lib/traceme.h"


extern "C" {
#include "maskApi.h"
}

#define USE_RLE_ENCODE_SPATIAL_BINARY

namespace segmentation {

MaskProcessor::MaskProcessor(
    int maskIndex, float* maskData, int* refBoxesData,
    int numMasks, int maskSize, int imgHeight, int imgWidth) {
  imgHeight_ = imgHeight;
  imgWidth_ = imgWidth;
  maskSize_ = maskSize;
  maskIndex_ = maskIndex;
  refBoxes_ = refBoxesData;
  masks_ = maskData;
}

template<class T>
T* MaskProcessor::allocateMatrix(size_t w, size_t h) {
  // Create a contiguous matrix of size w x h initialized to 0s.
  // tensorflow::profiler::TraceMe t("MaskProcessor::allocateMatrix");
  return new T[w * h]();
}

void MaskProcessor::CreatePaddedMask() {
  // tensorflow::profiler::TraceMe t("MaskProcessor::CreatePaddedMask");
  float* mask = masks_ + maskIndex_ * maskSize_ * maskSize_;
  for (auto i = 1; i < maskSize_ + 1; ++i) {
    auto paddedMaskOffset = i * (maskSize_ + 2) + 1;
    auto maskOffset = (i - 1) * (maskSize_);

    std::memcpy(paddedMask_ + paddedMaskOffset, mask + maskOffset,
                sizeof(float) * maskSize_);
  }
}

void MaskProcessor::ResizeMask(int resizedWidth, int resizedHeight) {
  // First resize paddedMask_ using OpenCV
  // tensorflow::profiler::TraceMe t("MaskProcessor::ResizeMask");
  cv::Mat srcMat = cv::Mat(
      maskSize_ + 2, maskSize_ + 2, CV_32F, paddedMask_);
  {
    //tensorflow::profiler::TraceMe t2("MaskProcessor::ResizeMask::Cv2Resize");
    cv::resize(
        srcMat, resizedMat_,
        cv::Size(resizedWidth, resizedHeight), 0, 0, cv::INTER_LINEAR);
  }

  // After cv::Resize, switch to fortran ordering for consistency with
  // the original Python implementation.
  {
    //tensorflow::profiler::TraceMe t2("MaskProcessor::ResizeMask::Reshape");
    resizedMat_ = resizedMat_.reshape(0, {resizedHeight, resizedWidth});
  }

  resizedMask_ = (float*)resizedMat_.data;
}

void MaskProcessor::CreateImageMask(
    int resizedWidth, int resizedHeight,
    int x, int y, int x_0, int x_1, int y_0, int y_1) {
  //tensorflow::profiler::TraceMe t("MaskProcessor::CreateImageMask");

  for (int i = 0; i < y_1 - y_0; ++i) {
    for (int j = 0; j < x_1 - x_0; ++j) {
      uint8_t element_value = 0;

      auto x_coord = (i + y_0 - y) % resizedHeight;
      auto y_coord = (j + x_0 - x) % resizedWidth;
      float value = resizedMask_[x_coord * resizedWidth + y_coord];
      if (value > 0.5) {
        element_value = 1;
      }

      // RLE encoding expects Fortran ordering rather than C-style ordering.
      imageMask_[(y_0 + i) + imgHeight_ * (x_0 + j)] = element_value;
    }
  }
}

int MaskProcessor::GetBoxValue(int position) {
  return refBoxes_[maskIndex_ * 4 + position];
}

std::shared_ptr<RLEType> MaskProcessor::ProcessMask() {
  //tensorflow::profiler::TraceMe t("MaskProcessor::ProcessMask");
  std::shared_ptr<RLEType> result;

  paddedMask_ = allocateMatrix<float>(maskSize_ + 2, maskSize_ + 2);
  imageMask_ = allocateMatrix<uint8_t>(imgHeight_, imgWidth_);
  CreatePaddedMask();
  RLE rle = RLE();

  int x = GetBoxValue(0);
  int y = GetBoxValue(1);
  int w = GetBoxValue(2);
  int h = GetBoxValue(3);

  auto resizedWidth = std::max(w - x + 1, 1);
  auto resizedHeight = std::max(h - y + 1, 1);

  auto x_0 = std::max(x, 0);
  auto x_1 = std::min(w + 1, imgWidth_);
  auto y_0 = std::max(y, 0);
  auto y_1 = std::min(h + 1, imgHeight_);

  ResizeMask(resizedWidth, resizedHeight);
  CreateImageMask(resizedWidth, resizedHeight, x, y, x_0, x_1, y_0, y_1);

  //tensorflow::profiler::TraceMe t1("MaskProcessor::ComputeRLE");
#ifdef USE_RLE_ENCODE_SPATIAL_BINARY
  auto cnts = new unsigned int[imgHeight_ * imgWidth_ + 1]();
  rleInitNoMalloc(&rle, 0, 0, 0, cnts);
  rleEncodeSpatialBinary(&rle, (const byte*)imageMask_, imgHeight_, imgWidth_);
#else
  rleInit(&rle, 0, 0, 0, 0);
  rleEncode(&rle, (const byte*)imageMask_, imgHeight_, imgWidth_, 1);
#endif  // USE_RLE_ENCODE_SPATIAL_BINARY

  result = std::make_shared<RLEType>(
      std::make_pair(std::make_pair(rle.h, rle.w), rleToString(&rle)));

  delete[] paddedMask_;
  delete[] imageMask_;
#ifdef USE_RLE_ENCODE_SPATIAL_BINARY
  delete[] cnts;
#endif  // USE_RLE_ENCODE_SPATIAL_BINARY
  return result;
}

MaskProcessorNoMalloc::MaskProcessorNoMalloc(
    int maskIndex, float* maskData, int* refBoxesData, int numMasks,
    int maskSize, int imgHeight, int imgWidth, uint8_t* imageMaskBuffer,
    float* paddedMaskBuffer, unsigned int* rleCntsBuffer)
    : MaskProcessor(maskIndex, maskData, refBoxesData, numMasks, maskSize,
                    imgHeight, imgWidth) {
  imageMask_ = imageMaskBuffer;
  paddedMask_ = paddedMaskBuffer;
  rleCntsBuffer_ = rleCntsBuffer;
}

std::shared_ptr<RLEType> MaskProcessorNoMalloc::ProcessMask() {
  //tensorflow::profiler::TraceMe t("MaskProcessor::ProcessMask");
  std::shared_ptr<RLEType> result;
  CreatePaddedMask();
  RLE rle = RLE();

  int x = GetBoxValue(0);
  int y = GetBoxValue(1);
  int w = GetBoxValue(2);
  int h = GetBoxValue(3);

  auto resizedWidth = std::max(w - x + 1, 1);
  auto resizedHeight = std::max(h - y + 1, 1);

  auto x_0 = std::max(x, 0);
  auto x_1 = std::min(w + 1, imgWidth_);
  auto y_0 = std::max(y, 0);
  auto y_1 = std::min(h + 1, imgHeight_);

  ResizeMask(resizedWidth, resizedHeight);
  CreateImageMask(resizedWidth, resizedHeight, x, y, x_0, x_1, y_0, y_1);

  // tensorflow::profiler::TraceMe t1("MaskProcessor::ComputeRLE");
#ifdef USE_RLE_ENCODE_SPATIAL_BINARY
  rleInitNoMalloc(&rle, 0, 0, 0, rleCntsBuffer_);
  rleEncodeSpatialBinary(&rle, (const byte*)imageMask_, imgHeight_, imgWidth_);
#else
  rleInit(&rle, 0, 0, 0, 0);
  rleEncode(&rle, (const byte*)imageMask_, imgHeight_, imgWidth_, 1);
#endif  // USE_RLE_ENCODE_SPATIAL_BINARY

  result = std::make_shared<RLEType>(
      std::make_pair(std::make_pair(rle.h, rle.w), rleToString(&rle)));
  return result;
}

PyObject* SegmentationProcessor::ProcessIndividualMask(
    PyObject* maskObject, int maskIndex,
    PyObject* refBoxesObject, int imgHeight, int imgWidth) {
  auto maskShape = PyArray_SHAPE(reinterpret_cast<PyArrayObject*>(maskObject));
  int numMasks = maskShape[0];
  int maskSize = maskShape[2];

  MaskProcessor processor = MaskProcessor(
      maskIndex, static_cast<float*>(PyArray_DATA(maskObject)),
      static_cast<int*>(PyArray_DATA(refBoxesObject)),
      numMasks, maskSize, imgHeight, imgWidth);
  return SegmentationResults::ConvertRLEToDict(processor.ProcessMask());
}

}  // namespace segmentation

