#define PY_ARRAY_UNIQUE_SYMBOL _postprocess_ARRAY_API
#include "/usr/local/lib/python3.7/dist-packages/numpy/core/include/numpy/arrayobject.h"

#include "postprocess/postprocess.h"
#include "/usr/include/opencv4/opencv2/core/utility.hpp"
// #include "ird_party/tensorflow/core/profiler/lib/traceme.h"
#include "/usr/include/opencv4/opencv2/core.hpp"
#include "/usr/include/opencv4/opencv2/imgproc.hpp"
#include <string>
#include <iostream>

#define _DEFAULT_NUM_MASK_PROCESSING_THREADS 1

namespace segmentation {

int init_numpy() {
  std::cout << "PKKK init_numpy pre";
  import_array();  // PyError if not successful
  std::cout << "PKKK init_numpy post";
  return 1;
}
// it is only a trick to ensure import_array() is called, when *.so is loaded
// just called only once
const int numpy_initialized = init_numpy();

PostProcessor::PostProcessor(PyObject* boxObject, PyObject* maskObject,
                             PyObject* imageInfoObject, PyObject* numWorkers) {
  auto maskShape = PyArray_SHAPE(reinterpret_cast<PyArrayObject*>(maskObject));
  auto boxShape = PyArray_SHAPE(reinterpret_cast<PyArrayObject*>(boxObject));

  numSamples_ = maskShape[0];

  numBoxes_ = boxShape[1];
  numMasks_ = maskShape[1];
  maskSize_ = maskShape[3];

  numWorkers_ = static_cast<int>(PyLong_AsLong(numWorkers));
  // TODO: set?
  cv::setNumThreads(1);

  resultsHandler_ = new SegmentationResults();

  boxSampleArea_ = numBoxes_ * 7;
  maskSampleArea_ = numMasks_ * maskSize_ * maskSize_;
  imageInfoSampleArea_ = 5;

  maskObject_ = CopyNumpyArrayToMemory<float>(
      &masks_, maskObject, numSamples_ * maskSampleArea_);
  imageInfoObject_ = CopyNumpyArrayToMemory<int>(
      &imageInfos_, imageInfoObject, numSamples_ * imageInfoSampleArea_);
  boxObject_ = CopyNumpyArrayToMemory<float>(
      &boxes_, boxObject, numSamples_ * boxSampleArea_);
}

PostProcessor::~PostProcessor() {
  Py_DecRef(reinterpret_cast<PyObject*>(maskObject_));
  Py_DecRef(reinterpret_cast<PyObject*>(imageInfoObject_));
  Py_DecRef(reinterpret_cast<PyObject*>(boxObject_));
  delete resultsHandler_;
}

void PostProcessor::ProcessAndExpandBoxes(
    int* expandedBoxes, int* numValidBoxes, float* boxes,
    std::vector<DetectionObject*>& detections) {
  //tensorflow::profiler::TraceMe t("ProcessAndExpandBoxes");
  auto scale = (maskSize_ + 2.) / maskSize_;
  for (int i = 0; i < numBoxes_; i++) {
    float x = boxes[i * 7 + 1];
    float y = boxes[i * 7 + 2];
    float w = boxes[i * 7 + 3];
    float h = boxes[i * 7 + 4];
    float score = boxes[i * 7 + 5];

    //if (x == 0 && y == 0 && w == 0 && h == 0) {
    if (i > 0 && score == 0) {
      *numValidBoxes = i;
      return;
    }
  
    detections.emplace_back(reinterpret_cast<DetectionObject*>(&boxes[i * 7]));


    float w_half = w * 0.5;
    float h_half = h * 0.5;
    float x_c = x + w_half;
    float y_c = y + h_half;

    w_half *= scale;
    h_half *= scale;

    expandedBoxes[i * 4] = static_cast<int>(x_c - w_half);
    expandedBoxes[i * 4 + 2] = static_cast<int>(x_c + w_half);
    expandedBoxes[i * 4 + 1] = static_cast<int>(y_c - h_half);
    expandedBoxes[i * 4 + 3] = static_cast<int>(y_c + h_half);
  }
  *numValidBoxes = numBoxes_;
}

void PostProcessor::ProcessMasksSequentially(
    int sampleIndex, int numValidBoxes, int maskOffset, int* expandedBoxes,
    int imageHeight, int imageWidth, std::vector<DetectionObject*>& detections) {
  //tensorflow::profiler::TraceMe t("ProcessMasksSequentially");

  float* paddedMaskBuffer;
  uint8_t* imageMaskBuffer;
  unsigned int* rleCntsBuffer;
  int imageArea = imageHeight * imageWidth;
  {
    //tensorflow::profiler::TraceMe t("Allocate buffers");
    paddedMaskBuffer = new float[(maskSize_ + 2)*(maskSize_ + 2)]();
    imageMaskBuffer = new uint8_t[imageArea]();
    rleCntsBuffer = new unsigned int[imageArea + 1]();
  }

  for (int maskIndex = 0; maskIndex < std::min(numMasks_, numValidBoxes);
    ++maskIndex) {
    MaskProcessorNoMalloc processor =
        MaskProcessorNoMalloc(maskIndex, masks_ + maskOffset, expandedBoxes,
                              numMasks_, maskSize_, imageHeight, imageWidth,
                              imageMaskBuffer, paddedMaskBuffer, rleCntsBuffer);
    auto results = processor.ProcessMask();
    this->resultsHandler_->AddResult(
        sampleIndex * numMasks_ + maskIndex, results, detections[maskIndex]);
    {
      //tensorflow::profiler::TraceMe t("Reset buffers");
      std::fill(imageMaskBuffer, imageMaskBuffer + imageArea, 0);
    }
  }
  delete[] paddedMaskBuffer;
  delete[] imageMaskBuffer;
  delete[] rleCntsBuffer;
}

/*void PostProcessor::ProcessMasksConcurrently(
    int sampleIndex, int numValidBoxes, int maskOffset, int* expandedBoxes,
    int imageHeight, int imageWidth, std::vector<DetectionObject>& detections) {
  int numSamplesToProcess = std::min(numMasks_, numBoxes_);
  int numThreads = std::min(numSamplesToProcess,
                            _DEFAULT_NUM_MASK_PROCESSING_THREADS);
  int samplesPerThread = ceil(
      static_cast<float>(numSamplesToProcess) / numThreads);
  std::thread::Bundle bundle;
  for (int i = 0; i < numThreads; ++i) {
    bundle.Add([this, i, samplesPerThread, sampleIndex, numValidBoxes,
               imageWidth, imageHeight, maskOffset, expandedBoxes, detections] {
      int startIndex = i * samplesPerThread;
      int endIndex = (i + 1) * samplesPerThread;

      for (int maskIndex = startIndex; maskIndex < endIndex; ++maskIndex) {
        if (maskIndex >= numMasks_ || maskIndex >= numValidBoxes) {
          break;
        }
        MaskProcessor processor = MaskProcessor(
            maskIndex, masks_ + maskOffset, expandedBoxes,
            numMasks_, maskSize_, imageHeight, imageWidth);
        auto results = processor.ProcessMask();
        this->resultsHandler_->AddResult(
            sampleIndex * numMasks_ + maskIndex,
            results, detections[maskIndex]);
      }
    });
  }
  bundle.JoinAll();
}
*/

void PostProcessor::ProcessAllSamples() {
  ProcessSamples(0, numSamples_);
}

void PostProcessor::ProcessSamples(int startIndex, int endIndex) {
  for (int sampleIndex = startIndex; sampleIndex < endIndex; ++sampleIndex) {
    if (sampleIndex >= numSamples_)
      break;

    int boxesOffset = sampleIndex * boxSampleArea_;
    int maskOffset = sampleIndex * maskSampleArea_;
    int imageInfoOffset = sampleIndex * imageInfoSampleArea_;
    int imageHeight = imageInfos_[imageInfoOffset + 3];
    int imageWidth = imageInfos_[imageInfoOffset + 4];
    int numValidBoxes = 0;
    int* expandedBoxes = new int[numBoxes_ * 4];
    std::vector<DetectionObject*> detections;
    detections.reserve(numBoxes_);
    ProcessAndExpandBoxes(
        expandedBoxes, &numValidBoxes, boxes_ + boxesOffset, detections);

    if (_DEFAULT_NUM_MASK_PROCESSING_THREADS == 1) {
      this->ProcessMasksSequentially(
          sampleIndex, numValidBoxes, maskOffset, expandedBoxes,
          imageHeight, imageWidth, detections);
    /*} else {
      this->ProcessMasksConcurrently(
          sampleIndex, numValidBoxes, maskOffset, expandedBoxes,
          imageHeight, imageWidth, detections);
    */
    }
    delete[] expandedBoxes;
  }
}

void PostProcessor::Start() {
  // tensorflow::profiler::TraceMe t("PostProcessor::Start");
  //std::thread::TreeOptions options;

  int sampleProcessingThreads = std::min(numSamples_, numWorkers_);

  //options.set_max_cpu_slots(
  //     _DEFAULT_NUM_MASK_PROCESSING_THREADS * sampleProcessingThreads);
  int samplesPerThread = ceil(
      static_cast<float>(numSamples_) / sampleProcessingThreads);

  Py_BEGIN_ALLOW_THREADS
  std::vector<std::thread> workers;
  for (int i = 0; i < sampleProcessingThreads; ++i) {
     workers.push_back(std::thread(
          [this, i, samplesPerThread] {
            int startIndex = i * samplesPerThread;
            int endIndex = (i + 1) * samplesPerThread;
            ProcessSamples(startIndex, endIndex);
          }
          )
        );
      }
  for (auto& worker: workers) {
     worker.join();
  }
  Py_END_ALLOW_THREADS
}

std::pair<PyList, PyList> PostProcessor::GetResults() {
  return resultsHandler_->results();
}

template<class T>
PyArrayObject* PostProcessor::CopyNumpyArrayToMemory(
    T** buffer, PyObject* npyObject, int area) {
  // tensorflow::profiler::TraceMe t("CopyNumpyArrayToMemory");

  PyArrayObject* array_object =
     PyArray_GETCONTIGUOUS(reinterpret_cast<PyArrayObject*>(npyObject));
  *buffer = static_cast<T*>(PyArray_DATA(array_object));
  return array_object;
}

std::pair<PyList, PyList> EvalPostprocessor::PostProcess(
    PyObject* boxObject, PyObject* maskObject, PyObject* imageInfoObject,
    PyObject* numWorkers) {
  PostProcessor processor =
      PostProcessor(boxObject, maskObject, imageInfoObject, numWorkers);
  processor.Start();
  return processor.GetResults();
}

}  // namespace segmentation
