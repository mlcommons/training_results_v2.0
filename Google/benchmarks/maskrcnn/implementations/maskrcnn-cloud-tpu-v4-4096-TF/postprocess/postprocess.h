#ifndef THIRD_PARTY_TENSORFLOW_MODELS_MLPERF_MODELS_ROUGH_MASK_RCNN_SEGMENTATION_POSTPROCESS_H_
#define THIRD_PARTY_TENSORFLOW_MODELS_MLPERF_MODELS_ROUGH_MASK_RCNN_SEGMENTATION_POSTPROCESS_H_

#include "/usr/local/lib/python3.7/dist-packages/numpy/core/include/numpy/arrayobject.h"
#include "postprocess/segmentation_results.h"
#include "postprocess/segmentation.h"

#include <thread>
#include <vector>

namespace segmentation {

class PostProcessor {
 public:
  PostProcessor(PyObject* boxObject, PyObject* maskObject,
                PyObject* imageInfoObject, PyObject* numWorkers);

  ~PostProcessor();

  void ProcessAllSamples();
  void Start();
  std::pair<PyList, PyList> GetResults();

 private:
  template<class T>
  PyArrayObject* CopyNumpyArrayToMemory(
      T** buffer, PyObject* npyObject, int area);

  void ProcessSamples(int startIndex, int endIndex);

  void ProcessAndExpandBoxes(
      int* expandedBoxes, int* numValidBoxes, float* boxes,
      std::vector<DetectionObject*>& detections);
  /*void ProcessMasksConcurrently(
    int sampleIndex, int numValidBoxes, int maskOffset, int* expandedBoxes,
    int imageHeight, int imageWidth, std::vector<DetectionObject*>& detections);
    */
  void ProcessMasksSequentially(
    int sampleIndex, int numValidBoxes, int maskOffset, int* expandedBoxes,
    int imageHeight, int imageWidth, std::vector<DetectionObject*>& detections);

  PyArrayObject* boxObject_;
  PyArrayObject* maskObject_;
  PyArrayObject* imageInfoObject_;

  int numSamples_;
  int numBoxes_;
  int numMasks_;
  int maskSize_;

  int maskSampleArea_;
  int boxSampleArea_;
  int imageInfoSampleArea_;

  SegmentationResults* resultsHandler_;
  // std::unique_ptr<std::thread::Fiber> fiberTree_;
  float* boxes_;
  float* masks_;
  int* imageInfos_;
  int numWorkers_;
};

class EvalPostprocessor{
 public:
  std::pair<PyList, PyList> PostProcess(PyObject* boxObject,
                                        PyObject* maskObject,
                                        PyObject* imageInfoObject,
                                        PyObject* numWorkers);
};

}  // namespace segmentation



#endif  // THIRD_PARTY_TENSORFLOW_MODELS_MLPERF_MODELS_ROUGH_MASK_RCNN_SEGMENTATION_POSTPROCESS_H_

