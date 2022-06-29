#ifndef THIRD_PARTY_TENSORFLOW_MODELS_MLPERF_MODELS_ROUGH_MASK_RCNN_SEGMENTATION_SEGMENTATION_RESULTS_H_
#define THIRD_PARTY_TENSORFLOW_MODELS_MLPERF_MODELS_ROUGH_MASK_RCNN_SEGMENTATION_SEGMENTATION_RESULTS_H_

// #include "third_party/absl/container/flat_hash_map.h"
// #include "third_party/absl/synchronization/mutex.h"
#include <thread>
#include <mutex>
#include "/usr/local/lib/python3.7/dist-packages/numpy/core/include/numpy/arrayobject.h"
#include <string>
#include <vector>
#include <array>

#define MAX_TOTAL_SAMPLES 500000

typedef std::pair<std::pair<int, int>, std::string> RLEType;
typedef std::vector<PyObject*> PyList;

namespace segmentation {

struct DetectionObject {
  float i0;
  float i1;
  float i2;
  float i3;
  float i4;
  float i5;
  float i6;
};

class SegmentationResults {
  // Simple threadsafe helper class to store RLE results.

 public:
  std::pair<PyList, PyList> results();

  void AddResult(int resultIndex, std::shared_ptr<RLEType> result,
                 DetectionObject* detection);

  static PyObject* ConvertDetectionObjectToObj(DetectionObject b) {
    npy_intp dims[1] = {7};
    PyObject* result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    *reinterpret_cast<float*>(PyArray_GETPTR1(result, 0)) = b.i0;
    *reinterpret_cast<float*>(PyArray_GETPTR1(result, 1)) = b.i1;
    *reinterpret_cast<float*>(PyArray_GETPTR1(result, 2)) = b.i2;
    *reinterpret_cast<float*>(PyArray_GETPTR1(result, 3)) = b.i3;
    *reinterpret_cast<float*>(PyArray_GETPTR1(result, 4)) = b.i4;
    *reinterpret_cast<float*>(PyArray_GETPTR1(result, 5)) = b.i5;
    *reinterpret_cast<float*>(PyArray_GETPTR1(result, 6)) = b.i6;
    return result;
  }

  static PyObject* ConvertRLEToDict(std::shared_ptr<RLEType> rle) {
    PyObject* result = PyDict_New();
    PyObject* size = PyList_New(2);
    PyList_SetItem(size, 0, PyLong_FromLong(rle->first.first));
    PyList_SetItem(size, 1, PyLong_FromLong(rle->first.second));
    PyDict_SetItemString(result, "size", size);
    PyObject* count_string = PyUnicode_FromString(rle->second.c_str());
    PyDict_SetItemString(result, "counts", count_string);
    Py_DecRef(size);
    Py_DecRef(count_string);
    return result;
  }

 private:
  std::array<std::mutex, MAX_TOTAL_SAMPLES> mutex_;
  std::array<bool, MAX_TOTAL_SAMPLES> accessed_ = { false };
  std::array<std::shared_ptr<RLEType>, MAX_TOTAL_SAMPLES> rles_;
  std::array<DetectionObject, MAX_TOTAL_SAMPLES> detections_;
};

}  // namespace segmentation

#endif  // THIRD_PARTY_TENSORFLOW_MODELS_MLPERF_MODELS_ROUGH_MASK_RCNN_SEGMENTATION_SEGMENTATION_RESULTS_H_

