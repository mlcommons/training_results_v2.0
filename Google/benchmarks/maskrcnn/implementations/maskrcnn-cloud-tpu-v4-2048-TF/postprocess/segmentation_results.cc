#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL _postprocess_ARRAY_API
#include "/usr/local/lib/python3.7/dist-packages/numpy/core/include/numpy/arrayobject.h"

#include "postprocess/segmentation_results.h"
#include "/usr/local/lib/python3.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h"
#include "/usr/local/lib/python3.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h"
#include "/usr/local/lib/python3.7/dist-packages/numpy/core/include/numpy/npy_common.h"

/*#include "third_party/py/numpy/core/include/numpy/ndarrayobject.h"
#include "third_party/py/numpy/core/include/numpy/ndarraytypes.h"
#include "third_party/py/numpy/core/include/numpy/npy_common.h"
#include "third_party/tensorflow/core/profiler/lib/traceme.h"
*/
#include <mutex>
#include <iostream>

namespace segmentation {

std::pair<PyList, PyList> SegmentationResults::results() {
  //tensorflow::profiler::TraceMe t("SegmentationResults::toList");
  std::vector<PyObject*> detectionResults;
  std::vector<PyObject*> rleResults;

  for (int resultIndex = 0; resultIndex < MAX_TOTAL_SAMPLES; ++resultIndex) {
    mutex_[resultIndex].lock();
    if (accessed_[resultIndex]) {
      rleResults.push_back(ConvertRLEToDict(rles_[resultIndex]));
      detectionResults.push_back(
          ConvertDetectionObjectToObj(detections_[resultIndex]));
    }
    mutex_[resultIndex].unlock();
  }
  return std::make_pair(detectionResults, rleResults);
}

void SegmentationResults::AddResult(
    int resultIndex,
    std::shared_ptr<RLEType> result,
    DetectionObject* detection) {
  //tensorflow::profiler::TraceMe t("SegmentationResults::AddResult");
  assert(resultIndex < MAX_TOTAL_SAMPLES);
  mutex_[resultIndex].lock();
  // std::mutex lock(&mutex_[resultIndex]);
  accessed_[resultIndex] = true;
  rles_[resultIndex] = std::move(result);
  detections_[resultIndex] = *detection;
  mutex_[resultIndex].unlock();
}

}  // namespace segmentation
