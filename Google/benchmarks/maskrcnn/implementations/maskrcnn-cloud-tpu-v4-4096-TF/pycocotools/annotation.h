#ifndef THIRD_PARTY_PY_PYCOCOTOOLS_ANNOTATION_H_
#define THIRD_PARTY_PY_PYCOCOTOOLS_ANNOTATION_H_

#include <stdint.h>

#include <vector>

namespace pycocotools {

struct Annotation {
  float area = 0.0;
  bool iscrowd = false;
  int64_t image_id;
  int64_t category_id;
  int64_t id;
  // xmin, ymin, width, height
  std::vector<float> bbox;
  // Compressed RLE format
  std::vector<uint32_t> cnts;
  uint32_t h;
  uint32_t w;
  bool ignore = false;
  bool uignore = false;
  float score = 0.0;
};

}  // namespace pycocotools

#endif  // THIRD_PARTY_PY_PYCOCOTOOLS_ANNOTATION_H_
