#ifndef THIRD_PARTY_PY_PYCOCOTOOLS_COCOEVAL_FUNCTIONAL_H_
#define THIRD_PARTY_PY_PYCOCOTOOLS_COCOEVAL_FUNCTIONAL_H_

#include "pycocotools/annotation.h"
#include "pycocotools/cocoeval_base.h"

namespace pycocotools {

// COCOEval based only on Annotations passed to constructor.
// This lets you to get COCO metrics without having to link in Python/JSON
// dependencies.
class COCOevalFunctional : public COCOevalBase {
 public:
  COCOevalFunctional(std::vector<Annotation*> gt_annotations,
                     std::vector<Annotation*> dt_annotations);

 private:
  // Prepare gts_ and dts_ for evaluation based on params
  // Only consider "bbox" type.
  void Prepare() override;

  std::vector<Annotation*> gt_annotations_;
  std::vector<Annotation*> dt_annotations_;
};

}  // namespace pycocotools
#endif  // THIRD_PARTY_PY_PYCOCOTOOLS_COCOEVAL_FUNCTIONAL_H_
