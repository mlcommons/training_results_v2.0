#include "pycocotools/cocoeval_functional.h"

#include <algorithm>

namespace pycocotools {

COCOevalFunctional::COCOevalFunctional(std::vector<Annotation*> gt_annotations,
                                       std::vector<Annotation*> dt_annotations)
    : COCOevalBase("bbox"),
      gt_annotations_(gt_annotations),
      dt_annotations_(dt_annotations) {
  std::set<int64_t> image_ids;
  std::set<int64_t> category_ids;

  auto get_ids = [&image_ids,
                  &category_ids](const std::vector<Annotation*> annotations) {
    for (const auto* annotation : annotations) {
      image_ids.insert(annotation->image_id);
      category_ids.insert(annotation->category_id);
    }
  };

  get_ids(gt_annotations);
  get_ids(dt_annotations);

  std::copy(image_ids.begin(), image_ids.end(),
            std::back_inserter(params_.img_ids));
  std::copy(category_ids.begin(), category_ids.end(),
            std::back_inserter(params_.cat_ids));
  std::stable_sort(params_.cat_ids.begin(), params_.cat_ids.end());
  max_category_id_ = params_.cat_ids.back();
}

void COCOevalFunctional::Prepare() {
  for (auto* gt : gt_annotations_) {
    gt->ignore |= gt->iscrowd;
  }

  gts_.clear();
  dts_.clear();

  for (auto* gt : gt_annotations_) {
    gts_[Key(gt->image_id, gt->category_id)].push_back(gt);
  }

  for (auto* dt : dt_annotations_) {
    dts_[Key(dt->image_id, dt->category_id)].push_back(dt);
  }

  eval_imgs_.clear();
}
}  // namespace pycocotools
