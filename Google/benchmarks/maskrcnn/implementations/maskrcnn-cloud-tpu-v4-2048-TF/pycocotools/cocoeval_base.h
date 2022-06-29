#ifndef PYCOCOTOOLS_COCOEVAL_BASE_H_
#define PYCOCOTOOLS_COCOEVAL_BASE_H_

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "pycocotools/annotation.h"

namespace pycocotools {

typedef std::vector<std::vector<float>> IoUMatrix;

//  Params for coco evaluation api
struct Params {
 public:
  Params(std::string input_iou_type = "bbox");
  void SetDetParams();
  std::string iou_type;
  std::vector<int64_t> img_ids;
  std::vector<int64_t> cat_ids;
  std::vector<double> iou_thrs;
  std::vector<double> rec_thrs;
  std::vector<int64_t> max_dets;
  std::vector<std::vector<float>> area_rng;
  std::vector<std::string> area_rng_lbl;
  bool use_cats;
};

// Eval results for each image.
struct EvalImgs {
  int64_t image_id;
  int64_t category_id;
  int64_t max_det;
  // dt_matches - [TxD] matching gt id at each IoU or 0
  std::vector<int64_t> dt_matches;
  // dt_scores   - [1xD] confidence of each dt
  std::vector<double> dt_scores;
  // gt_ignore   - [1xG] ignore flag for each gt
  std::vector<bool> gt_ignore;
  // dt_ignore   - [TxD] ignore flag for each dt at each IoU
  std::vector<bool> dt_ignore;
};

struct Eval {
  Params params;
  std::vector<int64_t> counts;
  std::vector<float> precision;
  std::vector<float> recall;
};

// Interface for evaluating detection on the Microsoft COCO dataset.
//
// The usage for CocoEval is as follows:
//  COCO coco_gt=..., coco_dt =...       // load dataset and results
//  COCOeval E = Cocoeval(coco_gt,coco_dt); # initialize CocoEval object
//  E.Evaluate();                // run per image evaluation
//  E.Accumulate();              // accumulate per image results
//  E.Summarize();               // display summary metrics of results
//
// The evaluation parameters are as follows (defaults in brackets):
//  img_ids     - [all] N img ids to use for evaluation
//  cat_ids     - [all] K cat ids to use for evaluation
//  iou_thrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
//  rec_thrs    - [0:.01:1] R=101 recall thresholds for evaluation
//  area_rng    - [...] A=4 object area ranges for evaluation
//  max_dets    - [1 10 100] M=3 thresholds on max detections per image
//  iou_type    - ['bbox'] set iouType to 'segm', 'bbox' or 'keypoints'
//  use_cats    - [1] if true use category labels for evaluation
//  Note: if use_cats == 0 category labels are ignored as in proposal scoring.
//  Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
//
//  Evaluate(): evaluates detections on every image and every category and
//  concats the results into the "evalImgs" with fields:
//   dt_ids      - [1xD] id for each of the D detections (dt)
//   gt_ids      - [1xG] id for each of the G ground truths (gt)
//   dt_matches  - [TxD] matching gt id at each IoU or 0
//   gt_matches  - [TxG] matching dt id at each IoU or 0
//   dt_scores   - [1xD] confidence of each dt
//   gt_ignore  - [1xG] ignore flag for each gt
//   dt_ignore  - [TxD] ignore flag for each dt at each IoU
//
//  Accumulate(): accumulates the per-image, per-category evaluation
//  results in "evalImgs" into the dictionary "eval" with fields:
//   params     - parameters used for evaluation
//   counts     - [T,R,K,A,M] parameter dimensions (see above)
//   precision  - [TxRxKxAxM] precision for every evaluation setting
//   recall     - [TxKxAxM] max recall for every evaluation setting
//   Note: precision and recall==-1 for settings with no gt objects.
class COCOevalBase {
 public:
  // For subclasses, constructor should set up the params field.
  COCOevalBase(const std::string &iou_type) : params_(iou_type) {}

  virtual ~COCOevalBase() {}

  // Computes IoU of each predicted detections dt and ground truth
  // detections dt correspond to a img_id and cat_id,
  // complexity: O(dt.size() * gt.size()).
  IoUMatrix ComputeIoU(int64_t img_id, int64_t cat_id);

  // Evaluates detections on img_id and every cat_id on each threshold level and
  // concats the results into the `eval_imgs_`.
  // Complexity: O(T * D * G)
  // T, D, G see above.
  void EvaluateImg(int64_t img_id, int64_t cat_id,
                   const std::vector<std::vector<float>> &area_rng,
                   int64_t max_det, EvalImgs results[]);

  // Run per image evaluation on given images and store results in eval_imgs_
  // return: None
  // Complexity: O (#images * #categories * T * D * G)
  void Evaluate();

  // accumulates the per-image, per-category evaluation
  // results in "eval_imgs_" into the dictionary "eval" with fields:
  // params     - parameters used for evaluation
  // date       - date evaluation was performed
  // counts     - [T,R,K,A,M] parameter dimensions (see above)
  // precision  - [TxRxKxAxM] precision for every evaluation setting
  // recall     - [TxKxAxM] max recall for every evaluation setting
  // Complexity: O(T * R * K * A * M)
  void Accumulate();

  void Summarize();

  std::vector<float> GetStats() { return stats_; }

 protected:
  // Calculates average metric and per-class metrics given a specific setting.
  // ap        - Use true for average precision, false for average recall.
  // iou_thr   - IOU thresholds, can be -1.0, 0.5, 0.75.
  // area_rng  - Area range thresholds, can be all, small, medium, large.
  // max_dets  - Max number of detections per image, can be 1, 10, 100.
  std::pair<float, std::vector<float>> SummarizeInternal(
      bool ap = true, double iou_thr = -1.0, std::string area_rng = "all",
      int64_t max_dets = 100);

  // evaluation parameters.
  Params params_;
  // gt for evaluation.
  std::unordered_map<int64_t, std::vector<Annotation *>> gts_;
  // dt for evaluation.
  std::unordered_map<int64_t, std::vector<Annotation *>> dts_;
  // per-image per-category evaluation results [KxAxI] elements.
  std::vector<EvalImgs> eval_imgs_;

  // Used to compute unique id for each image id and category id pair.
  int64_t max_category_id_;
  // Key used in various maps.
  inline size_t Key(int64_t image_id, int64_t cat_id) const {
    return image_id * max_category_id_ + cat_id;
  }

 private:
  // Prepare gts_ and dts_ for evaluation based on params
  // :return: None
  // Only consider "bbox" type.
  virtual void Prepare() = 0;

  void SummarizeDets();

  // accumulated evaluation results.
  Eval eval_;
  // parameters for evaluation.
  Params params_eval_;
  // result summarization.
  std::vector<float> stats_;
  // category result
  std::vector<std::vector<float>> category_stats_;
  // ious between all gts and dts.
  std::unordered_map<int64_t, IoUMatrix> ious_;
};
}  // namespace pycocotools

#endif  // PYCOCOTOOLS_COCOEVAL_BASE_H_
