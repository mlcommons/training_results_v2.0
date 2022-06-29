#ifndef THIRD_PARTY_PY_PYCOCOTOOLS_COCO_H_
#define THIRD_PARTY_PY_PYCOCOTOOLS_COCO_H_

#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <jsoncpp/json/json.h>
#include "/usr/local/lib/python3.7/dist-packages/numpy/core/include/numpy/arrayobject.h"
#include "pycocotools/annotation.h"
extern "C" {
#include "maskApi.h"
}

#include <algorithm>
#include <vector>

#include "pycocotools/cocoeval_base.h"
namespace pycocotools {

class COCO {
 public:
  COCO(std::string annotation_file = "", bool use_mask = false);

  void CreateIndex();
  void CreateIndexWithAnns();
  // Get ann ids that satisfy given filter conditions. default skips that filter
  // param: img_ids  (int array)     : get anns for given imgs
  // param: cat_ids  (int array)     : get anns for given cats
  // param: area_area (float array)   : get anns for given area range (e.g. [0
  // inf]) param: is_crowd (boolean)       : get anns for given crowd label
  // (False or True) return: ids (int array)       : integer array of ann ids
  std::vector<int64_t> GetAnnIds(const std::vector<int64_t>& img_ids,
                                 const std::vector<int64_t>& cat_ids,
                                 const std::vector<float>& area_rng,
                                 const bool is_crowd,
                                 const bool respect_is_crowd = false) const;

  // filtering parameters. default skips that filter.
  // param: cat_names (str array)  : get cats for given cat names
  // param: sup_names (str array)  : get cats for given supercategory names
  // param: cat_ids (int array)  : get cats for given cat ids
  // return: ids (int array)   : integer array of cat ids
  std::vector<int64_t> GetCatIds(const std::vector<std::string>& cat_names,
                                 const std::vector<std::string>& sup_names,
                                 const std::vector<int64_t>& cat_ids) const;

  // Get img ids that satisfy given filter conditions.
  // param: img_ids (int array) : get imgs for given ids
  // param: cat_ids (int array) : get imgs with all given cats
  // return: ids (int array)  : integer array of img ids
  std::vector<int64_t> GetImgIds(const std::vector<int64_t>& img_ids,
                                 const std::vector<int64_t>& cat_ids) const;

  // Load anns with the specified ids.
  // param: ids (int array)       : integer ids specifying anns
  // return: anns (object array) : loaded ann objects
  std::vector<Annotation*> LoadAnns(std::vector<int64_t>& ids);

  // Load cats with the specified ids.
  // param: ids (int array)       : integer ids specifying cats
  // return: cats (object array) : loaded cat objects
  std::vector<Json::Value> LoadCats(const std::vector<int64_t>& ids) const;

  // Load imgs with the specified ids.
  // param: ids (int array)       : integer ids specifying img
  // return: imgs (object array) : loaded img objects
  std::vector<Json::Value> LoadImgs(const std::vector<int64_t>& ids) const;

  // Load result numpy array and return a result api object.
  // param:   py_array    :  numpy array of the result
  // return: res (obj)         : result api object
  // Only supports 'bbox' mode.
  COCO LoadRes(PyObject* py_object);
  COCO LoadResJson(std::string annotation_file);
  COCO LoadResMask(PyObject* py_object, PyObject* py_mask_object);

  // Convert result data from a numpy array [Nx7] where each row contains
  // {imageID,x1,y1,w,h,score,class}
  // param:  py_array (numpy.ndarray)
  // return: annotations (Json Value array)
  std::vector<Annotation> LoadNumpyAnnotations(
      PyArrayObject* py_array, PyObject* py_mask_array = nullptr);

  // Set value in dataset_;
  void SetMember(std::string name, Json::Value& value);

  // Set Ann vec_;
  void SetAnn(std::vector<Annotation>& anns) { anns_vec_ = anns; }

 private:
  void Convert(const Json::Value& json_item, Annotation* result);
  Json::Value dataset_;
  std::map<int64_t, int64_t> anns_;
  std::vector<Annotation> anns_vec_;
  std::map<int64_t, Json::Value> cats_;
  std::map<int64_t, Json::Value> imgs_;
  std::map<int64_t, std::vector<int64_t>> img_to_anns_;
  std::map<int64_t, std::vector<int64_t>> cat_to_imgs_;
  bool use_mask_;
};
}

#endif  // THIRD_PARTY_PY_PYCOCOTOOLS_COCOEVAL_H_

#ifndef THIRD_PARTY_PY_PYCOCOTOOLS_COCOEVAL_H_
#define THIRD_PARTY_PY_PYCOCOTOOLS_COCOEVAL_H_

#include <algorithm>
#include <vector>

#include "pycocotools/coco.h"
#include "pycocotools/cocoeval_base.h"

namespace pycocotools {

class COCOeval : public COCOevalBase {
 public:
  // For test only.
  COCOeval() : COCOevalBase("bbox") {}

  COCOeval(COCO &coco_gt, COCO &coco_dt, std::string iou_type)
      : COCOevalBase(iou_type), coco_gt_(coco_gt), coco_dt_(coco_dt) {
    std::vector<int64_t> img_ids, cat_ids;
    std::vector<std::string> cat_names, sup_names;
    params_.img_ids = coco_gt_.GetImgIds(img_ids, cat_ids);
    params_.cat_ids = coco_gt_.GetCatIds(cat_names, sup_names, cat_ids);
    std::stable_sort(params_.cat_ids.begin(), params_.cat_ids.end());
    max_category_id_ = params_.cat_ids.back();
  }

 private:
  // Prepare gts_ and dts_ for evaluation based on params
  // Only consider "bbox" type.
  void Prepare() override;

  // ground truth COCO API.
  COCO coco_gt_;
  // detections COCO API
  COCO coco_dt_;
};
}  // namespace pycocotools

#endif  // THIRD_PARTY_PY_PYCOCOTOOLS_COCOEVAL_H_
