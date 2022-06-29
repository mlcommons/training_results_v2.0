// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "pycocotools/coco.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <set>
#include <string>
#include <unordered_set>

namespace pycocotools {

namespace {

int init_numpy() {
  import_array();  // PyError if not successful
  return 0;
}
// it is only a trick to ensure import_array() is called, when *.so is loaded
// just called only once
const static int numpy_initialized = init_numpy();

constexpr char kAnns[] = "annotations";
constexpr char kImages[] = "images";
constexpr char kImageId[] = "image_id";
constexpr char kCategories[] = "categories";
constexpr char kCategoryId[] = "category_id";
constexpr char kId[] = "id";
constexpr char kArea[] = "area";
constexpr char kIsCrowd[] = "iscrowd";
constexpr char kName[] = "name";
constexpr char kSupCategory[] = "supercategory";
constexpr char kBbox[] = "bbox";
constexpr char kScore[] = "score";
constexpr char kSegm[] = "segmentation";
constexpr char kCounts[] = "counts";
constexpr char kSize[] = "size";

}  // namespace

COCO::COCO(std::string annotation_file, bool use_mask) : use_mask_(use_mask) {
  if (!annotation_file.empty()) {
    std::cout << "loading annotations into memory..." << std::endl;

    std::ifstream ifs(annotation_file);
    std::string json_string((std::istreambuf_iterator<char>(ifs)),
                            (std::istreambuf_iterator<char>()));
    Json::Value dataset;
    Json::Reader reader;
    if (!reader.parse(json_string, dataset)) {
      std::cout << "Failed to parse json from json_string." << std::endl;
    }
    dataset_ = dataset;

    CreateIndex();
  }
}

void COCO::Convert(const Json::Value& json_item, Annotation* result) {
  result->image_id = json_item[kImageId].asInt64();
  result->category_id = json_item[kCategoryId].asInt64();

  if (json_item.isMember(kId)) {
    result->id = json_item[kId].asInt64();
  }

  if (json_item.isMember(kIsCrowd)) {
    result->iscrowd = json_item[kIsCrowd].asBool();
  }

  for (const auto& bb : json_item[kBbox]) {
    result->bbox.push_back(bb.asFloat());
  }

  if (json_item.isMember(kArea)) {
    result->area = json_item[kArea].asFloat();
  }

  if (json_item.isMember(kScore)) {
    result->score = json_item[kScore].asFloat();
  }

  if (use_mask_) {
    Json::Value seg_item = json_item[kSegm];
    RLE R;
    std::string rle_str;
    uint32_t h, w;
    // Seg in Poly Format
    if (seg_item.isArray()) {
      h = imgs_[result->image_id]["height"].asUInt();
      w = imgs_[result->image_id]["width"].asUInt();
      std::vector<RLE> RLEs;

      for (const auto& segs : seg_item) {
        RLE Rtmp;
        std::vector<double> xy;
        for (const auto& poly_val : segs) {
          xy.push_back(poly_val.asDouble());
        }
        uint32_t k = xy.size() / 2;

        rleFrPoly(&Rtmp, &xy[0], k, h, w);
        RLEs.push_back(Rtmp);
      }
      if (RLEs.size() == 1) {
        R = RLEs[0];
      } else {
        rleMerge(&RLEs[0], &R, RLEs.size(), 0);
        // Free memory from temp RLEs
        for (auto i : RLEs) {
          rleFree(&i);
        }
      }
      result->cnts.assign(R.cnts, R.cnts + R.m);
      rleFree(&R);
    } else if (seg_item.isMember(kCounts) && seg_item.isMember(kSize)) {
      h = seg_item[kSize][0].asUInt();
      w = seg_item[kSize][1].asUInt();
      if (seg_item[kCounts].isString()) {
        // Seg in compressed RLE format
        rle_str = seg_item[kCounts].asString();
        rleFrString(&R, rle_str.c_str(), h, w);
        result->cnts.assign(R.cnts, R.cnts + R.m);
        rleFree(&R);
      } else {
        // Seg in uncompressed RLE format
        for (const auto& cnts_val : seg_item[kCounts]) {
          result->cnts.push_back(cnts_val.asUInt());
        }
      }
    } else {
      std::cout << "Type of seg format not supported." << std::endl;
      return;
    }
    result->h = h;
    result->w = w;
  }
}

void COCO::CreateIndex() {
  std::cout << "Create index..." << std::endl;
  anns_.clear();
  anns_vec_.clear();
  cats_.clear();
  imgs_.clear();
  img_to_anns_.clear();
  cat_to_imgs_.clear();

  // Parse image info first , as it gets used when parsing Annotations
  if (dataset_.isMember(kImages)) {
    Json::Value images = dataset_[kImages];
    for (auto it = images.begin(); it != images.end(); it++) {
      Json::Value image = *it;
      int64_t id = image[kId].asInt64();
      imgs_[id] = image;
    }
  }

  if (dataset_.isMember(kAnns)) {
    Json::Value anns = dataset_[kAnns];
    int64_t ann_index = 0;

    for (auto it = anns.begin(); it != anns.end(); it++) {
      anns_vec_.emplace_back();
      Annotation& ann = anns_vec_.back();
      Convert(*it, &ann);
      img_to_anns_[ann.image_id].push_back(ann_index);
      anns_[ann.id] = ann_index;
      ann_index++;
    }
  }

  if (dataset_.isMember(kCategories)) {
    Json::Value categories = dataset_[kCategories];
    for (auto it = categories.begin(); it != categories.end(); it++) {
      Json::Value cat = *it;
      int64_t id = cat[kId].asInt64();
      cats_[id] = cat;
    }
  }

  if (dataset_.isMember(kAnns) && dataset_.isMember(kCategories)) {
    Json::Value anns = dataset_[kAnns];
    for (auto it = anns.begin(); it != anns.end(); it++) {
      Json::Value ann = *it;
      int64_t category_id = ann[kCategoryId].asInt64();
      int64_t image_id = ann[kImageId].asInt64();
      cat_to_imgs_[category_id].push_back(image_id);
    }
  }

  std::cout << "index created!" << std::endl;
}

void COCO::CreateIndexWithAnns() {
  std::cout << "Create index..." << std::endl;
  anns_.clear();
  cats_.clear();
  imgs_.clear();
  img_to_anns_.clear();
  cat_to_imgs_.clear();

  if (!anns_vec_.empty()) {
    int64_t ann_index = 0;

    for (auto it = anns_vec_.begin(); it != anns_vec_.end(); it++) {
      img_to_anns_[it->image_id].push_back(ann_index);
      anns_[it->id] = ann_index;
      ann_index++;
    }
  }

  if (dataset_.isMember(kImages)) {
    Json::Value images = dataset_[kImages];
    for (auto it = images.begin(); it != images.end(); it++) {
      Json::Value image = *it;
      int64_t id = image[kId].asInt64();
      imgs_[id] = image;
    }
  }

  if (dataset_.isMember(kCategories)) {
    Json::Value categories = dataset_[kCategories];
    for (auto it = categories.begin(); it != categories.end(); it++) {
      Json::Value cat = *it;
      int64_t id = cat[kId].asInt64();
      cats_[id] = cat;
    }
  }

  if (!anns_vec_.empty() && dataset_.isMember(kCategories)) {
    for (auto it = anns_vec_.begin(); it != anns_vec_.end(); it++) {
      int64_t category_id = it->category_id;
      int64_t image_id = it->image_id;
      cat_to_imgs_[category_id].push_back(image_id);
    }
  }

  std::cout << "index created!" << std::endl;
}

std::vector<int64_t> COCO::GetAnnIds(const std::vector<int64_t>& img_ids,
                                     const std::vector<int64_t>& cat_ids,
                                     const std::vector<float>& area_rng,
                                     const bool is_crowd,
                                     const bool respect_is_crowd) const {
  std::vector<int64_t> ids;
  // index into anns_vec_
  std::vector<int64_t> anns_idx;

  if (img_ids.empty() && cat_ids.empty() && area_rng.empty()) {
    anns_idx.resize(anns_vec_.size());
    std::iota(anns_idx.begin(), anns_idx.end(), 0);
  } else {
    std::vector<int64_t> tmp_anns_idx;
    if (!img_ids.empty()) {
      for (const int64_t& img_id : img_ids) {
        const auto& iter = img_to_anns_.find(img_id);
        if (iter != img_to_anns_.end()) {
          std::copy(iter->second.begin(), iter->second.end(),
                    std::back_inserter(tmp_anns_idx));
        }
      }
    } else {
      tmp_anns_idx.resize(anns_vec_.size());
      std::iota(tmp_anns_idx.begin(), tmp_anns_idx.end(), 0);
    }

    std::vector<int64_t> tmp_anns_idx_with_cat_ids;
    if (!cat_ids.empty()) {
      std::unordered_set<int> cat_id_set(cat_ids.begin(), cat_ids.end());
      for (auto id : tmp_anns_idx) {
        int64_t category_id = anns_vec_[id].category_id;
        if (cat_id_set.find(category_id) != cat_id_set.end()) {
          tmp_anns_idx_with_cat_ids.push_back(id);
        }
      }
    } else {
      tmp_anns_idx_with_cat_ids = tmp_anns_idx;
    }

    std::vector<int64_t> tmp_anns_idx_with_area_rng;
    if (area_rng.size() > 1) {
      for (auto id : tmp_anns_idx_with_cat_ids) {
        float area = anns_vec_[id].area;
        if (area > area_rng[0] && area < area_rng[1]) {
          tmp_anns_idx_with_area_rng.push_back(id);
        }
      }
    } else {
      tmp_anns_idx_with_area_rng = tmp_anns_idx_with_cat_ids;
    }

    anns_idx = tmp_anns_idx_with_area_rng;
  }

  if (respect_is_crowd) {
    for (auto idx : anns_idx) {
      if (anns_vec_[idx].iscrowd == is_crowd) {
        ids.push_back(anns_vec_[idx].id);
      }
    }
  } else {
    for (auto idx : anns_idx) {
      ids.push_back(anns_vec_[idx].id);
    }
  }

  std::stable_sort(ids.begin(), ids.end());
  return ids;
}

std::vector<int64_t> COCO::GetCatIds(
    const std::vector<std::string>& cat_names,
    const std::vector<std::string>& sup_names,
    const std::vector<int64_t>& cat_ids) const {
  Json::Value tmp_cats = dataset_[kCategories];
  Json::Value tmp_cats_with_cat_names(Json::arrayValue);
  if (!cat_names.empty()) {
    std::set<std::string> cat_names_set(cat_names.begin(), cat_names.end());
    for (auto it = tmp_cats.begin(); it != tmp_cats.end(); it++) {
      Json::Value cat = *it;
      std::string name = cat[kName].asString();
      if (cat_names_set.find(name) != cat_names_set.end()) {
        tmp_cats_with_cat_names.append(cat);
      }
    }
  } else {
    tmp_cats_with_cat_names = tmp_cats;
  }

  Json::Value tmp_cats_with_sup_names(Json::arrayValue);
  if (!cat_names.empty()) {
    std::set<std::string> sup_names_set(sup_names.begin(), sup_names.end());
    for (auto it = tmp_cats_with_cat_names.begin();
         it != tmp_cats_with_cat_names.end(); it++) {
      Json::Value cat = *it;
      std::string name = cat[kSupCategory].asString();
      if (sup_names_set.find(name) != sup_names_set.end()) {
        tmp_cats_with_sup_names.append(cat);
      }
    }
  } else {
    tmp_cats_with_sup_names = tmp_cats_with_cat_names;
  }

  Json::Value tmp_cats_with_cat_ids(Json::arrayValue);
  if (!cat_names.empty()) {
    std::set<int64_t> cat_ids_set(cat_ids.begin(), cat_ids.end());
    for (auto it = tmp_cats_with_sup_names.begin();
         it != tmp_cats_with_sup_names.end(); it++) {
      Json::Value cat = *it;
      int64_t cat_id = cat[kId].asInt64();
      if (cat_ids_set.find(cat_id) != cat_ids_set.end()) {
        tmp_cats_with_cat_ids.append(cat);
      }
    }
  } else {
    tmp_cats_with_cat_ids = tmp_cats_with_sup_names;
  }

  std::vector<int64_t> ids;
  for (auto it = tmp_cats_with_sup_names.begin();
       it != tmp_cats_with_sup_names.end(); it++) {
    Json::Value cat = *it;
    int64_t cat_id = cat[kId].asInt64();
    ids.push_back(cat_id);
  }

  std::sort(ids.begin(), ids.end());
  return ids;
}

std::vector<int64_t> COCO::GetImgIds(
    const std::vector<int64_t>& img_ids,
    const std::vector<int64_t>& cat_ids) const {
  std::set<int64_t> ids;

  if (img_ids.empty() && cat_ids.empty()) {
    for (const auto& it : imgs_) {
      ids.insert(it.first);
    }
  } else {
    for (int cat_id : cat_ids) {
      const auto& iter = cat_to_imgs_.find(cat_id);
      if (iter != cat_to_imgs_.end()) {
        std::copy(iter->second.begin(), iter->second.end(),
                  std::inserter(ids, ids.end()));
      }
    }
  }

  return std::vector<int64_t>(ids.begin(), ids.end());
}

std::vector<Annotation*> COCO::LoadAnns(std::vector<int64_t>& ids) {
  std::vector<Annotation*> anns;
  for (auto id : ids) {
    const auto& iter = anns_.find(id);
    if (iter != anns_.end()) {
      anns.push_back(&anns_vec_[iter->second]);
    }
  }
  return anns;
}

std::vector<Json::Value> COCO::LoadCats(const std::vector<int64_t>& ids) const {
  std::vector<Json::Value> cats;
  for (auto id : ids) {
    const auto& iter = cats_.find(id);
    if (iter != cats_.end()) {
      cats.push_back(iter->second);
    }
  }
  return cats;
}

std::vector<Json::Value> COCO::LoadImgs(const std::vector<int64_t>& ids) const {
  std::vector<Json::Value> imgs;
  for (auto id : ids) {
    const auto& iter = imgs_.find(id);
    if (iter != imgs_.end()) {
      imgs.push_back(iter->second);
    }
  }
  return imgs;
}

void COCO::SetMember(std::string name, Json::Value& value) {
  dataset_[name] = value;
}

COCO COCO::LoadRes(PyObject* py_object) {
  return LoadResMask(py_object, nullptr);
}

COCO COCO::LoadResMask(PyObject* py_object, PyObject* py_mask_object) {
  auto res = COCO("", use_mask_);
  if (!PyArray_Check(py_object)) {
    std::cout << "Destination PyObject is not a PyArrayObject." << std::endl;
    return res;
  }

  if (py_mask_object != nullptr && (!PyList_Check(py_mask_object))) {
    std::cout << "Destination PyObject is not a PyDictObject." << std::endl;
    return res;
  }

  PyArrayObject* py_array = reinterpret_cast<PyArrayObject*>(py_object);

  res.SetMember(kImages, dataset_[kImages]);

  auto anns = LoadNumpyAnnotations(py_array, py_mask_object);

  res.SetMember(kCategories, dataset_[kCategories]);

  int id = 0;
  for (auto it = anns.begin(); it != anns.end(); it++) {
    id++;
    float bb2 = it->bbox[2];
    float bb3 = it->bbox[3];

    it->area = bb2 * bb3;
    it->id = id;
    it->iscrowd = 0;
  }

  res.SetAnn(anns);
  res.CreateIndexWithAnns();

  return res;
}

COCO COCO::LoadResJson(std::string annotation_file) {
  auto res = COCO("", use_mask_);
  if (!annotation_file.empty()) {
    res.SetMember(kImages, dataset_[kImages]);

    std::ifstream ifs(annotation_file);
    std::string json_string((std::istreambuf_iterator<char>(ifs)),
                            (std::istreambuf_iterator<char>()));
    Json::Value dataset;
    Json::Reader reader;
    if (!reader.parse(json_string, dataset)) {
      std::cout << "Failed to parse json from json_string." << std::endl;
    }
    std::vector<Annotation> anns;
    for (auto it = dataset.begin(); it != dataset.end(); it++) {
      anns.emplace_back();
      Annotation& ann = anns.back();
      Convert(*it, &ann);
    }

    res.SetMember(kCategories, dataset_[kCategories]);

    int id = 0;
    for (auto it = anns.begin(); it != anns.end(); it++) {
      id++;
      float bb2 = it->bbox[2];
      float bb3 = it->bbox[3];

      it->area = bb2 * bb3;
      it->id = id;
      it->iscrowd = 0;
    }

    res.SetAnn(anns);
    res.CreateIndexWithAnns();
  }
  return res;
}

std::vector<Annotation> COCO::LoadNumpyAnnotations(PyArrayObject* py_array,
                                                   PyObject* py_mask_array) {
  std::vector<Annotation> anns;
  if (PyArray_SHAPE(py_array)[1] != 7) {
    std::cout << "Input array's second dimension must be equal to 7."
              << std::endl;
  }
  int N = PyArray_SHAPE(py_array)[0];

  for (int i = 0; i < N; i++) {
    Annotation ann;
    // Get image id.
    auto image_id_ptr =
        reinterpret_cast<const float*>(PyArray_GETPTR2(py_array, i, 0));
    int image_id = static_cast<int>(*image_id_ptr);
    ann.image_id = image_id;

    // Get bbox.
    for (int d = 1; d <= 4; d++) {
      auto bbox_ptr =
          reinterpret_cast<const float*>(PyArray_GETPTR2(py_array, i, d));
      float bbox_coor = static_cast<float>(*bbox_ptr);
      ann.bbox.push_back(bbox_coor);
    }

    // Get score.
    auto score_ptr =
        reinterpret_cast<const float*>(PyArray_GETPTR2(py_array, i, 5));
    float score = static_cast<float>(*score_ptr);
    ann.score = score;

    // Get category_id.
    auto category_id_ptr =
        reinterpret_cast<const float*>(PyArray_GETPTR2(py_array, i, 6));
    int category_id = static_cast<int>(*category_id_ptr);
    ann.category_id = category_id;

    // Mask Processing
    if (py_mask_array != nullptr) {
      PyObject* py_dic_segm = PyList_GetItem(py_mask_array, i);
      PyObject* py_counts = PyDict_GetItemString(py_dic_segm, "counts");
#if PY_MAJOR_VERSION >= 3
      std::string rle_str = std::string(PyUnicode_AsUTF8(py_counts));
#else
      std::string rle_str = std::string(PyBytes_AsString(py_counts));
#endif
      PyObject* py_size = PyDict_GetItemString(py_dic_segm, "size");
      ann.h = PyLong_AsLong(PyList_GetItem(py_size, 0));
      ann.w = PyLong_AsLong(PyList_GetItem(py_size, 1));

      RLE R;
      rleFrString(&R, rle_str.c_str(), ann.h, ann.w);
      ann.cnts.assign(R.cnts, R.cnts + R.m);
      rleFree(&R);
    }

    anns.push_back(ann);
  }

  return anns;
}

void COCOeval::Prepare() {
  const Params &p = params_;
  std::vector<float> dummy_area_rng;
  bool dummy_is_crowd = false;
  std::vector<int64_t> gt_ids, dt_ids;
  if (p.use_cats) {
    gt_ids = coco_gt_.GetAnnIds(p.img_ids, p.cat_ids, dummy_area_rng,
                                dummy_is_crowd);

    dt_ids = coco_dt_.GetAnnIds(p.img_ids, p.cat_ids, dummy_area_rng,
                                dummy_is_crowd);
  } else {
    std::vector<int64_t> cat_ids;
    gt_ids =
        coco_gt_.GetAnnIds(p.img_ids, cat_ids, dummy_area_rng, dummy_is_crowd);
    dt_ids =
        coco_dt_.GetAnnIds(p.img_ids, cat_ids, dummy_area_rng, dummy_is_crowd);
  }

  std::vector<Annotation *> gts = coco_gt_.LoadAnns(gt_ids);
  std::vector<Annotation *> dts = coco_dt_.LoadAnns(dt_ids);

  // set ignore flag
  for (auto gt : gts) {
    gt->ignore |= gt->iscrowd;
  }

  gts_.clear();
  dts_.clear();

  for (auto gt : gts) {
    gts_[Key(gt->image_id, gt->category_id)].push_back(gt);
  }

  for (auto dt : dts) {
    dts_[Key(dt->image_id, dt->category_id)].push_back(dt);
  }

  eval_imgs_.clear();
}

}  // namespace pycocotools
