#include "pycocotools/cocoeval_base.h"

#include <algorithm>
#include <chrono>  // NOLINT
#include <cmath>
#include <iostream>
#include <numeric>
#include <thread>



extern "C" {
#include "maskApi.h"
}

namespace pycocotools {

namespace {

constexpr char kBbox[] = "bbox";
constexpr char kSegm[] = "segm";

bool DtComp(const Annotation *lhs, const Annotation *rhs) {
  return lhs->score > rhs->score;
}

template <typename T>
std::vector<int64_t> arg_sort(std::vector<T> &v) {
  std::vector<int64_t> indices(v.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::stable_sort(indices.begin(), indices.end(),
                   [&v](int64_t i1, int64_t i2) { return v[i1] > v[i2]; });

  return indices;
}

template <typename T>
void append_to_col(const std::vector<T> &new_data,
                   std::vector<std::vector<T>> *data, int num_rows,
                   int num_cols, int max_det) {
  if (data->empty()) {
    data->resize(num_rows);
  } else if (data->size() != num_rows || num_rows == 0) {
    return;
  }

  auto col = std::min(num_cols, max_det);

  for (size_t r = 0; r < num_rows; r++) {
    std::copy(new_data.begin() + r * num_cols,
              new_data.begin() + r * num_cols + col,
              std::back_inserter((*data)[r]));
  }
}

}  // namespace

std::vector<double> linspace(double start, double stop, size_t N) {
  std::vector<double> result(N);
  for (int i = 0; i < N; i++) {
    result[i] =
        static_cast<double>(i) * (stop - start) / static_cast<double>(N - 1) +
        start;
  }
  return result;
}

Params::Params(std::string input_iou_type) {
  iou_type = input_iou_type;
  if (iou_type == "bbox" || iou_type == "segm") {
    SetDetParams();
  } else {
    std::cout << "Other iou type not supported. " << std::endl;
  }
}

void Params::SetDetParams() {
  img_ids.clear();
  cat_ids.clear();
  iou_thrs = linspace(0.5, 0.95, round((0.95 - 0.5) / 0.05) + 1);
  rec_thrs = linspace(0.0, 1.00, round((1.00 - 0.0) / 0.01) + 1);
  max_dets = std::vector<int64_t>{1, 10, 100};
  std::vector<std::vector<float>> area_rng_tmp = {
      {static_cast<float>(pow(0.0, 2)), static_cast<float>(pow(1.0e5, 2))},
      {static_cast<float>(pow(0.0, 2)), static_cast<float>(pow(32.0, 2))},
      {static_cast<float>(pow(32.0, 2)), static_cast<float>(pow(96.0, 2))},
      {static_cast<float>(pow(96.0, 2)), static_cast<float>(pow(1.0e5, 2))}};
  area_rng = area_rng_tmp;
  std::vector<std::string> area_rng_lbl_tmp = {"all", "small", "medium",
                                               "large"};
  area_rng_lbl = area_rng_lbl_tmp;
  use_cats = true;
}

IoUMatrix COCOevalBase::ComputeIoU(int64_t img_id, int64_t cat_id) {
  std::vector<std::vector<float>> result;
  Params &p = params_;
  std::vector<Annotation *> *pgt, *pdt;
  std::vector<Annotation *> rgt, rdt;
  if (p.use_cats) {
    pgt = &gts_.find(Key(img_id, cat_id))->second;
    pdt = &dts_.find(Key(img_id, cat_id))->second;
  } else {
    pgt = &rgt;
    pdt = &rdt;
    for (auto cid : p.cat_ids) {
      auto gt_cid = gts_[Key(img_id, cid)];
      for (auto g : gt_cid) {
        rgt.push_back(g);
      }

      auto dt_cid = dts_[Key(img_id, cid)];
      for (auto d : dt_cid) {
        rdt.push_back(d);
      }
    }
  }

  std::vector<Annotation *> &gt = *pgt;
  std::vector<Annotation *> &dt = *pdt;

  if (gt.empty() || dt.empty()) {
    return result;
  }

  std::stable_sort(dt.begin(), dt.end(), DtComp);

  if (dt.size() > p.max_dets.back()) {
    dt.resize(p.max_dets.back());
  }

  uint32_t m = dt.size();
  uint32_t n = gt.size();
  IoUMatrix result_iou(m, std::vector<float>(n, 0.0));
  std::vector<double> o(m * n);
  std::vector<byte> iscrowd(n);
  if (p.iou_type == kBbox) {
    std::vector<double> g, d;

    for (int i = 0; i < n; i++) {
      g.insert(g.end(), gt[i]->bbox.begin(), gt[i]->bbox.end());
      iscrowd[i] = (byte)gt[i]->iscrowd;
    }

    for (int i = 0; i < m; i++) {
      d.insert(d.end(), dt[i]->bbox.begin(), dt[i]->bbox.end());
    }
    bbIou(&d[0], &g[0], m, n, &iscrowd[0], &o[0]);
  } else if (p.iou_type == kSegm) {
    RLE *g = new RLE[n];
    RLE *d = new RLE[m];
    for (int i = 0; i < n; i++) {
      g[i].h = gt[i]->h;
      g[i].w = gt[i]->w;
      g[i].cnts = &(gt[i]->cnts[0]);
      g[i].m = (gt[i]->cnts.size());
      iscrowd[i] = (byte)gt[i]->iscrowd;
    }

    for (int i = 0; i < m; i++) {
      d[i].h = dt[i]->h;
      d[i].w = dt[i]->w;
      d[i].cnts = &(dt[i]->cnts[0]);
      d[i].m = (dt[i]->cnts.size());
    }
    rleIou(d, g, m, n, &iscrowd[0], &o[0]);
    delete[] g;
    delete[] d;
  } else {
    std::cout << "iou_type is not supported." << std::endl;
  }

  // C-code returns in coloumn major format, so need to transpose
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      result_iou[j][i] = o[i * m + j];
    }
  }

  return result_iou;
}

void COCOevalBase::EvaluateImg(int64_t img_id, int64_t cat_id,
                               const std::vector<std::vector<float>> &area_rng_vec,
                               int64_t max_det, 
			       EvalImgs results[]) {
  const Params &p = params_;
  const std::vector<Annotation *> *pgt_orig, *pdt_orig;
  const std::vector<Annotation *> rgt_orig, rdt_orig;
  if (p.use_cats) {
    auto lookup_key = Key(img_id, cat_id);
    pgt_orig = &gts_.find(lookup_key)->second;
    pdt_orig = &dts_.find(lookup_key)->second;
  } else {
    pgt_orig = &rgt_orig;
    pdt_orig = &rdt_orig;
    for (auto cid : p.cat_ids) {
      auto gt_cid = gts_.find(Key(img_id, cid))->second;
      for (auto g : gt_cid) {
	   const_cast<std::vector<Annotation *> &>(rgt_orig).push_back(g);
      }

      auto dt_cid = dts_.find(Key(img_id, cid))->second;
      for (auto d : dt_cid) {
	     const_cast<std::vector<Annotation *> &>(rdt_orig).push_back(d);
      }
    }
  }
  
  int h = 0;
  for (const auto &area_rng : area_rng_vec) {
    auto gt = *pgt_orig;
    auto dt = *pdt_orig;
    if (gt.empty() && dt.empty()) {
      continue;
    }
    EvalImgs &result = results[h++];

    for (int i = 0; i < gt.size(); i++) {
      gt[i]->uignore = gt[i]->ignore || gt[i]->area < area_rng[0] ||
                       gt[i]->area > area_rng[1];
    }

    // sort dt highest score first, sort gt ignore last.
    std::vector<int> gind(gt.size());
    std::vector<Annotation *> gtmp(gt.size());

    int gt_index = 0;
    std::vector<bool> gt_ig(gt.size());
    for (int i = 0; i < gt.size(); i++) {
      if (!gt[i]->uignore) {
        gind[gt_index] = i;
        gtmp[gt_index] = gt[i];
        gt_ig[gt_index] = false;
        gt_index++;
      }
    }

    for (int i = 0; i < gt.size(); i++) {
      if (gt[i]->uignore) {
        gind[gt_index] = i;
        gtmp[gt_index] = gt[i];
        gt_ig[gt_index] = true;
        gt_index++;
      }
    }

    // std::stable_sort(gt.begin(), gt.end(), GtComp);
    gt = gtmp;

    int64_t G = gt.size();
    int64_t D = dt.size();
    std::vector<bool> iscrowd(gt.size());
    iscrowd.reserve(gt.size());
    for (int i = 0; i < G; i++) {
      iscrowd[i] = gt[i]->iscrowd;
    }

    // load computed ious.
    IoUMatrix ious(D, std::vector<float>(G, 0.0));

    auto it = ious_.find(Key(img_id, cat_id));
    if (it == ious_.end()) {
      // pass
    } else if (!it->second.empty()) {
      IoUMatrix ious_tmp = it->second;
      for (int d_i = 0; d_i < ious_tmp.size(); d_i++) {
        for (int g_i = 0; g_i < ious_tmp[d_i].size(); g_i++) {
          ious[d_i][g_i] = ious_tmp[d_i][gind[g_i]];
        }
      }
    } else {
      ious = it->second;
    }

    int64_t T = p.iou_thrs.size();

    std::vector<int64_t> gtm(T * G, 0);
    std::vector<int64_t> dtm(T * D, 0);
    std::vector<bool> dt_ig(T * D, false);

    if (!ious.empty()) {
      for (int tind = 0; tind < T; tind++) {
        auto t = p.iou_thrs[tind];
        for (int dind = 0; dind < D; dind++) {
          const auto &d = dt[dind];
          // information about best match so far (m=-1 -> unmatched)
          auto iou = std::min(t, 1 - 1.0e-10);
          int64_t m = -1;
          for (int gind = 0; gind < G; gind++) {
            // if this gt already matched, and not a crowd, continue
            if (gtm[tind * G + gind] > 0 && !iscrowd[gind]) {
              continue;
            }
            // if dt matched to reg gt, and on ignore gt, stop
            if (m > -1 && !gt_ig[m] && gt_ig[gind]) {
              break;
            }
            // continue to next gt unless better match made
            if (ious[dind][gind] < iou) {
              continue;
            }
            // if match successful and best so far, store appropriately
            iou = ious[dind][gind];
            m = gind;
          }
          // if match made store id of match for both dt and gt
          if (m == -1) {
            continue;
          }
          dt_ig[tind * D + dind] = gt_ig[m];
          dtm[tind * D + dind] = gt[m]->id;
          gtm[tind * G + m] = d->id;
        }
      }
    }

    // set unmatched detections outside of area range to ignore.
    std::vector<bool> ignore_area(D, false);
    for (int i = 0; i < D; i++) {
      float area = dt[i]->area;
      if (area < area_rng[0] || area > area_rng[1]) {
        ignore_area[i] = true;
      }
    }

    for (int tind = 0; tind < T; tind++) {
      for (int dind = 0; dind < D; dind++) {
        dt_ig[tind * D + dind] =
            (dt_ig[tind * D + dind] ||
             (dtm[tind * D + dind] == 0 && ignore_area[dind]));
      }
    }

    result.image_id = img_id;
    result.category_id = cat_id;
    result.max_det = max_det;
    result.dt_matches = dtm;
    for (const auto &d : dt) {
      result.dt_scores.push_back(d->score);
    }
    result.gt_ignore = gt_ig;
    result.dt_ignore = dt_ig;
  }
}

void COCOevalBase::Evaluate() {
  std::cout << "Running per image evaluation..." << std::endl;
  auto tic = std::chrono::system_clock::now();
  Params &p = params_;

  std::cout << "Evaluate annotation type: " << p.iou_type << std::endl;
  auto last = std::unique(p.img_ids.begin(), p.img_ids.end());
  p.img_ids.resize(std::distance(p.img_ids.begin(), last));
  std::cout << "Number of images: " << p.img_ids.size() << std::endl;

  if (p.use_cats) {
    auto last = std::unique(p.cat_ids.begin(), p.cat_ids.end());
    p.cat_ids.resize(std::distance(p.cat_ids.begin(), last));
  }
  std::cout << "Number of categories: " << p.cat_ids.size() << std::endl;

  std::sort(p.max_dets.begin(), p.max_dets.end());

  Prepare();

  // loop through images, area range, max detection number
  std::vector<int64_t> cat_ids;
  if (p.use_cats) {
    cat_ids = p.cat_ids;
  } else {
    cat_ids.push_back(-1);
  }

  
  ious_.reserve(p.img_ids.size() * cat_ids.size());
  for (auto img_id : p.img_ids) {
      for (auto cat_id : cat_ids) {
        ious_.try_emplace(Key(img_id, cat_id));
        gts_.try_emplace(Key(img_id, cat_id));
        dts_.try_emplace(Key(img_id, cat_id));
      }
  }
  std::vector<std::thread> iou_bundle;
  for (auto cat_id : cat_ids) {
     iou_bundle.emplace_back([&, cat_id] {
        for (auto img_id : p.img_ids) {
          ious_.find(Key(img_id, cat_id))->second = ComputeIoU(img_id, cat_id);
       }
     });
  }
  for (auto& thread : iou_bundle) { thread.join(); }


  int64_t max_det = p.max_dets.back();

  const size_t img_ids_size = p.img_ids.size();
  const size_t area_rng_size = p.area_rng.size();
  eval_imgs_.resize(cat_ids.size() * img_ids_size * area_rng_size);
  std::vector<std::thread> img_bundle;
  size_t i = 0;
  for (auto cat_id : cat_ids) {
     img_bundle.emplace_back([&, cat_id, i] {
     size_t j = 0;
     for (auto img_id : p.img_ids) {
          EvaluateImg(img_id, cat_id, p.area_rng, max_det,
                      &eval_imgs_[i * img_ids_size * area_rng_size +
                                  j * area_rng_size]);
          j++;
        }
     });
     i++;
  }
  for (auto& thread : img_bundle) { thread.join(); }

  auto toc = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = toc - tic;
  std::cout << "Evaluation Done (t= " << elapsed_seconds.count() << ")s."
            << std::endl;
}

void COCOevalBase::Accumulate() {
  auto tic = std::chrono::system_clock::now();
  Params &p = params_;
  std::cout << "Accumulating evaluation results..." << std::endl;
  if (!p.use_cats) {
    p.cat_ids = std::vector<int64_t>{-1};
  }
  int64_t T = p.iou_thrs.size();
  int64_t R = p.rec_thrs.size();
  int64_t K = p.cat_ids.size();
  if (!p.use_cats) {
    K = 1;
  }

  int64_t A = p.area_rng.size();
  int64_t M = p.max_dets.size();

  eval_.precision = std::vector<float>(T * R * K * A * M, -1.0);
  eval_.recall = std::vector<float>(T * K * A * M, -1.0);
  std::vector<int64_t> cat_ids;

  if (p.use_cats) {
    cat_ids = p.cat_ids;
  } else {
    cat_ids.push_back(-1);
  }

  std::vector<int64_t> m_list;

  for (int n = 0; n < p.max_dets.size(); n++) {
    auto m = p.max_dets[n];
    m_list.push_back(m);
  }

  int64_t I0 = p.img_ids.size();
  int64_t A0 = p.area_rng.size();


  std::vector<std::thread> accumulate_bundle;
  // TODO(wangtao): make variable names more readable to understand.
  for (int64_t k = 0; k < K; k++) {
    accumulate_bundle.emplace_back([&, k] {
      auto Nk = k * A0 * I0;
      for (int64_t a = 0; a < A; a++) {
        auto Na = a;
        for (int64_t m = 0; m < M; m++) {
          size_t max_det = m_list[m];
          std::vector<const EvalImgs *> E;
          int64_t eval_imgs_size = eval_imgs_.size();
          for (int ii = 0; ii < I0; ii++) {
            auto index = Nk + Na + ii * A0;
            if (index < eval_imgs_size) {
              auto &e = eval_imgs_[index];
              if (!e.dt_scores.empty() || !e.gt_ignore.empty()) {
                E.push_back(&eval_imgs_[index]);
              }
            }
          }
          if (E.empty()) {
            continue;
          }

          std::vector<double> dt_scores;
          std::vector<std::vector<int64_t>> dtm(T);
          std::vector<std::vector<bool>> dt_ig(T);
          std::vector<bool> gt_ig;

          for (const EvalImgs *e : E) {
            if (!e->dt_scores.empty()) {
              dt_scores.insert(
                dt_scores.end(), e->dt_scores.begin(),
                e->dt_scores.begin() + std::min(max_det, e->dt_scores.size()));
            append_to_col<int64_t>(e->dt_matches, &dtm, T, e->dt_scores.size(),
                                   max_det);
            append_to_col<bool>(e->dt_ignore, &dt_ig, T, e->dt_scores.size(),
                                max_det);
          }
          gt_ig.insert(gt_ig.end(), e->gt_ignore.begin(), e->gt_ignore.end());
        }

        auto inds = arg_sort(dt_scores);

        int64_t npig = std::count_if(gt_ig.begin(), gt_ig.end(),
                                     [](bool value) { return !value; });
        if (npig == 0) {
          continue;
        }

        std::vector<std::vector<bool>> tps(
            dtm.size(), std::vector<bool>(inds.size(), false));
        std::vector<std::vector<bool>> fps(
            dtm.size(), std::vector<bool>(inds.size(), false));

        for (int i = 0; i < dtm.size(); i++) {
          for (int j = 0; j < inds.size(); j++) {
            int64_t sorted_j = inds[j];
            tps[i][j] = dtm[i][sorted_j] && (!dt_ig[i][sorted_j]);
            fps[i][j] = (!dtm[i][sorted_j]) && (!dt_ig[i][sorted_j]);
          }
        }

        std::vector<std::vector<double>> tp_sum(
            dtm.size(), std::vector<double>(inds.size(), 0.0f));
        std::vector<std::vector<double>> fp_sum(
            dtm.size(), std::vector<double>(inds.size(), 0.0f));

        for (int i = 0; i < dtm.size(); i++) {
          int tp_sum_tmp = 0;
          int fp_sum_tmp = 0;
          for (int j = 0; j < inds.size(); j++) {
            tp_sum_tmp += tps[i][j];
            fp_sum_tmp += fps[i][j];
            tp_sum[i][j] = static_cast<double>(tp_sum_tmp);
            fp_sum[i][j] = static_cast<double>(fp_sum_tmp);
          }
        }

        for (int64_t t = 0; t < tp_sum.size(); t++) {
          auto nd = tp_sum[t].size();
          std::vector<double> rc(nd, 0.0);
          std::vector<double> pr(nd, 0.0);
          double eps = 2.22044604925e-16;
          for (int i = 0; i < nd; i++) {
            rc[i] = tp_sum[t][i] / static_cast<double>(npig);
            pr[i] = tp_sum[t][i] / (fp_sum[t][i] + tp_sum[t][i] + eps);
          }

	  eval_.recall[t * K * A * M + k * A * M + a * M + m] =
		   nd ? rc.back() : 0;

          std::vector<double> q(R, 0);

          for (int i = nd - 1; i > 0; --i) {
            if (pr[i] > pr[i - 1]) {
              pr[i - 1] = pr[i];
            }
          }

          std::vector<int> inds_thrs(p.rec_thrs.size());
          for (size_t i = 0; i < p.rec_thrs.size(); i++) {
            auto it = std::lower_bound(rc.begin(), rc.end(), p.rec_thrs[i]);
            inds_thrs[i] = it - rc.begin();
          }

          for (int ri = 0; ri < inds_thrs.size(); ri++) {
            auto pi = inds_thrs[ri];
            if (pi >= pr.size()) {
              continue;
            }
            q[ri] = pr[pi];
          }

          for (size_t i = 0; i < inds_thrs.size(); i++) {
            // precision[t,:,k,a,m] = np.array(q)
            size_t index =
                t * R * K * A * M + i * K * A * M + k * A * M + a * M + m;
            eval_.precision[index] = q[i];
          }
        }
      }
    }
  });
  }
  for (auto& thread : accumulate_bundle) { thread.join(); }

  eval_.params = p;
  eval_.counts = std::vector<int64_t>{T, R, K, A, M};

  auto toc = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = toc - tic;
  std::cout << "Accumulate Done (t= " << elapsed_seconds.count() << ")s."
            << std::endl;
}

std::pair<float, std::vector<float>> COCOevalBase::SummarizeInternal(
    bool ap, double iou_thr, std::string area_rng, int64_t max_dets) {
  Params &p = params_;
  int64_t aind;
  for (int64_t i = 0; i < p.area_rng_lbl.size(); i++) {
    if (p.area_rng_lbl[i] == area_rng) {
      aind = i;
    }
  }

  int64_t mind;
  for (int64_t i = 0; i < p.max_dets.size(); i++) {
    if (p.max_dets[i] == max_dets) {
      mind = i;
    }
  }

  int64_t T = eval_.counts[0];
  int64_t R = eval_.counts[1];
  int64_t K = eval_.counts[2];
  int64_t A = eval_.counts[3];
  int64_t M = eval_.counts[4];

  int thr_index = -1;
  if (iou_thr > 0.0) {
    for (int i = 0; i < p.iou_thrs.size(); i++) {
      if (std::fabs(iou_thr - p.iou_thrs[i]) <
          std::numeric_limits<float>::epsilon()) {
        thr_index = i;
        break;
      }
    }
  }

  std::vector<float> s;
  if (ap) {
    if (thr_index != -1) {
      s.resize(R * K);
      for (int64_t r = 0; r < R; r++) {
        for (int64_t k = 0; k < K; k++) {
          auto s_index = r * K + k;
          auto index = thr_index * R * K * A * M + r * K * A * M + k * A * M +
                       aind * M + mind;
          s[s_index] = eval_.precision[index];
        }
      }
    } else {
      s.resize(T * R * K);
      for (int64_t t = 0; t < T; t++) {
        for (int64_t r = 0; r < R; r++) {
          for (int64_t k = 0; k < K; k++) {
            auto s_index = t * R * K + r * K + k;
            auto index =
                t * R * K * A * M + r * K * A * M + k * A * M + aind * M + mind;
            s[s_index] = eval_.precision[index];
          }
        }
      }
    }

  } else {
    if (thr_index != -1) {
      s.resize(K);
    } else {
      s.resize(T * K);
    }
    for (int64_t k = 0; k < K; k++) {
      if (thr_index != -1) {
        auto s_index = k;
        auto index = thr_index * K * A * M + k * A * M + aind * M + mind;
        s[s_index] = eval_.recall[index];
      } else {
        for (int64_t t = 0; t < T; t++) {
          auto s_index = t * K + k;
          auto index = t * K * A * M + k * A * M + aind * M + mind;
          s[s_index] = eval_.recall[index];
        }
      }
    }
  }

  float mean_s;
  std::vector<float> category_mean_s(p.cat_ids.size(), 0.0);

  int count = std::count_if(s.begin(), s.end(),
                            [](float value) { return value > -1.0; });

  if (count == 0) {
    mean_s = -1.0;
    category_mean_s = std::vector<float>(p.cat_ids.size(), -1.0);
  } else {
    auto get_mean = [](std::vector<float> &input) {
      std::vector<float> array = input;
      auto pend = std::remove_if(array.begin(), array.end(),
                                 [](float value) { return value <= -1.0; });

      array.erase(pend, array.end());
      float sum = std::accumulate(array.begin(), array.end(), 0.0f);
      if (!array.empty()) {
        float count = static_cast<float>(array.size());
        return sum / count;
      } else {
        return -1.0f;
      }
    };
    mean_s = get_mean(s);
    for (int n = 0; n < p.cat_ids.size(); n++) {
      std::vector<float> category_slice;
      if (ap) {
        if (thr_index != -1) {
          category_slice.resize(R);
          for (int r = 0; r < R; r++) {
            category_slice[r] = s[r * K + n];
          }
        } else {
          category_slice.resize(T * R);
          for (int t = 0; t < T; t++) {
            for (int r = 0; r < R; r++) {
              category_slice[t * R + r] = s[t * R * K + r * K + n];
            }
          }
        }
      } else {
        if (thr_index != -1) {
          category_slice.resize(1);
          category_slice[0] = s[n];
        } else {
          category_slice.resize(T);
          for (int t = 0; t < T; t++) {
            category_slice[t] = s[t * K + n];
          }
        }
      }
      category_mean_s[n] = get_mean(category_slice);
    }
  }

  return std::make_pair(mean_s, category_mean_s);
}

void COCOevalBase::SummarizeDets() {
  std::vector<float> stats(12, 0.0);
  std::vector<std::vector<float>> category_stats(12, std::vector<float>());
  auto result = SummarizeInternal(true);
  stats[0] = result.first;
  category_stats[0] = result.second;
  result = SummarizeInternal(true, 0.5, "all", params_.max_dets[2]);
  stats[1] = result.first;
  category_stats[1] = result.second;
  result = SummarizeInternal(true, 0.75, "all", params_.max_dets[2]);
  stats[2] = result.first;
  category_stats[2] = result.second;
  result = SummarizeInternal(true, -1.0, "small", params_.max_dets[2]);
  stats[3] = result.first;
  category_stats[3] = result.second;
  result = SummarizeInternal(true, -1.0, "medium", params_.max_dets[2]);
  stats[4] = result.first;
  category_stats[4] = result.second;
  result = SummarizeInternal(true, -1.0, "large", params_.max_dets[2]);
  stats[5] = result.first;
  category_stats[5] = result.second;
  result = SummarizeInternal(false, -1.0, "all", params_.max_dets[0]);
  stats[6] = result.first;
  category_stats[6] = result.second;
  result = SummarizeInternal(false, -1.0, "all", params_.max_dets[1]);
  stats[7] = result.first;
  category_stats[7] = result.second;
  result = SummarizeInternal(false, -1.0, "all", params_.max_dets[2]);
  stats[8] = result.first;
  category_stats[8] = result.second;
  result = SummarizeInternal(false, -1.0, "small", params_.max_dets[2]);
  stats[9] = result.first;
  category_stats[9] = result.second;
  result = SummarizeInternal(false, -1.0, "medium", params_.max_dets[2]);
  stats[10] = result.first;
  category_stats[10] = result.second;
  result = SummarizeInternal(false, -1.0, "large", params_.max_dets[2]);
  stats[11] = result.first;
  category_stats[11] = result.second;
  stats_ = stats;
  category_stats_ = category_stats;
}

void COCOevalBase::Summarize() {
  auto tic = std::chrono::system_clock::now();
  if (eval_.precision.empty()) {
    std::cout << "Please run accumulate() first" << std::endl;
    return;
  }

  if ((params_.iou_type == kBbox) || (params_.iou_type == kSegm)) {
    SummarizeDets();
  } else {
    std::cout << "This iou type is not supported." << std::endl;
  }

  auto toc = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = toc - tic;
  std::cout << "Summarize Done (t= " << elapsed_seconds.count() << ")s."
            << std::endl;
}

}  // namespace pycocotools
