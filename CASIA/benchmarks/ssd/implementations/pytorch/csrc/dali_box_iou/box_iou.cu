// Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//           http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime_api.h>
#include "box_iou.h"

namespace other_ns {


__global__ void box_iou_cuda_kernel(float *box_iou, float4 *box1, float4 *box2, long num_images, long M, 
                                    long N, int idxJump) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    size_t b1_idx, b2_idx, b1_row_offset, b2_row_offset, im_id, im_offset; 
    float xmin1, xmin2, xmax1, xmax2, ymin1, ymin2, ymax1, ymax2;
    float x_tl, y_tl, x_br, y_br, w, h, inter, area1, area2, iou;
          
    for (long i = idx; i < num_images * M * N; i += idxJump){
        im_id = i / (M * N);
        im_offset = i % (M * N);
        b1_idx = im_offset / N;
        b2_idx = i % N;
        b1_row_offset = im_id * M + b1_idx;
        b2_row_offset = im_id * N + b2_idx;

        xmin1 = box1[b1_row_offset].x;
        ymin1 = box1[b1_row_offset].y;
        xmax1 = box1[b1_row_offset].z;
        ymax1 = box1[b1_row_offset].w;
        xmin2 = box2[b2_row_offset].x;
        ymin2 = box2[b2_row_offset].y;
        xmax2 = box2[b2_row_offset].z;
        ymax2 = box2[b2_row_offset].w;

        if (xmin1 == -1.0 && ymin1 == -1.0 && xmax1 == -1.0 && ymax1 == -1.0) {
            // do not consider padded targets
            box_iou[im_id * M * N + b1_idx * N + b2_idx] = -1;
        } else {
            x_tl = fmaxf(xmin1, xmin2);
            y_tl = fmaxf(ymin1, ymin2);

            x_br = fminf(xmax1, xmax2);
            y_br = fminf(ymax1, ymax2);                                
            w = (x_br - x_tl) < 0 ? 0.0f : (x_br - x_tl);
            h = (y_br - y_tl) < 0 ? 0.0f : (y_br - y_tl);

            inter = w * h;
            area1 = (xmax1 - xmin1) * (ymax1 - ymin1);
            area2 = (xmax2 - xmin2) * (ymax2 - ymin2);
            iou = inter / (area1 + area2 - inter);
            box_iou[im_id * M * N + b1_idx * N + b2_idx] = iou;
        }
    }  
}


template<>
void box_iou<::dali::GPUBackend>::RunImpl(::dali::DeviceWorkspace &ws) {
  const auto &box1 = ws.Input<::dali::GPUBackend>(0);
  const auto &box2 = ws.Input<::dali::GPUBackend>(1);
  const auto &shape1 = box1.shape();
  const auto &shape2 = box2.shape();
  auto &output = ws.Output<::dali::GPUBackend>(0);

  int minGridSize;
  int blockSize;

  cudaOccupancyMaxPotentialBlockSize(&minGridSize,
                                     &blockSize,
                                     (void*) box_iou_cuda_kernel,
                                     0,  // dynamic memory
                                     0); // maximum utilized threads 

  dim3 gridDim(minGridSize);
  dim3 blockDim(blockSize);
  int idxJump = minGridSize * blockSize;
  int numImages = shape1.num_samples();

  for (int sample_idx = 0; sample_idx < numImages; sample_idx++) {
    long M = shape1[sample_idx][0];
    long N = shape2[0][0];

    box_iou_cuda_kernel<<<gridDim, blockDim, 0, ws.stream()>>>(
		    (float*) output.raw_mutable_tensor(sample_idx),
		    (float4*) box1.raw_tensor(sample_idx),
		    (float4*) box2.raw_tensor(0),
		    1, M, N,
		    idxJump);
  }
}

}  // namespace other_ns


DALI_REGISTER_OPERATOR(box_iou, ::other_ns::box_iou<::dali::GPUBackend>, ::dali::GPU);

