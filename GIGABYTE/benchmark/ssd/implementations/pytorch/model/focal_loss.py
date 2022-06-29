# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F

try:
    from apex.contrib.focal_loss.focal_loss import FocalLoss
    focal_loss_opt = FocalLoss.apply
except ImportError as err:
    print("Could not import APEX fused focal loss, it's fine if you do not use --apex-focal-loss")


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


# The following focal loss implementation is similar to the previous one, besides an additional mask operation.
# The mask operation is handy when using CUDA graphs, since it will enable fixed tensor dimension (otherwise,
# for each image a different sized tensor would be used).
def sigmoid_focal_loss_masked(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    assert(reduction == "sum")

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    loss = loss * mask
    loss = loss.sum(dim=[1, 2])

    return loss


def sigmoid_focal_loss_masked_fused(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    label_smoothing: float = 0.0,
    reduction: str = "none",
    one_ptr: torch.Tensor = None
):
    assert(reduction == "sum")

    num_classes = inputs.size(2)
    inputs_ = inputs.reshape([inputs.size(0), 1, 13343, 9, num_classes])
    # -2 indicates the kernel to ignore that value
    targets_ = torch.where(mask, targets, -2)
    targets_ = targets_.reshape([inputs.size(0), 1, 13343, 9])

    # TODO: implement within the kernel and not with a loop
    loss = []
    inputs_list = torch.chunk(inputs_, inputs_.size(0))
    targets_list = torch.chunk(targets_, targets_.size(0))
    for b in range(inputs_.size(0)):
        loss.append(focal_loss_opt(inputs_list[b], targets_list[b], one_ptr, num_classes, alpha, gamma, label_smoothing))

    return torch.stack(loss)

