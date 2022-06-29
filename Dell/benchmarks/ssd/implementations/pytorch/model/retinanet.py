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

import math
from collections import OrderedDict
import warnings

import torch
from torch import nn, Tensor
from torch.hub import load_state_dict_from_url
from typing import Dict, List, Tuple, Optional

from model.anchor_utils import AnchorGenerator
from model.transform import GeneralizedRCNNTransform
from model.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from model.feature_pyramid_network import LastLevelP6P7
from model.focal_loss import sigmoid_focal_loss, sigmoid_focal_loss_masked, sigmoid_focal_loss_masked_fused
from model.boxes import box_iou, clip_boxes_to_image, batched_nms
from model.utils import Matcher, MatcherBatch, overwrite_eps, BoxCoder

from ssd_logger import mllogger
from mlperf_logging.mllog.constants import WEIGHTS_INITIALIZATION
import utils


try:
    from apex.contrib.conv_bias_relu import ConvBiasReLU, ConvBias
except ImportError as err:
    print("Could not import APEX fused Conv-Bias-ReLU, it's fine if you do not use --apex-head")


__all__ = [
    "retinanet_from_backbone",
    "retinanet_resnet50_fpn",
    "retinanet_resnet101_fpn",
    "retinanet_resnext50_32x4d_fpn",
    "retinanet_resnext101_32x8d_fpn",
]


class GradClone_(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x):
        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone()


GradClone = GradClone_.apply


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


def cudnn_fusion_warmup(bs_list):
    hw_dim_list = [100, 50, 25, 13, 7]

    for bs in bs_list:
        for hw in hw_dim_list:
            ConvBiasReLU(torch.rand([bs, 256, hw, hw], dtype=torch.half).to(memory_format=torch.channels_last).cuda(),
                         torch.rand([256, 256, 3, 3], dtype=torch.half).to(memory_format=torch.channels_last).cuda(),
                         torch.rand([1, 256, 1, 1], dtype=torch.half).to(memory_format=torch.channels_last).cuda(), 1, 1)
            ConvBias(torch.rand([bs, 256, hw, hw], dtype=torch.half).to(memory_format=torch.channels_last).cuda(),
                     torch.rand([2376, 256, 3, 3], dtype=torch.half).to(memory_format=torch.channels_last).cuda(),
                     torch.rand([1, 2376, 1, 1], dtype=torch.half).to(memory_format=torch.channels_last).cuda(), 1, 1)
            ConvBias(torch.rand([bs, 256, hw, hw], dtype=torch.half).to(memory_format=torch.channels_last).cuda(),
                     torch.rand([36, 256, 3, 3], dtype=torch.half).to(memory_format=torch.channels_last).cuda(),
                     torch.rand([1, 36, 1, 1], dtype=torch.half).to(memory_format=torch.channels_last).cuda(), 1, 1)


class RetinaNetHead(nn.Module):
    """
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes, fusion=False):
        super().__init__()
        self.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes, fusion=fusion,
            module_name="module.head.classification_head")
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors, fusion=fusion,
            module_name="module.head.regression_head")

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        return {
            'classification': self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            'bbox_regression': self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }

    def forward(self, x):
        return [self.classification_head(x), self.regression_head(x)]


class RetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes, prior_probability=0.01, fusion=False, module_name=""):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for name, layer in self.conv.named_children():
            if isinstance(layer, nn.Conv2d):
                mllogger.event(key=WEIGHTS_INITIALIZATION, metadata={"tensor": f"{module_name}.conv.{name}.weight"})
                torch.nn.init.normal_(layer.weight, std=0.01)
                mllogger.event(key=WEIGHTS_INITIALIZATION, metadata={"tensor": f"{module_name}.conv.{name}.bias"})
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        mllogger.event(key=WEIGHTS_INITIALIZATION, metadata={"tensor": f"{module_name}.cls_logits.weight"})
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        mllogger.event(key=WEIGHTS_INITIALIZATION, metadata={"tensor": f"{module_name}.cls_logits.bias"})
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = Matcher.BETWEEN_THRESHOLDS

        self.register_buffer("one", torch.Tensor([1.]))

        self.fusion = fusion

    # --- original implementation ---
    def compute_loss(self, targets, head_outputs, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        losses = []

        cls_logits = head_outputs['cls_logits']

        for labels_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets['labels'], cls_logits, matched_idxs):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                labels_per_image[matched_idxs_per_image[foreground_idxs_per_image]]
            ] = 1.0

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            losses.append(sigmoid_focal_loss(
                cls_logits_per_image[valid_idxs_per_image],
                gt_classes_target[valid_idxs_per_image],
                reduction='sum',
            ) / max(1, num_foreground))

        # doesn't matter which targets['?'] is taken, this represent the batch size
        return _sum(losses) / len(targets['boxes'])

    def compute_loss_prologue(self, target_labels, matched_idxs, one_hot):
        # determine only the foreground
        foreground_idxs_ = matched_idxs >= 0
        num_foreground_ = foreground_idxs_.sum(dim=1)

        # find indices for which anchors should be ignored
        valid_idxs_ = matched_idxs != self.BETWEEN_THRESHOLDS

        # TODO: unable to parallelize, try again
        for i, (labels_per_image, matched_idxs_per_image, foreground_idxs_per_image) in \
                enumerate(zip(target_labels, matched_idxs, foreground_idxs_)):

            # create the target classification
            if one_hot:
                utils.ScratchPad.gt_classes_target[i][
                    foreground_idxs_per_image,
                    labels_per_image[matched_idxs_per_image[foreground_idxs_per_image]]
                ] = 1.0
            else:
                utils.ScratchPad.gt_classes_target[i][foreground_idxs_per_image] = \
                    labels_per_image[matched_idxs_per_image[foreground_idxs_per_image]]

        return utils.ScratchPad.gt_classes_target, num_foreground_, valid_idxs_

    def compute_loss_prologue_padded(self, target_labels, matched_idxs, one_hot, max_boxes):
        # buffers are initialized in init_scratchpad
        # utils.ScratchPad.gt_classes_target.fill_(0 if one_hot else -1)

        # determine only the foreground
        foreground_idxs_ = matched_idxs >= 0
        num_foreground_ = foreground_idxs_.sum(dim=1)

        # find indices for which anchors should be ignored
        valid_idxs_ = matched_idxs != self.BETWEEN_THRESHOLDS

        if one_hot:
            idxs = torch.gather(target_labels, 1, torch.where(foreground_idxs_, matched_idxs, max_boxes))
            utils.ScratchPad.gt_classes_target.scatter_(2, idxs[:, :, None], 1)
            gt_classes_target = utils.ScratchPad.gt_classes_target[:, :, :-1]
        else:
            utils.ScratchPad.gt_classes_target = \
                torch.gather(target_labels, 1, torch.where(foreground_idxs_, matched_idxs, max_boxes))
            gt_classes_target = utils.ScratchPad.gt_classes_target

        return gt_classes_target, num_foreground_, valid_idxs_

    def compute_loss_core(self, cls_logits, gt_classes_target, valid_idxs, num_foreground, fused_focal_loss=False):
        # notice that in the original implementation, the focal loss input dimension may differ
        if not fused_focal_loss:
            losses = sigmoid_focal_loss_masked(cls_logits, gt_classes_target, valid_idxs[:, :, None], reduction='sum')
        else:
            losses = sigmoid_focal_loss_masked_fused(cls_logits, gt_classes_target, valid_idxs, reduction='sum',
                                                     one_ptr=self.one)
        losses = losses / num_foreground

        return _sum(losses) / num_foreground.size(0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []

        # since weights are shared, we can cast weights and biases only one time per iteration
        if self.fusion:
            conv1_w = self.conv[0].weight.half()
            conv2_w = self.conv[2].weight.half()
            conv3_w = self.conv[4].weight.half()
            conv4_w = self.conv[6].weight.half()
            conv5_w = self.cls_logits.weight.half()
            conv1_b = self.conv[0].bias.reshape(1, -1, 1, 1).half()
            conv2_b = self.conv[2].bias.reshape(1, -1, 1, 1).half()
            conv3_b = self.conv[4].bias.reshape(1, -1, 1, 1).half()
            conv4_b = self.conv[6].bias.reshape(1, -1, 1, 1).half()
            conv5_b = self.cls_logits.bias.reshape(1, -1, 1, 1).half()

        for features in x:
            if not self.fusion:
                cls_logits = self.conv(features)
                cls_logits = self.cls_logits(cls_logits)
            else:
                cls_logits = ConvBiasReLU(features, conv1_w, conv1_b, 1, 1)
                cls_logits = ConvBiasReLU(cls_logits, conv2_w, conv2_b, 1, 1)
                cls_logits = ConvBiasReLU(cls_logits, conv3_w, conv3_b, 1, 1)
                cls_logits = ConvBiasReLU(cls_logits, conv4_w, conv4_b, 1, 1)
                cls_logits = ConvBias(cls_logits, conv5_w, conv5_b, 1, 1)

                # cloning grad in backprop to make it contiguous for fusion code
                cls_logits = GradClone(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """
    __annotations__ = {
        'box_coder': BoxCoder,
    }

    def __init__(self, in_channels, num_anchors, fusion=False, module_name=""):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        mllogger.event(key=WEIGHTS_INITIALIZATION, metadata={"tensor": f"{module_name}.bbox_reg.weight"})
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        mllogger.event(key=WEIGHTS_INITIALIZATION, metadata={"tensor": f"{module_name}.bbox_reg.bias"})
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for name, layer in self.conv.named_children():
            if isinstance(layer, nn.Conv2d):
                mllogger.event(key=WEIGHTS_INITIALIZATION, metadata={"tensor": f"{module_name}.conv.{name}.weight"})
                torch.nn.init.normal_(layer.weight, std=0.01)
                mllogger.event(key=WEIGHTS_INITIALIZATION, metadata={"tensor": f"{module_name}.conv.{name}.bias"})
                torch.nn.init.zeros_(layer.bias)

        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        self.fusion = fusion

    # --- original implementation ---
    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        losses = []

        bbox_regression = head_outputs['bbox_regression']

        for boxes_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in \
                zip(targets['boxes'], bbox_regression, anchors, matched_idxs):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = boxes_per_image[matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the regression targets
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)

            # compute the loss
            losses.append(torch.nn.functional.l1_loss(
                bbox_regression_per_image,
                target_regression,
                reduction='sum'
            ) / max(1, num_foreground))

        # doesn't matter which targets['?'] is taken, this represent the batch size
        return _sum(losses) / len(targets['boxes'])

    def compute_loss_prologue(self, target_boxes, matched_idxs, anchors):
        foreground_idxs_mask, num_foreground_, target_regression_ = [], [], []

        for boxes_per_image, anchors_per_image, matched_idxs_per_image in zip(target_boxes, anchors, matched_idxs):
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            foreground_idxs_mask.append(foreground_idxs_per_image)
            num_foreground_.append(num_foreground)

            # select only the foreground boxes
            matched_gt_boxes_per_image = boxes_per_image[matched_idxs_per_image[foreground_idxs_per_image]]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the regression targets
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)

            target_regression_.append(target_regression)

        return target_regression_, num_foreground_, foreground_idxs_mask

    def compute_loss_core(self, bbox_regression, target_regression, foreground_idxs, num_foreground):
        losses = []

        for bbox_regression_i, target_regression_i, foreground_idxs_i, num_foreground_i in \
                zip(bbox_regression, target_regression, foreground_idxs, num_foreground):

            bbox_regression_i_ = bbox_regression_i[foreground_idxs_i, :]

            losses.append(torch.nn.functional.l1_loss(bbox_regression_i_, target_regression_i, reduction='sum')
                          / max(1, num_foreground_i))

        return _sum(losses) / num_foreground.size(0)

    def compute_loss_prologue_padded(self, target_boxes, matched_idxs, anchors):
        # notice the number of boxes is padded in this implementation
        # make sure we do not trim bboxes
        # assert (matched_idxs.max() < max_boxes)

        foreground_idxs_mask = matched_idxs >= 0
        num_foreground_ = foreground_idxs_mask.sum(dim=1)
        # clamping to avoid -2, -1
        matched_idxs_clamped = torch.clamp(matched_idxs, min=0)

        # check that the premade vector size is relevant to the current batch size
        # not sure what will happen if it is not
        assert(utils.ScratchPad.batch_size_vector.size(0) == len(target_boxes))

        matched_gt_boxes_ = target_boxes[utils.ScratchPad.batch_size_vector, matched_idxs_clamped]
        target_regression_ = self.box_coder.encode_batch(matched_gt_boxes_,
                                                         torch.stack(anchors)) * foreground_idxs_mask[:, :, None]

        return target_regression_, num_foreground_, foreground_idxs_mask

    def compute_loss_core_padded(self, bbox_regression, target_regression, foreground_idxs, num_foreground):
        bbox_regression_masked = bbox_regression * foreground_idxs[:, :, None]
        losses = torch.norm(bbox_regression_masked - target_regression, 1, dim=[1, 2]) / \
                 torch.max(torch.ones_like(num_foreground), num_foreground)

        # The denominator is just the batch size
        return _sum(losses) / num_foreground.size(0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_bbox_regression = []

        # since weights are shared, we can cast weights and biases only one time per iteration
        if self.fusion:
            conv1_w = self.conv[0].weight.half()
            conv2_w = self.conv[2].weight.half()
            conv3_w = self.conv[4].weight.half()
            conv4_w = self.conv[6].weight.half()
            conv5_w = self.bbox_reg.weight.half()
            conv1_b = self.conv[0].bias.reshape(1, -1, 1, 1).half()
            conv2_b = self.conv[2].bias.reshape(1, -1, 1, 1).half()
            conv3_b = self.conv[4].bias.reshape(1, -1, 1, 1).half()
            conv4_b = self.conv[6].bias.reshape(1, -1, 1, 1).half()
            conv5_b = self.bbox_reg.bias.reshape(1, -1, 1, 1).half()

        for features in x:
            if not self.fusion:
                bbox_regression = self.conv(features)
                bbox_regression = self.bbox_reg(bbox_regression)
            else:
                bbox_regression = ConvBiasReLU(features, conv1_w, conv1_b, 1, 1)
                bbox_regression = ConvBiasReLU(bbox_regression, conv2_w, conv2_b, 1, 1)
                bbox_regression = ConvBiasReLU(bbox_regression, conv3_w, conv3_b, 1, 1)
                bbox_regression = ConvBiasReLU(bbox_regression, conv4_w, conv4_b, 1, 1)
                bbox_regression = ConvBias(bbox_regression, conv5_w, conv5_b, 1, 1)

                # cloning grad in backprop to make it contiguous for fusion code
                bbox_regression = GradClone(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)


class RetinaNet(nn.Module):
    """
    Implements RetinaNet.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training.
        topk_candidates (int): Number of best detections to keep before NMS.

    Example:

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import RetinaNet
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # RetinaNet needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the network generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(
        >>>     sizes=((32, 64, 128, 256, 512),),
        >>>     aspect_ratios=((0.5, 1.0, 2.0),)
        >>> )
        >>>
        >>> # put the pieces together inside a RetinaNet model
        >>> model = RetinaNet(backbone,
        >>>                   num_classes=2,
        >>>                   anchor_generator=anchor_generator)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """
    __annotations__ = {
        'box_coder': BoxCoder,
        'proposal_matcher': Matcher,
    }

    def __init__(self, backbone, num_classes, data_layout='channels_first', head_fusion=False,
                 # transform parameters
                 image_size=None, image_mean=None, image_std=None,
                 # Anchor parameters
                 anchor_generator=None, head=None,
                 # Detection parameters
                 proposal_matcher=None,
                 score_thresh=0.05,
                 nms_thresh=0.5,
                 detections_per_img=300,
                 fg_iou_thresh=0.5, bg_iou_thresh=0.4,
                 topk_candidates=1000):
        super().__init__()

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")
        self.backbone = backbone
        self.data_layout = data_layout

        assert isinstance(anchor_generator, (AnchorGenerator, type(None)))

        if anchor_generator is None:
            anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        self.anchor_generator = anchor_generator
        self.anchors = None

        if head is None:
            head = RetinaNetHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes,
                                 fusion=head_fusion)
        self.head = head

        if proposal_matcher is None:
            proposal_matcher = Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
            )
        else:
            warnings.warn('proposal_matcher_batch is statically assigned to MatcherBatch')
        self.proposal_matcher = proposal_matcher
        self.proposal_matcher_batch = MatcherBatch(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.anchors = None

        if image_size is None:
            image_size = [800, 800]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]

        self.transform = GeneralizedRCNNTransform(image_size=image_size,
                                                  image_mean=image_mean, image_std=image_std)

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    # --- original code ---
    def get_matched_idxs(self, target_boxes):
        matched_idxs = []
        for anchors_per_image, boxes_per_image in zip(self.anchors, target_boxes):
            if boxes_per_image.numel() == 0:
                matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64,
                                               device=anchors_per_image.device))
                continue

            match_quality_matrix = box_iou(boxes_per_image, anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return torch.stack(matched_idxs)

    # --- parallel implementation ---
    # this implementation is not in use, since (1) it is done as part of DALI pipe; and (2) because of the
    # significant padding to target_boxes, box_iou has significant computational overheads
    def get_matched_idxs_padded(self, target_boxes, batch_sz, max_boxes):
        target_boxes_ = target_boxes.reshape(-1, 4)

        match_quality_matrix = box_iou(target_boxes_, self.anchors[0])
        match_quality_matrix = match_quality_matrix.reshape([batch_sz, max_boxes, -1])
        matched_idxs = self.proposal_matcher_batch(match_quality_matrix)

        return matched_idxs

    # --- original code ---
    def compute_loss(self, targets, head_outputs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor]) -> Dict[str, Tensor]

        matched_idxs = []
        for anchors_per_image, boxes_per_image in zip(self.anchors, targets['boxes']):

            # Uncomment to support trim of targets according to MAX_BOXES, so can be used a reference
            # boxes_per_image = boxes_per_image[0:MAX_BOXES, :]

            if boxes_per_image.numel() == 0:
                matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64,
                                               device=anchors_per_image.device))
                continue

            match_quality_matrix = box_iou(boxes_per_image, anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.head.compute_loss(targets, head_outputs, self.anchors, matched_idxs)

    def update_anchors(self, images, device, features=None, dtype=torch.float16, force=False):
        # TODO: should perhaps create once in the relevant constructor
        if self.anchors is None or force is True:
            if features is None:
                # forward_opt uses the default grid size (100, 50, 25, 13, 7)
                # images is the image tensor shape
                self.anchors = self.anchor_generator.forward_opt(image_shape=images, device=device, dtype=dtype)
            else:
                # using the old method if the features are passed
                self.anchors = self.anchor_generator.forward(images, features)

    def eval_postprocess_detections(self, head_outputs, anchors, image_shapes):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_outputs['cls_logits']
        box_regression = head_outputs['bbox_regression']

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in \
                    zip(box_regression_per_image, logits_per_image, anchors_per_image):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = min(self.topk_candidates, topk_idxs.size(0))
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode='floor')
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(box_regression_per_level[anchor_idxs],
                                                               anchors_per_level[anchor_idxs])
                boxes_per_level = clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]

            detections.append({
                'boxes': image_boxes[keep],
                'scores': image_scores[keep],
                'labels': image_labels[keep],
            })

        return detections

    def eval_postprocess(self, images, features, targets, head_outputs):
        # recover level sizes
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
        HW = 0
        for v in num_anchors_per_level:
            HW += v
        HWA = head_outputs['cls_logits'].size(1)
        A = HWA // HW
        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

        # split outputs per level
        split_head_outputs: Dict[str, List[Tensor]] = {}
        for k in head_outputs:
            split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in self.anchors]

        # get the original image sizes
        original_image_sizes = []
        for target in targets:
            original_image_sizes.append(target['original_image_size'])

        # compute the detections
        detections = self.eval_postprocess_detections(split_head_outputs, split_anchors,
                                                      [(image.size(1), image.size(2)) for image in images])
        detections = self.transform.postprocess(detections,
                                                [(image.size(1), image.size(2)) for image in images],
                                                original_image_sizes)

        return detections

    def validate_input(self, images, targets):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for boxes in targets["boxes"]:
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # check for degenerate boxes
        if targets is not None:
            for target_idx, boxes in enumerate(targets["boxes"]):
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

    def forward(self, images: Tensor) -> Tuple[Tensor]:
        """
        Args:
            images (Tensor): images to be processed

        Returns:
            result (Tuple[Tensor]): the output from the model; [0]: pyramid 100x100, [1] 50x50, [2] 25x25,
            [3] 13x13, [4] 7x7, [5] cls head, [6] bbox head
        """

        # get the features from the backbone
        features = self.backbone(images)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)

        features.extend(head_outputs)
        out = tuple(features)

        return out


model_urls = {
    'retinanet_resnet50_fpn_coco':
        'https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth',
}


def retinanet_resnet50_fpn(num_classes, image_size, data_layout='channels_first',
                           pretrained=False, progress=True, pretrained_backbone=True,
                           trainable_backbone_layers=None):
    """
    Constructs a RetinaNet model with a ResNet-50-FPN backbone.

    Reference: `"Focal Loss for Dense Object Detection" <https://arxiv.org/abs/1708.02002>`_.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Example::

        >>> model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        num_classes (int): number of output classes of the model (including the background)
        image_size (list(int, int)): Image size
        data_layout (str): model data layout (channels_first or channels_last)
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, returned_layers=[2, 3, 4],
                                   extra_blocks=LastLevelP6P7(256, 256, module_name="module.backbone.fpn.extra_blocks"),
                                   trainable_layers=trainable_backbone_layers)
    model = RetinaNet(backbone=backbone, num_classes=num_classes, data_layout=data_layout, image_size=image_size)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['retinanet_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)
    return model


def retinanet_resnext50_32x4d_fpn(num_classes, image_size, data_layout='channels_first',
                                  pretrained=False, progress=True, pretrained_backbone=True,
                                  trainable_backbone_layers=None, jit=False, head_fusion=False):
    """
    Constructs a RetinaNet model with a resnext50_32x4d-FPN backbone.

    Reference: `"Focal Loss for Dense Object Detection" <https://arxiv.org/abs/1708.02002>`_.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Example::

        >>> model = torchvision.models.detection.retinanet_resnext50_32x4d_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        num_classes (int): number of output classes of the model (including the background)
        image_size (list(int, int)): Image size
        data_layout (str): model data layout (channels_first or channels_last)
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = resnet_fpn_backbone('resnext50_32x4d', pretrained_backbone, returned_layers=[2, 3, 4],
                                   extra_blocks=LastLevelP6P7(256, 256, module_name="module.backbone.fpn.extra_blocks"),
                                   trainable_layers=trainable_backbone_layers,
                                   jit=jit)
    model = RetinaNet(backbone=backbone, num_classes=num_classes, data_layout=data_layout, image_size=image_size,
                      head_fusion=head_fusion)
    if pretrained:
        raise ValueError("Torchvision doesn't have a pretrained retinanet_resnext50_32x4d_fpn model")

    return model


def retinanet_resnet101_fpn(num_classes, image_size, data_layout='channels_first',
                            pretrained=False, progress=True, pretrained_backbone=True,
                            trainable_backbone_layers=None):
    """
    Constructs a RetinaNet model with a ResNet-101-FPN backbone.

    Reference: `"Focal Loss for Dense Object Detection" <https://arxiv.org/abs/1708.02002>`_.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Example::

        >>> model = torchvision.models.detection.retinanet_resnet101_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        num_classes (int): number of output classes of the model (including the background)
        image_size (list(int, int)): Image size
        data_layout (str): model data layout (channels_first or channels_last)
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = resnet_fpn_backbone('resnet101', pretrained_backbone, returned_layers=[2, 3, 4],
                                   extra_blocks=LastLevelP6P7(256, 256, module_name="module.backbone.fpn.extra_blocks"),
                                   trainable_layers=trainable_backbone_layers)
    model = RetinaNet(backbone=backbone, num_classes=num_classes, data_layout=data_layout, image_size=image_size)
    if pretrained:
        raise ValueError("Torchvision doesn't have a pretrained retinanet_resnet101_fpn model")

    return model


def retinanet_resnext101_32x8d_fpn(num_classes, image_size, data_layout='channels_first',
                                   pretrained=False, progress=True, pretrained_backbone=True,
                                   trainable_backbone_layers=None):
    """
    Constructs a RetinaNet model with a resnext101_32x8d-FPN backbone.

    Reference: `"Focal Loss for Dense Object Detection" <https://arxiv.org/abs/1708.02002>`_.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Example::

        >>> model = torchvision.models.detection.retinanet_resnext101_32x8d_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        num_classes (int): number of output classes of the model (including the background)
        image_size (list(int, int)): Image size
        data_layout (str): model data layout (channels_first or channels_last)
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = resnet_fpn_backbone('resnext101_32x8d', pretrained_backbone, returned_layers=[2, 3, 4],
                                   extra_blocks=LastLevelP6P7(256, 256, module_name="module.backbone.fpn.extra_blocks"),
                                   trainable_layers=trainable_backbone_layers)
    model = RetinaNet(backbone=backbone, num_classes=num_classes, data_layout=data_layout, image_size=image_size)
    if pretrained:
        raise ValueError("Torchvision doesn't have a pretrained retinanet_resnext101_32x8d_fpn model")

    return model


def retinanet_from_backbone(backbone,
                            num_classes=91, data_layout='channels_first', image_size=None,
                            pretrained=False, progress=True, pretrained_backbone=True,
                            trainable_backbone_layers=None, jit=False, head_fusion=False):
    if image_size is None:
        image_size = [800, 800]

    if backbone == "resnet50":
        return retinanet_resnet50_fpn(num_classes=num_classes, data_layout=data_layout, image_size=image_size,
                                      pretrained=pretrained, progress=progress,
                                      pretrained_backbone=pretrained_backbone,
                                      trainable_backbone_layers=trainable_backbone_layers)
    elif backbone == "resnext50_32x4d":
        return retinanet_resnext50_32x4d_fpn(num_classes=num_classes, data_layout=data_layout, image_size=image_size,
                                             pretrained=pretrained, progress=progress,
                                             pretrained_backbone=pretrained_backbone,
                                             trainable_backbone_layers=trainable_backbone_layers, jit=jit,
                                             head_fusion=head_fusion)
    elif backbone == "resnet101":
        return retinanet_resnet101_fpn(num_classes=num_classes, data_layout=data_layout, image_size=image_size,
                                       pretrained=pretrained, progress=progress,
                                       pretrained_backbone=pretrained_backbone,
                                       trainable_backbone_layers=trainable_backbone_layers)
    elif backbone == "resnext101_32x8d":
        return retinanet_resnext101_32x8d_fpn(num_classes=num_classes, data_layout=data_layout, image_size=image_size,
                                              pretrained=pretrained, progress=progress,
                                              pretrained_backbone=pretrained_backbone,
                                              trainable_backbone_layers=trainable_backbone_layers)
    else:
        raise ValueError(f"Unknown backbone {backbone}")
