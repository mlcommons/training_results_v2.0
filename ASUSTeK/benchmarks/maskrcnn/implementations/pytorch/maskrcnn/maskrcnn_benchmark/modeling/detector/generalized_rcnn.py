# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.layers.nhwc import nchw_to_nhwc_transform, nhwc_to_nchw_transform

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..roi_heads.mask_head.mask_head import keep_only_positive_boxes


class Graphable(nn.Module):
    def __init__(self, cfg):
        super(Graphable, self).__init__()

        self.backbone = build_backbone(cfg)
        from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn_head
        self.anchor_generator, self.head = build_rpn_head(cfg)
        self.nhwc = cfg.NHWC
        self.stream1 = torch.cuda.Stream()

    def forward(self, images_tensor, image_sizes_tensor):
        current_stream = torch.cuda.current_stream()
        features = self.backbone(images_tensor)
        self.stream1.wait_stream(current_stream)
        with torch.cuda.stream(self.stream1):
            objectness, rpn_box_regression = self.head(features)
        with torch.no_grad():
            anchor_boxes, anchor_visibility = self.anchor_generator(image_sizes_tensor.int(), features)
        current_stream.wait_stream(self.stream1)
        return features + tuple(objectness) + tuple(rpn_box_regression) + (anchor_boxes, anchor_visibility)


class Combined_RPN_ROI(nn.Module):
    def __init__(self, cfg):
        super(Combined_RPN_ROI, self).__init__()

        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.take_shortcut = True if not cfg.MODEL.RPN_ONLY and not cfg.MODEL.KEYPOINT_ON and cfg.MODEL.MASK_ON and not cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR else False
        if self.take_shortcut:
            self.stream1 = torch.cuda.Stream()
            self.stream2 = torch.cuda.Stream()
            self.stream3 = torch.cuda.Stream()

    def forward(self, images, anchor_boxes, anchor_visibility, objectness, rpn_box_regression, targets, features):
        if self.take_shortcut:
            if self.training:
                current_stream = torch.cuda.current_stream()

                #
                # RPN inference, roi subsample
                #

                batched_anchor_data = [anchor_boxes, anchor_visibility, [tuple(image_size_wh) for image_size_wh in images.image_sizes_wh]]
                self.stream1.wait_stream(current_stream)
                with torch.no_grad():
                    proposals = self.rpn.box_selector_train(
                        batched_anchor_data, objectness, rpn_box_regression, images.image_sizes_tensor, targets
                    )
                    detections = self.roi_heads.box.loss_evaluator.subsample(proposals, targets)
                self.stream2.wait_stream(current_stream)
                self.stream3.wait_stream(current_stream)

                #
                # loss calculations
                #

                # rpn losses
                with torch.cuda.stream(self.stream1):
                    loss_objectness, loss_rpn_box_reg = self.rpn.loss_evaluator(
                        batched_anchor_data, objectness, rpn_box_regression, targets
                        )

                # box losses
                with torch.cuda.stream(self.stream2):
                    x = self.roi_heads.box.feature_extractor(features, detections)
                    class_logits, box_regression = self.roi_heads.box.predictor(x)

                    loss_classifier, loss_box_reg = self.roi_heads.box.loss_evaluator(
                        [class_logits.float()], [box_regression.float()]
                    )
                    loss_box = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)

                # mask losses
                with torch.cuda.stream(self.stream3):
                    _, _, loss_mask = self.roi_heads.mask(features, detections, targets, syncfree=True)

                current_stream.wait_stream(self.stream1)
                current_stream.wait_stream(self.stream2)
                current_stream.wait_stream(self.stream3)

                losses = {}
                losses.update(loss_box)
                losses.update(loss_mask)
                proposal_losses = {
                    "loss_objectness": loss_objectness,
                    "loss_rpn_box_reg": loss_rpn_box_reg,
                    }
                losses.update(proposal_losses)
    
                return losses
            else:
                batched_anchor_data = [anchor_boxes, anchor_visibility, [tuple(image_size_wh) for image_size_wh in images.image_sizes_wh]]
                proposals = self.rpn.box_selector_test(batched_anchor_data, objectness, rpn_box_regression, images.image_sizes_tensor)

                x = self.roi_heads.box.feature_extractor(features, proposals)
                class_logits, box_regression = self.roi_heads.box.predictor(x)
                detections = self.roi_heads.box.post_processor((class_logits, box_regression), proposals)

                x = self.roi_heads.mask.feature_extractor(features, detections, None)
                mask_logits = self.roi_heads.mask.predictor(x)
                detections = self.roi_heads.mask.post_processor(mask_logits, detections)
                    
                return detections
        else:
            proposals, proposal_losses = self.rpn(images, anchor_boxes, anchor_visibility, objectness, rpn_box_regression, targets)
            if self.roi_heads:
                x, result, detector_losses = self.roi_heads(features, proposals, targets, syncfree=True)
            ## for NHWC layout case, features[0] are NHWC features, and [1] NCHW
            ## when syncfree argument is True, x == None
            else:
                # RPN-only models don't have roi_heads
                ## TODO: take care of NHWC/NCHW cases for RPN-only case 
                x = features
                result = proposals
                detector_losses = {}

            if self.training:
                losses = {}
                losses.update(detector_losses)
                losses.update(proposal_losses)
                return losses

            return result


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.precompute_rpn_constant_tensors = cfg.PRECOMPUTE_RPN_CONSTANT_TENSORS
        self.graphable = Graphable(cfg)
        self.combined_rpn_roi = Combined_RPN_ROI(cfg)
        self.nhwc = cfg.NHWC
        self.dali = cfg.DATALOADER.DALI
        self.hybrid_loader = cfg.DATALOADER.HYBRID
        self.scale_bias_callables = None

    def compute_scale_bias(self):
        if self.scale_bias_callables is None:
            self.scale_bias_callables = []
            for module in self.graphable.modules():
                if getattr(module, "get_scale_bias_callable", None):
                    #print(module)
                    c = module.get_scale_bias_callable()
                    self.scale_bias_callables.append(c)
        for c in self.scale_bias_callables:
            c()

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if not self.hybrid_loader:
            images = to_image_list(images)
            if self.nhwc and not self.dali:
                # data-loader outputs nchw images
                images.tensors = nchw_to_nhwc_transform(images.tensors)
            elif self.dali and not self.nhwc:
                # dali pipeline outputs nhwc images
                images.tensors = nhwc_to_nchw_transform(images.tensors)
        flat_res = self.graphable(images.tensors, images.image_sizes_tensor)
        features, objectness, rpn_box_regression, anchor_boxes, anchor_visibility = flat_res[0:5], list(flat_res[5:10]), list(flat_res[10:15]), flat_res[15], flat_res[16]
        return self.combined_rpn_roi(images, anchor_boxes, anchor_visibility, objectness, rpn_box_regression, targets, features)
