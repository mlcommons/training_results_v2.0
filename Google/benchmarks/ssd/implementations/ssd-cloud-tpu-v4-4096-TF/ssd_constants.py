# Copyright 2018 Google. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Central location for all constants related to MLPerf SSD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# ==============================================================================
# == Dataset ===================================================================
# ==============================================================================
DATASET = "OPENIMAGES_MLPERF"
assert DATASET in ("COCO", "OPENIMAGES_MLPERF")

if DATASET == "COCO":
  USE_BACKGROUND_CLASS = True
  # TODO: MLPerf uses 80, but COCO documents 90.
  # (RetinaNet uses 90)
  # Update(taylorrobie): Labels > 81 show up in the pipeline. This will need to
  #                      be resolved.
  # Including "no class". Not all COCO classes are used.
  NUM_CLASSES = 80 + int(USE_BACKGROUND_CLASS)

  # Note: Zero is special. (Background class) CLASS_INV_MAP[0] must be zero.
  CLASS_INV_MAP = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18,
                   19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                   37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52,
                   53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                   72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
                   88, 89, 90)
  _MAP = {j: i for i, j in enumerate(CLASS_INV_MAP)}
  CLASS_MAP = tuple(_MAP.get(i, -1) for i in range(max(CLASS_INV_MAP) + 1))
  # Target COCO/AP for mlperf.
  EVAL_TARGET = 0.23
  NUM_TRAIN_EXAMPLES = 117266
  EVAL_EXAMPLES = 5000
  IMAGE_SIZE = 300
elif DATASET == "OPENIMAGES_MLPERF":
  # Binary focal loss doesn't require background class
  USE_BACKGROUND_CLASS = False
  # The whole OpenImages dataset has 601 classes. The OpenImages_MLPerf subset
  # has 264. Fortunately, the preprocessed data has already made the subset's
  # class ids dense [0, 263].
  NUM_CLASSES = 264 + int(USE_BACKGROUND_CLASS)
  # CLASS_INV_MAP[class_id_in_our_model] = class_id_in_annotation
  CLASS_INV_MAP = tuple(range(NUM_CLASSES))
  # CLASS_MAP[class_id_in_annotation] = class_id_in_our_model
  CLASS_MAP = tuple(range(NUM_CLASSES))
  # Target COCO/AP for mlperf.
  EVAL_TARGET = 0.34
  NUM_TRAIN_EXAMPLES = 1170301
  EVAL_SAMPLES = 24781
  IMAGE_SIZE = 800

# ==============================================================================
# == Model =====================================================================
# ==============================================================================
SPACE_TO_DEPTH_BLOCK_SIZE = 2

# Whether to split tensors by FPN levels in:
# (a) Preprocessing and generating labels
# (b) Loss computation during training
# When SPLIT_LEVEL is False, we treat it as if there's one level.
# The tensors will be concat'd along the box dimension. It is then put in a
# singleton list and treated as a single level, so that many functions
# expecting a list of per-level result still work correctly.
SPLIT_LEVEL = False

# Anchor boxes
# RetinaNet: each level has 3 anchor sizes, each with 3 aspect ratio. So there
# are 9 anchor per location per level.
NUM_SSD_BOXES = 120087
ANCHOR_SIZES = tuple((x, int(x * 2**(1.0 / 3)), int(x * 2**(2.0 / 3)))
                     for x in [32, 64, 128, 256, 512])
ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)
# used by detection head to create output size
NUM_ANCHORS_PER_LOCATION_PER_LEVEL = 9
assert NUM_ANCHORS_PER_LOCATION_PER_LEVEL == len(ANCHOR_SIZES[0]) * len(
    ASPECT_RATIOS[0])
FEATURE_SIZES = (100, 50, 25, 13, 7
                )  # calculated based on model architecture and IMAGE_SIZE
assert len(FEATURE_SIZES) == len(ANCHOR_SIZES) == len(ASPECT_RATIOS)

# https://github.com/mlcommons/training/blob/8b73f726a5c1fc7299cdae635b687ca170b860e0/single_stage_detector/ssd/model/retinanet.py#L192
SCALE_XY = 1.0
SCALE_HW = 1.0
BOX_CODER_SCALES = (1 / SCALE_XY, 1 / SCALE_XY, 1 / SCALE_HW, 1 / SCALE_HW)
# Clamp decoded h and w before sending them into tf.exp()
BOX_DECODE_HW_CLAMP = math.log(1000. / 16)

# Matcher
UNMATCHED_CLS_TARGET = -1
# If we use background class, UNMATCHED_CLS_TARGET must be 0.
# If we don't use background class, UNMATCHED_CLS_TARGET should not be mapped
# to a valid class id.
assert (USE_BACKGROUND_CLASS and
        UNMATCHED_CLS_TARGET == 0) or (not USE_BACKGROUND_CLASS and
                                       UNMATCHED_CLS_TARGET < 0)
MATCH_IGNORED = -2
MATCH_THRESHOLD_HI = 0.5
MATCH_THRESHOLD_LO = 0.4

# https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)

# SSD Cropping
NUM_CROP_PASSES = 50
CROP_MIN_IOU_CHOICES = (0, 0.1, 0.3, 0.5, 0.7, 0.9)
P_NO_CROP_PER_PASS = 1 / (len(CROP_MIN_IOU_CHOICES) + 1)

# Hard example mining
NEGS_PER_POSITIVE = 3

# Batch normalization
BATCH_NORM_DECAY = 0
BATCH_NORM_EPSILON = 1e-5

# ==============================================================================
# == Optimizer =================================================================
# ==============================================================================
BASE_LEARNING_RATE = 1e-4
LEARNING_RATE_WARMUP_EPOCHS = 1.0
LEARNING_RATE_WARMUP_FACTOR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
DEFAULT_BATCH_SIZE = 32.0

# ==============================================================================
# == Keys ======================================================================
# ==============================================================================
BOXES = "boxes"
CLASSES = "classes"
NUM_MATCHED_BOXES = "num_matched_boxes"
IMAGE = "image"
SOURCE_ID = "source_id"
RAW_SHAPE = "raw_shape"
IS_PADDED = "is_padded"
MATCH_RESULT = "match_result"

# ==============================================================================
# == Evaluation ================================================================
# ==============================================================================

# Note: This is based on a batch size of 32
# https://github.com/mlperf/reference/blob/master/single_stage_detector/ssd/train.py#L21-L37  # pylint: disable=line-too-long
# TODO(zqfeng) verify these against reference
CHECKPOINT_FREQUENCY = 200
CANDIDATE_LOGITS_PER_LEVEL = 1000
MAX_NUM_EVAL_BOXES = 300
OVERLAP_CRITERIA = 0.5  # Used for nonmax supression
MIN_SCORE = 0.05  # Minimum score to be considered during evaluation.
DUMMY_SCORE = -1e5  # If no boxes are matched.

# For multiprocessing.
QUEUE_SIZE = 24
WORKER_COUNT = 10

# Checkpoint mapping -- keep the TF checkpoint file in user dir so we do not
# overwrite existing checkpoint.
PYT_PICKLE_FILE = "resnext50_32x4d-7cdf4587.pickle"
TF_CHECKPOINT_FILE = "retinanet_resnext50_32x4d.ckpt"
