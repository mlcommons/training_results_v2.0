# Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np

from mxnet.gluon.loss import Loss, SoftmaxCrossEntropyLoss

from model.layers import GatherBlock, SplitBlock


class DiceCELoss(Loss):
    def __init__(self,
                 to_onehot_y: bool = True,
                 use_softmax: bool = True,
                 include_background: bool = False,
                 spatial_group_size=1):
        super(DiceCELoss, self).__init__(weight=None, batch_axis=0)
        self.channel_axis = -1
        self.spatial_group_size = spatial_group_size
        self.dice = Dice(to_onehot_y=to_onehot_y, use_softmax=use_softmax, include_background=include_background,
                         spatial_group_size=self.spatial_group_size)
        self.cross_entropy = SoftmaxCrossEntropyLoss(sparse_label=True, axis=self.channel_axis)
        if self.spatial_group_size > 1:
            self.loss_gather = GatherBlock(spatial_group_size=self.spatial_group_size)
            self.split = SplitBlock(spatial_group_size=self.spatial_group_size)

    def hybrid_forward(self, F, y_pred, y_true, *args, **kwargs):
        if self.spatial_group_size > 1:
            y_true = self.split(y_true)
        dice = self.dice(y_pred, y_true)
        dice = 1.0 - F.mean(dice)
        ce_loss = self.cross_entropy(y_pred, y_true)
        if self.spatial_group_size > 1:
            ce_loss = self.loss_gather(F.reshape(ce_loss, shape=(1, 1, -1)))
        ce_loss = F.mean(ce_loss)
        return (dice + ce_loss) / 2


class DiceScore(Loss):
    def __init__(self,
                 to_onehot_y: bool = True,
                 use_argmax: bool = True,
                 include_background: bool = False,
                 spatial_group_size: int = 1):
        super(DiceScore, self).__init__(weight=None, batch_axis=0)
        self.spatial_group_size = spatial_group_size
        if spatial_group_size > 1:
            self.split = SplitBlock(spatial_group_size=self.spatial_group_size)
        else:
            self.split = None
        self.dice = Dice(to_onehot_y=to_onehot_y, to_onehot_x=True, use_softmax=False,
                         use_argmax=use_argmax, include_background=include_background, 
                         spatial_group_size=spatial_group_size)

    def hybrid_forward(self, F, y_pred, y_true, *args, **kwargs):
        if self.spatial_group_size > 1:
            y_true = self.split(y_true)
        return F.mean(self.dice(y_pred, y_true), axis=0)


class Dice(Loss):
    def __init__(self,
                 to_onehot_y: bool = True,
                 to_onehot_x: bool = False,
                 use_softmax: bool = True,
                 use_argmax: bool = False,
                 include_background: bool = False,
                 spatial_group_size: int = 1
                 ):
        super(Dice, self).__init__(weight=None, batch_axis=0)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.to_onehot_x = to_onehot_x
        self.use_softmax = use_softmax
        self.use_argmax = use_argmax
        self.smooth_nr = 1e-6
        self.smooth_dr = 1e-6
        self.spatial_group_size = spatial_group_size
        self.cast_type = np.float32
        if spatial_group_size > 1:
            self.loss_gather = GatherBlock(spatial_group_size=self.spatial_group_size)
            self.split = SplitBlock(spatial_group_size=self.spatial_group_size)
        else:
            self.loss_gather = None
            self.split = None

    def cast(self, dtype):
        self.cast_type = np.float16 if ((dtype == 'float16') or (dtype == np.float16)) else np.float32

    def hybrid_forward(self, F, y_pred, y_true, *args, **kwargs):
        channel_axis = -1
        reduce_axis = list(range(1, 4))
        num_pred_ch = 3

        if self.use_softmax:
            y_pred = F.softmax(y_pred, axis=channel_axis)
        elif self.use_argmax:
            y_pred = F.argmax(y_pred, axis=channel_axis, keepdims=True)

        if self.to_onehot_y:
            y_true = self.to_one_hot(F, y_true, channel_axis, num_pred_ch)

        if self.to_onehot_x:
            y_pred = self.to_one_hot(F, y_pred, channel_axis, num_pred_ch)

        if not self.include_background:
            assert num_pred_ch > 1, \
                f"To exclude background the prediction needs more than one channel. Got {num_pred_ch}."
            y_true = F.slice_axis(y_true, axis=-1, begin=1, end=3)
            y_pred = F.slice_axis(y_pred, axis=-1, begin=1, end=3)

        intersection = F.sum(y_true * y_pred, axis=reduce_axis)
        target_sum = F.sum(y_true, axis=reduce_axis)
        prediction_sum = F.sum(y_pred, axis=reduce_axis)
        if self.spatial_group_size > 1:
            loss_params = F.concat(intersection, prediction_sum, target_sum, dim=1)
            loss_params = self.loss_gather(loss_params)
            loss_params = F.reshape(loss_params, shape=(self.spatial_group_size, 3, 2))
            loss_params = F.sum(loss_params, axis=0)
            loss_params = F.split(loss_params, axis=0, num_outputs=3)
            intersection, prediction_sum, target_sum = loss_params
        dice = (2.0 * intersection + self.smooth_nr) / (target_sum + prediction_sum + self.smooth_dr)
        return dice

    def to_one_hot(self, F, array, channel_axis, num_pred_ch):
        return F.one_hot(F.squeeze(array, axis=channel_axis), depth=num_pred_ch)
