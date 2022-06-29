# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import json


def save_log_info(solver, config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    config['num_samples'] = 4195197692
    config['eval_num_samples'] = 89137319
    config['global_batch_size'] = solver.batchsize
    config['opt_base_learning_rate'] = solver.lr
    config['sgd_opt_base_learning_rate'] = solver.lr
    config['sgd_opt_learning_rate_decay_poly_power'] = solver.decay_power
    config['opt_learning_rate_warmup_steps'] = solver.warmup_steps
    config['opt_learning_rate_warmup_factor'] = 0.0
    config['lr_decay_start_steps'] = solver.decay_start
    config['sgd_opt_learning_rate_decay_steps'] = solver.decay_steps
    config['gradient_accumulation_steps'] = 1

    with open(config_file, 'w') as f:
        json.dump(config, f)
