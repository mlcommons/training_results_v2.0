# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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
import argparse
import os

from . import utils as mllogger
import mlperf_logging.mllog as mllog

# map keys traditionally used in hugectr configs
# to more descriptive ones more suitable for MLPerf
hugectr_to_mlperf_layer_name = {
    'sparse_embedding1': 'embeddings',
    'fc1': 'bottom_mlp_dense1',
    'fc2': 'bottom_mlp_dense2',
    'fc3': 'bottom_mlp_dense3',
    'fc4': 'top_mlp_dense1',
    'fc5': 'top_mlp_dense2',
    'fc6': 'top_mlp_dense3',
    'fc7': 'top_mlp_dense4',
    'fc8': 'top_mlp_dense5'
}


def log_hparams(config, time_ms=None):
    mllogger.log_event(time_ms=time_ms,
                       key='eval_samples',
                       value=config['eval_num_samples'])
    mllogger.log_event(time_ms=time_ms,
                       key='global_batch_size',
                       value=config['global_batch_size'])
    mllogger.log_event(time_ms=time_ms,
                       key='opt_base_learning_rate',
                       value=config['opt_base_learning_rate'])
    mllogger.log_event(time_ms=time_ms,
                       key='sgd_opt_base_learning_rate',
                       value=config['sgd_opt_base_learning_rate'])
    mllogger.log_event(time_ms=time_ms,
                       key='sgd_opt_learning_rate_decay_poly_power',
                       value=config['sgd_opt_learning_rate_decay_poly_power'])
    mllogger.log_event(time_ms=time_ms,
                       key='opt_learning_rate_warmup_steps',
                       value=config['opt_learning_rate_warmup_steps'])
    mllogger.log_event(time_ms=time_ms,
                       key='opt_learning_rate_warmup_factor',
                       value=0.0)  # not configurable
    mllogger.log_event(time_ms=time_ms,
                       key='lr_decay_start_steps',
                       value=config['lr_decay_start_steps'])
    mllogger.log_event(time_ms=time_ms,
                       key='sgd_opt_learning_rate_decay_steps',
                       value=config['sgd_opt_learning_rate_decay_steps'])
    mllogger.log_event(time_ms=time_ms,
                       key='gradient_accumulation_steps',
                       value=1)  # not configurable


def log_config(config, time_ms=None):
    # print hparams and submission info on the first node only
    if os.environ.get('SLURM_NODEID', '0') == '0':
        mllogger.mlperf_submission_log('dlrm', time_ms)
        log_hparams(config, time_ms)

        for mlperf_name in hugectr_to_mlperf_layer_name.values():
            mllogger.log_event(mllog.constants.WEIGHTS_INITIALIZATION,
                               time_ms=time_ms,
                               metadata={'tensor': mlperf_name})


class LogConverter:
    def __init__(self, start_timestamp):
        self.start_time = start_timestamp
        self._last_eval_accuracy = -1.

    def _get_log_foo(self, key):
        if '_start' in key:
            return mllogger.log_start
        if '_end' in key or '_stop' in key:
            return mllogger.log_end
        else:
            return mllogger.log_event

    def _get_key(self, data):
        key = data[0]
        if key == 'init_end':
            return 'init_stop'
        if key == 'train_epoch_start':
            return 'epoch_start'
        if key == 'train_epoch_end':
            return 'epoch_stop'
        return key

    def _get_value(self, data):
        if data[0] == 'eval_accuracy':
            return float(data[1])
        if data[0] == 'train_samples':
            return int(data[1])

    def _get_metadata(self, data):
        if data[0] == 'eval_accuracy':
            self._last_eval_accuracy = float(data[1])
            return {'epoch_num': float(data[2]) + 1}
        if 'eval' in data[0]:
            return {'epoch_num': float(data[1]) + 1}
        if 'epoch' in data[0]:
            return {'epoch_num': int(data[1]) + 1}
        if data[0] == 'run_stop':
            return {'status': 'success' if self._last_eval_accuracy > 0.8025 else 'aborted'}

    def _get_kvm(self, data):
        key = self._get_key(data)
        value = self._get_value(data)
        metadata = self._get_metadata(data)
        return key, value, metadata

    def _get_time_ms(self, ms):
        return self.start_time + int(float(ms))

    def validate_event(self, event):
        try:
            float(event[0])

            if not event[1].isidentifier():
                return False

            for x in event[2:]:
                float(x)
            return True
        except:
            return False

    def log_tracked_stats(self, line):
        """
        Read stats from the final log line:
            "Hit target accuracy AUC 0.802500 at 68274/75868 iterations (...). Average speed 37095001.70 records/s."
        """
        if line.startswith("Hit target accuracy"):
            line_split = line.split()
            mllogger.log_event(key="tracked_stats",
                               value={'throughput': float(line_split[-2])},
                               metadata={'step': eval(line_split[6])})

    def log_event(self, event_log):
        if self.validate_event(event_log):
            log_foo = self._get_log_foo(event_log[1])
            key, value, metadata = self._get_kvm(event_log[1:])
            time_ms = self._get_time_ms(event_log[0])

            log_foo(key=key, value=value, metadata=metadata, time_ms=time_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str,
                        help='Path to the logs to be translated')

    parser.add_argument('--config_file', type=str,
                        help='HugeCTR input config file in JSON format')

    parser.add_argument('--start_timestamp', type=int,
                        help='Seconds since 1970-01-01 00:00:00 UTC at the time of training start')
    args = parser.parse_args()

    # Convert to ms to be consistent with the MLPerf logging API
    start_timestamp_ms = args.start_timestamp * 1000

    with open(args.config_file, 'r') as f:
        config = json.load(f)
    log_config(config, start_timestamp_ms)

    converter = LogConverter(
        start_timestamp=start_timestamp_ms,
    )

    with open(args.log_path, errors='ignore') as f:
        log_lines = f.readlines()

    for line in log_lines:
        event_log = [x.strip() for x in line.strip().strip('][\x08 ,').split(',')]
        converter.log_event(event_log)
        converter.log_tracked_stats(line)


if __name__ == '__main__':
    main()
