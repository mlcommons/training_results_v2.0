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

#!/usr/bin/env python3

import fileinput
import re
import json
import argparse
import glob

###########################################################################
#
# Generate a single run result score from an individual mlperf log file
#
###########################################################################


def sliding_window(elements, window_size):
    if len(elements) <= window_size:
        return elements
    padded_elements = elements.copy() + elements[:window_size].copy()
    res = []
    for i in range(len(elements)):
        res.append(padded_elements[i:i+window_size])
    return res


class FilePos:
    def __init__(self):
        self.state='before'
        self.benchmark=None
        self.global_batch_size=None
        self.eval_epoch=None
        self.results = []
        self.default_epoch=1e9
        self.default_time=999999.9
        self.default_throughput=0.0

    def run_start(self, json_struct):
        self.start_time = json_struct['time_ms']
        if not (self.state == 'before' or self.state == 'run_stop'):
            print(self.benchmark, self.global_batch_size, self.default_time, self.default_throughput)
            self.results.append((self.default_epoch, self.default_time, self.default_throughput))
        self.state = 'run_start'

    def run_stop(self, json_struct):
        stop_time = json_struct['time_ms']
        result_success = json_struct['metadata']['status']
        throughput = json_struct['metadata']['throughput']
        time_in_seconds = (float(stop_time)-float(self.start_time))/1000.0
        epochs_to_stop = self.eval_epoch
        #throughput = float(epochs_to_stop)/time_in_seconds
        if result_success != 'success' or self.state != 'run_start':
            time_in_seconds += self.default_time
        print(self.benchmark, self.global_batch_size,
              time_in_seconds,
              epochs_to_stop,
              throughput)
        self.results.append((epochs_to_stop, time_in_seconds, throughput))
        self.state = 'run_stop'
        
        self.benchmark = None
        self.global_batch_size = None
        self.eval_epoch = None

    def file_end(self):
        if self.state == 'before':
            # no job ever started, so no time
            print()
        elif self.state != 'run_stop':
            # job started but never converged and never_stopped
            print(self.benchmark, self.global_batch_size, self.default_time, self.default_epoch, self.default_throughput)
            self.results.append((self.default_epoch, self.default_time, self.default_throughput))
        self.state = 'before'

    def dispatch_json(self, json_struct):
        if json_struct['key'] == 'run_start':
            self.run_start(json_struct)
        elif json_struct['key'] == 'run_stop':
            self.run_stop(json_struct)
        elif json_struct['key'] == 'global_batch_size':
            self.global_batch_size = json_struct['value']
        elif json_struct['key'] == 'submission_benchmark':
            self.benchmark = json_struct['value']
        elif json_struct['key'] == 'eval_accuracy':
            self.eval_epoch = json_struct['metadata']['epoch_num']

    def process_line(self, line):
        json_match = re.match(r'.*:::MLLOG\s+' + r'(.*)', line)
        if json_match:
            json_string = json_match.group(1)
            json_struct = json.loads(json_string)
            self.dispatch_json(json_struct)

    def gather_results(self, target_epoch):

        # num of runs
        print(f"Total number of runs: {len(self.results)}")

        # num of converged runs
        filtered = list(filter(lambda r: r[1] < self.default_time, self.results))
        print(f"Total number of converged runs: {len(filtered)}")


        filtered = list(filter(lambda r: r[0] <= target_epoch, self.results))
        print(f"Total number of converged runs at epoch {target_epoch}: {len(filtered)}")

        # time to converge at target epoch
        avg_time = sum(r[1] for r in filtered) / len(filtered)
        print(f"Averge time to converge at epoch {target_epoch}: {avg_time:.2f} seconds")

        # avg throughput
        avg_throughput = sum(float(r[2]) for r in filtered) / len(filtered)
        print(f"Averge throughput to converge at epoch {target_epoch}: {avg_throughput:.2f} imgs/s")

        # olympic score
        # TODO: add wrap
        window_size = 5
        windows = sliding_window([r[1] for r in self.results], window_size)
        windows = [sorted(w)[1:-1] for w in windows]
        scores = sorted([sum(w)/(window_size - 2) for w in windows])
        #print(scores)
        mid = len(scores) // 2
        print(f"Olympic score(median of scores): {(scores[mid] + scores[~mid]) / 2:.2f} seconds")

def run(log_files, target_epoch):
    file_pos = FilePos()
    for line in fileinput.input(files=log_files, openhook=fileinput.hook_encoded("utf-8", errors="ignore")):
        file_pos.process_line(line)
    file_pos.file_end()

    file_pos.gather_results(target_epoch)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Gather convergence info")
    parser.add_argument(
        "--log-dir", "-l",
        default="xxx",
        help="dir of log files",
        type=str,
    )
    parser.add_argument("--target_epoch", "-t", type=int, default=14, help="target epoch to converge")

    args = parser.parse_args()

    print(args.log_dir)
    log_files = sorted(glob.glob(args.log_dir + "/*.out"))
    print(log_files)
    target_epoch = args.target_epoch
    run(log_files, target_epoch)

    elements = [1, 2, 3, 4, 5, 6]
    window_size = 2
    sliding_window(elements, window_size)

#sample usage: python3 parser_log.py -l /lustre/fsw/joc/yudong/mlperf-maskrcnn/output/single_node_22_03/ -t 14
