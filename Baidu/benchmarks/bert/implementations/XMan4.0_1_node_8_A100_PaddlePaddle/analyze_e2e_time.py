# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import json


def get_mllog_json(line):
    prefix = ":::MLLOG"
    if not line.startswith(prefix):
        return None

    line = line[len(prefix):].strip()
    return json.loads(line)


def readlines(file_path):
    with open(file_path, "r") as f:
        return list(f.readlines())


def analyze_one_file(file_path, gbs):
    lines = readlines(file_path)
    run_start_t = None
    run_end_t = None
    train_samples = None
    success = None
    for line in lines:
        if "run_start" not in line and "run_stop" not in line and "train_samples" not in line:
            continue

        log_json = get_mllog_json(line)
        if log_json is None or "key" not in log_json:
            continue

        key = log_json["key"]
        if key == "run_start":
            run_start_t = log_json["time_ms"]
        elif key == "train_samples":
            train_samples = log_json["value"]
        elif key == "run_stop":
            run_end_t = log_json["time_ms"]
            success = 1 if log_json["metadata"]["status"] == "success" else 0
            break

    assert run_start_t is not None and run_end_t is not None or success is not None and train_samples is not None, file_path
    assert train_samples % gbs == 0
    return (run_end_t - run_start_t
            ) / 60.0 / 1000.0, success, train_samples, train_samples / gbs


def avg_without_min_max(times):
    min_t = min(times)
    max_t = max(times)
    min_idx = [i for i, t in enumerate(times) if t == min_t][0]
    max_idx = [i for i, t in enumerate(times) if t == max_t][0]
    times = [t for i, t in enumerate(times) if i != min_idx and i != max_idx]
    return sum(times) / len(times), min_idx, max_idx


class TablePrinter(object):
    def __init__(self, headers):
        self.headers = list([str(h) for h in headers])
        self.rows = []
        self.max_lens = [len(h) for h in self.headers]

    def add_row(self, row):
        assert len(row) == len(self.headers)
        row = [str(item) for item in row]
        self.max_lens = [
            max(length, len(row[i])) for i, length in enumerate(self.max_lens)
        ]
        self.rows.append(row)

    def _aligned_str(self, s, length):
        return s + (' ' * (length - len(s)))

    def _aligned_row(self, row, separator='  '):
        return separator.join([
            self._aligned_str(s, self.max_lens[i]) for i, s in enumerate(row)
        ])

    def print_table(self):
        print(self._aligned_row(self.headers))
        for row in self.rows:
            print(self._aligned_row(row))


def analyze(file_pattern, file_num, gbs, min_train_samples, win_size=10):
    results = []
    for file_idx in range(file_num):
        i = file_idx + 1
        file_path = file_pattern.format(i)
        ret = [i] + list(analyze_one_file(file_path, gbs))
        results.append(ret)

    table1 = TablePrinter([
        'FileIdx',
        'Success',
        'TrainSamples',
        'TrainingSteps',
        'Time(min)',
        'ValidTime(min)',
        'Throughput(s/step)',
    ])
    for file_idx, t, success, samples, step in results:
        table1.add_row([
            file_idx,
            success,
            samples,
            step,
            t,
            t if success else float('inf'),
            t / step * 60.0,
        ])
    table1.print_table()

    n = len(results)
    win_results = []
    for i in range(n - win_size + 1):
        times = [
            results[i + j][1] if results[i + j][2] else float('inf')
            for j in range(win_size)
        ]
        avg_time, min_idx, max_idx = avg_without_min_max(times)
        samples = [
            float(results[i + j][3]) for j in range(win_size)
            if j != min_idx and j != max_idx
        ]
        avg_samples = sum(samples) / len(samples)
        start_idx = results[i][0]
        end_idx = results[i + win_size - 1][0]
        win_results.append((start_idx, end_idx, avg_samples, avg_time))

    print('-' * 120)
    table2 = TablePrinter([
        'StartFileIdx',
        'EndFileIdx',
        'AvgSamples',
        'AvgTime(min)',
        'ValidAvgTime(min)',
    ])
    for start_idx, end_idx, avg_samples, avg_time in win_results:
        valid_avg_time = avg_time if avg_samples >= min_train_samples else float(
            'inf')
        table2.add_row(
            [start_idx, end_idx, avg_samples, avg_time, valid_avg_time])
    table2.print_table()


def get_or_default(idx, default, type=None):
    args = sys.argv
    value = args[idx] if idx < len(args) else default
    return type(value) if type is not None else value


if __name__ == "__main__":
    nargv = len(sys.argv)
    assert nargv >= 2 and nargv <= 5, "Usage: {} {} <file_path_pattern> [<file_num>] [<global_batch_size>] [<min_train_samples>]".format(
        sys.executable, sys.argv[0])

    file_pattern = sys.argv[1]
    file_num = get_or_default(2, 1, int)
    gbs = get_or_default(3, 8 * 56, int)
    min_train_samples = get_or_default(4, 2621696.0 / 1.0387858550359907, float)
    analyze(file_pattern, file_num, gbs, min_train_samples)
