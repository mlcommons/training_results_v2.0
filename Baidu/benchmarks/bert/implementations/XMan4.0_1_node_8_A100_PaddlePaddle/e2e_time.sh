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

set -e

get_timestamp() {
  local log_file="$1"     
  cat "$log_file" | grep -E 'run_start|run_stop' | awk '{print $5}' | awk -F',' '{print $1}' 
}

unset GREP_OPTIONS
for i in `seq 1 10`; do
  log_file="$1$i/worker.0"
  start_t=`get_timestamp "$log_file" | head -n 1`
  end_t=`get_timestamp "$log_file" | tail -n 1`
  time_cost=`python -c "print(($end_t - $start_t) / 60.0 / 1000.0)"`
  echo "$time_cost"
done
