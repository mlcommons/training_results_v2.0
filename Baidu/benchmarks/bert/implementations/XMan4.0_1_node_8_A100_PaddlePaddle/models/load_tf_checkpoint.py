# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import pickle


def load_tf_checkpoint(input_path, output_path=None):
    import tensorflow as tf
    tf_path = os.path.abspath(input_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    assert len(set(names)) == len(names)
    assert len(names) == len(arrays)
    name_to_array = dict(zip(names, arrays))
    if output_path:
        print("Save TF numpy weight to {}".format(output_path))
        save_pickled_tf_checkpoint(name_to_array, output_path)
    return name_to_array


def load_pickled_tf_checkpoint(input_path):
    with open(input_path, "rb") as f:
        return pickle.load(f)


def save_pickled_tf_checkpoint(name_to_array, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(name_to_array, f, protocol=2)


if __name__ == "__main__":
    #input_path = "/data/zengjinle/dataset/bert_data/phase1/model.ckpt-28252"
    #output_path = "tf_ckpt.pickle"
    assert len(sys.argv) == 3
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    load_tf_checkpoint(input_path, output_path)
