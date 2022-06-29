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

import torch
import torchvision
import pickle
import argparse
import hashlib

class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, root, ann_file):
        super(COCODataset, self).__init__(root, ann_file)

    def pickle_annotations(self, pickle_output_file):
        with open(pickle_output_file, "wb") as f:
            pickle.dump({'coco': self.coco, 'ids': self.ids}, f)
        print("Wrote pickled annotations to %s" % (pickle_output_file))

    def md5_verify(self, pickle_output_file):
        original_md5 = '95b325a39d1614fa2225b981aa6ed40c'
        with open(pickle_output_file, 'rb') as file_to_check:
            data = file_to_check.read()
            md5_returned = hashlib.md5(data).hexdigest()
        if md5_returned == original_md5:
            return True
        else:
            print(f"pickled file md5 verification failed: {md5_returned}")
            return False

def main(args):
    coco = COCODataset(args.root, args.ann_file)
    coco.pickle_annotations(args.pickle_output_file)
    assert coco.md5_verify(args.pickle_output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pickle the training annotations file")
    parser.add_argument("--root", help="detectron2 dataset directory", default="/coco")
    parser.add_argument("--ann_file", help="coco training annotation file path",
                            default="/coco/annotations/instances_train2017.json")
    parser.add_argument("--pickle_output_file", help="pickled coco training annotation file output path",
                            default="/pkl_coco/instances_train2017.json.pickled")
    args=parser.parse_args()
    main(args)
