#!/bin/bash

# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

: "${DOWNLOAD_PATH:=/datasets/downloads/coco2017}"
: "${OUTPUT_PATH:=/datasets/coco2017}"

while [ "$1" != "" ]; do
    case $1 in
        -d | --download-path )       shift
                                     DOWNLOAD_PATH=$1
                                     ;;
        -o | --output-path  )        shift
                                     OUTPUT_PATH=$1
                                     ;;
    esac
    shift
done

mkdir -p $DOWNLOAD_PATH
cd $DOWNLOAD_PATH
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "cced6f7f71b7629ddf16f17bbcfab6b2  ./train2017.zip"                | md5sum -c
echo "442b8da7639aecaf257c1dceb8ba8c80  ./val2017.zip"                  | md5sum -c
echo "f4bbac642086de4f52a3fdda2de5fa2c  ./annotations_trainval2017.zip" | md5sum -c

mkdir -p $OUTPUT_PATH
unzip train2017.zip -d $OUTPUT_PATH
unzip val2017.zip -d $OUTPUT_PATH
unzip annotations_trainval2017.zip -d $OUTPUT_PATH
