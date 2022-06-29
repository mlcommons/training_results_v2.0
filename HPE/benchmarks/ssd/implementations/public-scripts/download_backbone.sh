#!/usr/bin/env bash

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

DOWNLOAD_LINK='https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'
SHA512='15c9f0bc1c8d64750712f86ffaded3b0bc6a87e77a395dcda3013d8af65b7ebf3ca1c24dd3aae60c0d83e510b4d27731f0526b6f9392c0a85ffc18e5fecd8a13'
FILENAME='resnext50_32x4d-7cdf4587.pth'

wget -c $DOWNLOAD_LINK
echo "${SHA512}  ./${FILENAME}" | sha512sum -c
