# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")
#add_library(coco_cc_lib
#  maskApi.h
#  maskApi.c  
#  coco.h
#  coco.cc
#)

add_library(coco_cc_lib
  maskApi.h
  maskApi.c  
  cocoeval_base.h  
  cocoeval_functional.h
  cocoeval_base.cc  
  cocoeval_functional.cc
  coco.h
  coco.cc
)
# add_subdirectory(abseil-cpp)
# target_link_libraries(cocoeval_cc_lib /uslocal/lib/libabsl_hash.a  /usr/local/lib/libabsl_hashtablez_sampler.a  /usr/local/lib/libabsl_raw_hash_set.a  /usr/local/lib/libabsl_wyhash.a)
add_subdirectory("python")
# absl::algorithm absl::base absl::debugging absl::flat_hash_map absl::flags absl::memory absl::meta absl::numeric absl::random_random absl::strings absl::synchronization absl::time absl::utility)

