load(":build_defs.bzl", "cuda_header_library")

licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_nvcc",
    values = {
        "define": "using_cuda_nvcc=true",
    },
)

config_setting(
    name = "using_clang",
    values = {
        "define": "using_cuda_clang=true",
    },
)

# Equivalent to using_clang && -c opt.
config_setting(
    name = "using_clang_opt",
    values = {
        "define": "using_cuda_clang=true",
        "compilation_mode": "opt",
    },
)

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
)

config_setting(
    name = "freebsd",
    values = {"cpu": "freebsd"},
)

cuda_header_library(
    name = "cuda_headers",
    hdrs = [
        "cuda/cuda_config.h",
        ":cuda-include"
    ],
    include_prefix = "third_party/gpus",
    includes = [
        ".",  # required to include cuda/cuda/cuda_config.h as cuda/config.h
        "cuda/include",
    ],
)

cc_library(
    name = "cudart_static",
    srcs = ["cuda/lib/libcudart_static.a"],
    linkopts = select({
        ":freebsd": [],
        "//conditions:default": ["-ldl"],
    }) + [
        "-lpthread",
        "-lrt",
    ],
)

cc_library(
    name = "cuda_driver",
    srcs = ["cuda/lib/libcuda.so"],
)

cc_library(
    name = "cudart",
    srcs = ["cuda/lib/libcudart.so"],
    data = ["cuda/lib/libcudart.so"],
    linkstatic = 1,
)

cuda_header_library(
    name = "cublas_headers",
    hdrs = [":cublas-include"],
    include_prefix = "third_party/gpus/cuda/include",
    strip_include_prefix = "cublas/include",
    deps = [":cuda_headers"],
    includes = ["cublas/include"],
)

cc_library(
    name = "cublas",
    srcs = ["cuda/lib/libcublas.so"],
    data = ["cuda/lib/libcublas.so"],
    linkstatic = 1,
)

cc_library(
    name = "cusolver",
    srcs = ["cuda/lib/libcusolver.so"],
    data = ["cuda/lib/libcusolver.so"],
    linkopts = ["-lgomp"],
    linkstatic = 1,
)

cc_library(
    name = "cudnn",
    srcs = ["cuda/lib/libcudnn.so"],
    data = ["cuda/lib/libcudnn.so"],
    linkstatic = 1,
)

cc_library(
    name = "cudnn_header",
    hdrs = [":cudnn-include"],
    include_prefix = "third_party/gpus/cudnn",
    strip_include_prefix = "cudnn/include",
    deps = [":cuda_headers"],
)

cc_library(
    name = "cufft",
    srcs = ["cuda/lib/libcufft.so"],
    data = ["cuda/lib/libcufft.so"],
    linkstatic = 1,
)

cc_library(
    name = "curand",
    srcs = ["cuda/lib/libcurand.so"],
    data = ["cuda/lib/libcurand.so"],
    linkstatic = 1,
)

cc_library(
    name = "cuda",
    deps = [
        ":cublas",
        ":cuda_headers",
        ":cudart",
        ":cudnn",
        ":cufft",
        ":curand",
    ],
)

cuda_header_library(
    name = "cupti_headers",
    hdrs = [":cuda-extras"],
    include_prefix="third_party/gpus",
    includes = ["cuda/extras/CUPTI/include/"],
    deps = [":cuda_headers"],
)

cc_library(
    name = "cupti_dsos",
    data = ["cuda/lib/libcupti.so"],
)

cc_library(
    name = "cusparse",
    srcs = ["cuda/lib/libcusparse.so"],
    data = ["cuda/lib/libcusparse.so"],
    linkopts = ["-lgomp"],
    linkstatic = 1,
)

cc_library(
    name = "libdevice_root",
    data = [":cuda-nvvm"],
)


filegroup(name="cuda-include")
filegroup(name="cublas-include")
filegroup(name="cudnn-include")

