licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_hipcc",
    values = {
        "define": "using_rocm_hipcc=true",
    },
)

cc_library(
    name = "rocm_headers",
    hdrs = [
        "rocm/rocm_config.h",
        
    ],
    includes = [
        ".",
        "rocm/include",
        "rocm/include/rocrand",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hip",
    srcs = ["rocm/lib/libhip.so"],
    data = ["rocm/lib/libhip.so"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocblas",
    srcs = ["rocm/lib/librocblas.so"],
    data = ["rocm/lib/librocblas.so"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocfft",
    srcs = ["rocm/lib/librocfft.so"],
    data = ["rocm/lib/librocfft.so"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hiprand",
    srcs = ["rocm/lib/libhiprand.so"],
    data = ["rocm/lib/libhiprand.so"],
    includes = [
        ".",
        "rocm/include",
        "rocm/include/rocrand",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "miopen",
    srcs = ["rocm/lib/libmiopen.so"],
    data = ["rocm/lib/libmiopen.so"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_headers",
        ":hip",
        ":rocblas",
        ":rocfft",
        ":hiprand",
        ":miopen",
    ],
)


