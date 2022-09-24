set -ex

ROOT_DIR=`dirname "$0"`

function build_nvcc_obj() {
    nvcc -c $1 \
        -O3 \
	    -Xcompiler="-fPIC" \
        -Xcompiler="-O3" \
        -Xcompiler="-DVERSION_GE_1_1" \
        -Xcompiler="-DVERSION_GE_1_3" \
        -Xcompiler="-DDVERSION_GE_1_5" \
	    -gencode arch=compute_80,code=sm_80 \
	    -U__CUDA_NO_HALF_OPERATORS__ \
	    -U__CUDA_NO_HALF_CONVERSIONS__ \
        -I${ROOT_DIR}/src \
	    -I${APEX_DIR} \
	    --expt-relaxed-constexpr \
	    --expt-extended-lambda \
	    --use_fast_math \
	    -DVERSION_GE_1_1 \
	    -DVERSION_GE_1_3 \
	    -DVERSION_GE_1_5
}

rm -rf *.so
rm -rf *.o

build_nvcc_obj ${ROOT_DIR}/src/fmha_fprop_fp16_kernel.sm80.cu 
build_nvcc_obj ${ROOT_DIR}/src/fmha_dgrad_fp16_kernel_loop.sm80.cu 

FMHA_FILE=${ROOT_DIR}/fmha_api
rm -rf ${FMHA_FILE}.cu
cp ${FMHA_FILE}.cpp ${FMHA_FILE}.cu
build_nvcc_obj ${FMHA_FILE}.cu || rm -rf ${FMHA_FILE}.cu

rm -rf ${FMHA_FILE}.cu
nvcc -shared -Xcompiler="-fPIC" -o libflash_attn.so *.o 
rm -rf *.o
INSTALL_DIR=/usr/local/lib
rm -rf "$INSTALL_DIR/libflash_attn.so"
cp libflash_attn.so "$INSTALL_DIR/"
ldconfig

