CXX ?= g++
CXXFLAGS = -std=c++14 -g -fPIC -DONNX_NAMESPACE=onnx -Wall -Wno-sign-compare -Wno-switch -I./
LDLIBS = -lpopart -lpoplar -lpopops -lpoputil -lpoprand
INCLUDES = -I./

BUILD_DIR = ../../build

RNNT_LOSS_SOURCES = rnnt_loss.cpp rnnt_utils.cpp ipu_utils.cpp
RNNT_LOSS_TARGET = $(BUILD_DIR)/rnnt_loss.so

SPARSE_LOGSOFTMAX_SOURCES = sparse_logsoftmax_op.cpp rnnt_utils.cpp ipu_utils.cpp logsoftmax.cpp
SPARSE_LOGSOFTMAX_TARGET = $(BUILD_DIR)/sparse_logsoftmax.so

JOINT_NET_SOURCES = get_compact_log_probs_op.cpp add.cpp matmul.cpp relu.cpp sparse_log_softmax.cpp rnnt_utils.cpp ipu_utils.cpp logsoftmax.cpp random_mask.cpp dropout.cpp
JOINT_NET_TARGET = $(BUILD_DIR)/custom_joint_net.so

DROPOUT_SOURCES = dropout_op.cpp dropout.cpp rnnt_utils.cpp ipu_utils.cpp random_mask.cpp
DROPOUT_TARGET = $(BUILD_DIR)/dropout.so

all: release

debug: CXXFLAGS += -DDEBUG -O0
debug: targets

release: CXXFLAGS += -O3
release: targets

targets: $(RNNT_LOSS_TARGET) $(SPARSE_LOGSOFTMAX_TARGET) $(JOINT_NET_TARGET) $(DROPOUT_TARGET)

$(RNNT_LOSS_TARGET): $(RNNT_LOSS_SOURCES) $(MAKEFILE_LIST)
	$(CXX) $(RNNT_LOSS_SOURCES)  $(LDLIBS) $(CXXFLAGS)  -shared -o $(RNNT_LOSS_TARGET)

$(SPARSE_LOGSOFTMAX_TARGET): $(SPARSE_LOGSOFTMAX_SOURCES) $(MAKEFILE_LIST)
	$(CXX) $(SPARSE_LOGSOFTMAX_SOURCES) $(LDLIBS) $(CXXFLAGS)  -shared -o $(SPARSE_LOGSOFTMAX_TARGET)

NO_RECOMPUTE_FC_IN ?= 0
NO_RECOMPUTE_FC_OUT ?= 0
OUTPUT_DROPOUT_MASK ?= 0
EXTRA_FLAGS = -DNO_RECOMPUTE_FC_IN=$(NO_RECOMPUTE_FC_IN) -DNO_RECOMPUTE_FC_OUT=$(NO_RECOMPUTE_FC_OUT) -DOUTPUT_DROPOUT_MASK=$(OUTPUT_DROPOUT_MASK)
$(JOINT_NET_TARGET): $(JOINT_NET_SOURCES) $(MAKEFILE_LIST)
	$(CXX) $(JOINT_NET_SOURCES) $(CXXFLAGS) $(EXTRA_FLAGS) $(LDLIBS) $(INCLUDES) -shared -o $(JOINT_NET_TARGET)

$(DROPOUT_TARGET): $(DROPOUT_SOURCES) $(MAKEFILE_LIST)
	$(CXX) $(DROPOUT_SOURCES) $(LDLIBS) $(CXXFLAGS) $(EXTRA_FLAGS) -shared -o $(DROPOUT_TARGET)

.PHONY: clean
clean:
	rm -rf  $(RNNT_LOSS_TARGET) $(SPARSE_LOGSOFTMAX_TARGET) $(DROPOUT_TARGET)
