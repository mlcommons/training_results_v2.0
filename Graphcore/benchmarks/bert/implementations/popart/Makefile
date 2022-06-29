all: custom_ops/build/plugin_version custom_ops.so

BUILD_DIR := ./custom_ops/build
SRC_DIRS := ./custom_ops

SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' \
          ! -name 'compile_time_version.cpp' \
          ! -name 'dropout_with_switch.cpp'  \
		  ! -name 'detach.cpp' \
		  ! -name 'utils.cpp'  \
		  ! -name 'fused_replicated_allreduce.cpp' \
		  ! -name 'sparse_accumulate.cpp' \
		  ! -name 'tied_gather.cpp' \
		  ! -name 'fp8_stash.cpp' \
		  ! -name 'fp8_restore.cpp' \
		  ! -name 'fp8checkpoint_pattern.cpp' )

# String substitution for every C/C++ file.
# As an example, hello.cpp turns into ./build/hello.cpp.o
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

# String substitution (suffix version without %).
# As an example, ./build/hello.cpp.o turns into ./build/hello.cpp.d
DEPS := $(OBJS:.o=.d)

custom_ops/build/plugin_version: custom_ops/compile_time_version.cpp
	mkdir -p $(dir $@)
	g++ custom_ops/compile_time_version.cpp -o custom_ops/build/plugin_version -l poplar

# The final build step.
custom_ops.so: custom_ops/build/plugin_version $(OBJS)
	$(CC) $(OBJS) \
		-shared -lpopart -lpoplar -lpoplin -lpopnn -lpopops -lpoputil -lpoprand -lpopfloat \
		-o $@ 

# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX)  -MMD -MP -std=c++14 -fPIC -Wno-deprecated-declarations\
			-DSTATIC_VERSION=\"${shell ./custom_ops/build/plugin_version}\" \
			-DONNX_NAMESPACE=onnx \
			-c $< -o $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) custom_ops.so

# Include the .d makefiles. The - at the front suppresses the errors of missing
# Makefiles. Initially, all the .d files will be missing, and we don't want those
# errors to show up.
-include $(DEPS)