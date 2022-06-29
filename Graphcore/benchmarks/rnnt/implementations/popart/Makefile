BUILD_DIR = ./build

TARGET =

all: targets

debug: TARGET = debug
debug: targets

targets: $(BUILD_DIR) rnnt_loss exp_avg feat_augmentation

$(BUILD_DIR): 
	mkdir -p $(BUILD_DIR)

.PHONY: rnnt_loss
rnnt_loss:
	$(MAKE) $(TARGET) -C custom_ops/rnnt_loss

.PHONY: exp_avg
exp_avg:
	$(MAKE) -C custom_ops/exp_avg

.PHONY: feat_augmentation
feat_augmentation:
	$(MAKE) $(TARGET) -C custom_ops/feat_augmentation

.PHONY: clean
clean:
	rm -rf  $(BUILD_DIR)
