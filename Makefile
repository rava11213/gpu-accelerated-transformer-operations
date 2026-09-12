NVCC ?= nvcc
CUDA_ARCH ?= native
BUILD_DIR := build
TARGET := $(BUILD_DIR)/transformer_ops
NVCCFLAGS := -O3 -std=c++17 --use_fast_math -lineinfo

ifeq ($(CUDA_ARCH),native)
  ARCH_FLAGS := -arch=native
else
  ARCH_FLAGS := -arch=sm_$(CUDA_ARCH)
endif

.PHONY: all clean test sweep profile

all: $(TARGET)

$(TARGET): src/main.cu src/kernels.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) -Xcompiler=-Wall,-Wextra -o $@ src/main.cu

test: $(TARGET)
	$(TARGET) --op all --check --warmup 2 --iterations 10
	$(TARGET) --op all --check --m 127 --n 193 --k 61 --rows 17 --cols 31 --warmup 1 --iterations 3
	$(TARGET) --op matmul --variant naive --m 127 --n 193 --k 61 --check --warmup 1 --iterations 3

sweep: $(TARGET)
	bash benchmarks/sweep.sh $(TARGET)

profile: $(TARGET)
	@mkdir -p results
	nsys profile --stats=true --force-overwrite=true -o results/transformer_ops $(TARGET) --op all --iterations 100

clean:
	rm -rf $(BUILD_DIR) results
