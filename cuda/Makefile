# CUDA build settings
NVCC ?= nvcc
CXXFLAGS ?= -std=c++17 -O3
INCLUDES := -I.. -I../tensor_storage -I../tns_mat_mul

# Artifacts
TMM_OBJ := tmm_3d.o
TEST_BIN := cuda_tests

.PHONY: all clean

all: $(TMM_OBJ) $(TEST_BIN)

$(TMM_OBJ): tmm_3d.cu tmm_3d.cuh cuda_utils.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(TEST_BIN): cuda_tests.cu tmm_3d.o
	$(NVCC) $(CXXFLAGS) $(INCLUDES) cuda_tests.cu tmm_3d.o -o $@

clean:
	rm -f $(TMM_OBJ) $(TEST_BIN)
