# CUDA build settings
NVCC ?= nvcc
CXXFLAGS ?= -std=c++17 -O3
INCLUDES := -I.. -I../tensor_storage -I../tns_mat_mul

# Artifacts
TMM_OBJ := tmm_nd.o tmm_3d.o
TEST_BIN := cuda_tests

.PHONY: all clean

all: $(TMM_OBJ) $(TEST_BIN)

tmm_nd.o: tmm_nd.cu tmm_nd.cuh cuda_utils.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

tmm_3d.o: tmm_3d.cu tmm_3d.cuh tmm_nd.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(TEST_BIN): cuda_tests.cu $(TMM_OBJ)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) cuda_tests.cu $(TMM_OBJ) -o $@

clean:
	rm -f $(TMM_OBJ) $(TEST_BIN)
