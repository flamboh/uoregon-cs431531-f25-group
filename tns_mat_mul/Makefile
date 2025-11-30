# Makefile for compiling the C++ tensor multiplication tests using hipcc

# Define the compiler and flags
CXX = hipcc
CXXFLAGS = -std=c++17 -O3 -fopenmp

# Define the source file. It defaults to 'tmm_tests.cc' but can be overridden
# when calling 'make' (e.g., make SOURCE=new_test.cc).
SOURCE = core_tests.cc
TARGET = test_core

# Declare targets that are not files
.PHONY: all clean

# The default target that builds the executable
all: $(TARGET)

# Rule to compile the source file into the executable
$(TARGET): $(SOURCE)
	@echo "--- Compiling $(SOURCE) using hipcc ---"
	# The compilation command now uses the variable $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET) 
	@echo "--- Compilation complete. Executable: $(TARGET) ---"

# Rule to clean up the compiled executable
clean:
	@echo "--- Cleaning up compiled files ---"
	rm -f $(TARGET)
	@echo "--- Cleanup finished ---"