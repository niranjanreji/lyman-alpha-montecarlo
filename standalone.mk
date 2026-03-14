# standalone.mk — build the lyman-alpha RT code without PLUTO.
#
# place this file (or symlink it) into any test/problem directory
# alongside rt_definitions.h, user_setup.cpp, and main.cpp.
#
# usage:
#   make -f standalone.mk            build with MPI + OpenMP
#   make -f standalone.mk serial     build without MPI
#   make -f standalone.mk clean

# path from this directory to the project root
ROOT     = ../..
SRC_DIR  = $(ROOT)/src

CXX      = mpicxx
CXXFLAGS = -O3 -std=c++17 -fopenmp -I. -I$(ROOT) -I$(SRC_DIR)
LDFLAGS  = -fopenmp -lstdc++ -lm

# MPI is on by default; `make serial` overrides
PARALLEL = TRUE

ifeq ($(PARALLEL), TRUE)
  CXXFLAGS += -DPARALLEL
endif

# local sources (in this directory)
LOCAL_SRCS = main.cpp user_setup.cpp

# shared RT sources
RT_SRCS    = $(SRC_DIR)/mc_grid.cpp      \
             $(SRC_DIR)/monte_carlo.cpp   \
             $(SRC_DIR)/physics.cpp       \
             $(SRC_DIR)/photons.cpp       \
             $(SRC_DIR)/voigt.cpp

OBJS       = $(LOCAL_SRCS:.cpp=.o) $(notdir $(RT_SRCS:.cpp=.o))
TARGET     = lyart

# ---- default target ----

all: $(TARGET)
	@mkdir -p output

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

# ---- pattern rules ----

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ---- serial build (no MPI) ----

serial:
	$(MAKE) -f standalone.mk PARALLEL=FALSE CXX=g++

# ---- clean ----

clean:
	rm -f $(OBJS) $(TARGET)
	@echo "cleaned."

.PHONY: all serial clean
