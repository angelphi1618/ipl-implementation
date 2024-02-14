# Compile each source file into object files
#dpcpp -fsycl -c main.cpp -o main.o
#dpcpp -fsycl -c ./allocators/helper.cpp -o helper.o

# Link the object files together to create the executable
#dpcpp main.o helper.o -o main

CXX = icpx
CXXFLAGS = -w -fsycl
SRCDIR = src
BINDIR = bin
SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(patsubst $(SRCDIR)/%.cpp,$(BINDIR)/%.o,$(SRCS))
TARGET = $(BINDIR)/main.out

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(BINDIR)/%.o: $(SRCDIR)/%.cpp
	mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run:
	cd ./$(BINDIR) && ./main.out

clean:
	rm -rf $(BINDIR)/*.o && rm -rf $(BINDIR)/*.out
