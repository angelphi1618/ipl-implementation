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

.PHONY: all clean run tests install benchmarks

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(BINDIR)/%.o: $(SRCDIR)/%.cpp
	mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run:
	cd ./$(BINDIR) && ./main.out
wrapper $(BINDIR)/ipl.so:
	icpx -fPIC -shared -L/usr/lib/gcc/x86_64-linux-gnu/11 -fsycl $(SRCDIR)/wrapper/wrapper.cpp -o $(BINDIR)/ipl.so -w
runWrapper: $(BINDIR)/ipl.so
	cp $(BINDIR)/ipl.so $(SRCDIR)/wrapper && cd $(SRCDIR)/wrapper && python3 main.py
clean:
	rm -rf $(BINDIR)/*.o && rm -rf $(BINDIR)/*.out
tests:
	cd ./utils && sh compileTests.sh && sh runTests.sh
install: $(BINDIR)/ipl.so
	cd ./utils && ./installWrapper.sh
benchmarks:
	cd ./utils && sh compileBenchmarks.sh