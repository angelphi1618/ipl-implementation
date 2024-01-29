# Compile each source file into object files
dpcpp -fsycl -c main.cpp -o main.o
dpcpp -fsycl -c ./allocators/helper.cpp -o helper.o

# Link the object files together to create the executable
dpcpp main.o helper.o -o main
