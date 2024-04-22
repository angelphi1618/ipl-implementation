#!/bin/bash
cd ..

ROOTDIR=$(pwd)

SRCDIR="$ROOTDIR/benchmarks"
BINDIR="$ROOTDIR/bin/benchmarks"
CXX="icpx"
CXXFLAGS="-w -fsycl -g -O2 -L/usr/lib/gcc/x86_64-linux-gnu/11 -Rno-debug-disables-optimization"

# Creamos bin si no existe
mkdir -p "$BINDIR"

# Iteramos todos los tests
cd $SRCDIR
for dir in *; do
	if test -d "$dir"; then
		echo -n "Compilando benchmarks $dir... "
		rm -rf $BINDIR/$dir/$dir.o && rm -rf $BINDIR/$dir/$dir.out
		mkdir -p "$BINDIR/$dir"
		$CXX $CXXFLAGS -o "$BINDIR/$dir/$dir.out" "$SRCDIR/$dir/$dir.cpp" && echo "OK"
	fi
done


# #Compilar una vez
# icpx main.cpp -fsycl


#Compilar dos veces
#icpx -fPIC -c main.cpp -fsycl
#icpx -fPIC main.o -fsycl