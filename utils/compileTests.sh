#!/bin/bash
cd ..

ROOTDIR=$(pwd)
SRCDIR="$ROOTDIR/src/tests"
BINDIR="$ROOTDIR/bin/tests"
CXX="icpx"
CXXFLAGS="-w -O3 -fsycl"

# Creamos bin si no existe
mkdir -p "$BINDIR"

# Iteramos todos los tests
cd $SRCDIR
for dir in *; do
	if test -d "$dir"; then
		echo -n "Compilando test $dir... "
		rm -rf $BINDIR/$dir/*.o && rm -rf $BINDIR/$dir/*.out
		mkdir -p "$BINDIR/$dir"
		$CXX $CXXFLAGS "$SRCDIR/$dir/$dir.cpp" -o "$BINDIR/$dir/$dir.out" && echo "OK"
	fi
done


# #Compilar una vez
# icpx main.cpp -fsycl


#Compilar dos veces
#icpx -fPIC -c main.cpp -fsycl
#icpx -fPIC main.o -fsycl