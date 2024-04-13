#!/bin/bash
cd ..
ROOTDIR=$(pwd)
BINDIR="$ROOTDIR/bin"

cd $BINDIR/tests


for dir in $BINDIR/tests/*; do
	if test -d $dir; then
		
		dir_name=$(basename "$dir")
		echo -n "Test de $dir_name... "
		
		cd "$dir"

		if ./$dir_name.out > /dev/null; then
			if cmp -s $BINDIR/tests/$dir_name/$dir_name.bmp $BINDIR/images/expected/$dir_name/$dir_name.bmp && cmp -s $BINDIR/tests/$dir_name/${dir_name}_roi.bmp $BINDIR/images/expected/$dir_name/${dir_name}_roi.bmp; then
				echo "PASS"
			else
				echo "FAILED"
			fi
		fi
	fi
done