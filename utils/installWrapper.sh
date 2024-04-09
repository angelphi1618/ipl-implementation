#!/bin/bash

INSTALL_DIR="/usr/bin/ipl"

# vemos si /usr/bin/ipl existe && lo borramos
test -d /usr/bin/ipl && rm -rf /usr/bin/ipl

mkdir $INSTALL_DIR

# Vemos si /usr/bin/ipl está en el PYTHONPATH
if ! [[ ":$PYTHONPATH:" == *":$INSTALL_DIR:"* ]]; then
	PYTHONPATH=$PYTHONPATH:$INSTALL_DIR
fi

# Creamos un directorio con el paquete ipl
mkdir $INSTALL_DIR/ipl
cp ../bin/ipl.so $INSTALL_DIR/ipl/

# Copiamos el wrapper al paquete ipl
cp ../src/wrapper/wrapper.py $INSTALL_DIR/ipl/__init__.py

echo "Instalada correctamente en $INSTALL_DIR. Agrega esta ruta al extraPaths de tu IDE."