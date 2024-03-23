import wrapper as ipl


print("Antes de todo")
cola = ipl.createQueue()
print("Cola creada")

imagen = ipl.createImage(cola,1200, 900)
print("Imagen creada")
imagenBilateral = ipl.createImage(cola, 1200, 900)
imagenBox = ipl.createImage(cola, 1200, 900)

ipl.loadBMP(imagen, b"lolita.bmp")
print("Imagen cargada")
ipl.saveBMP(imagenBilateral, b"lolitaCargada.bmp")
print("Imagen guardada")

ipl.bilateralFilter(cola, imagen, imagenBilateral, 9, 75, 75)
ipl.saveBMP(imagenBilateral, b"lolitaBilateral.bmp")
print("Imagen bilateral")

