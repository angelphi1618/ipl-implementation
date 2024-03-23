import wrapper as ipl


print("Antes de todo")
cola = ipl.createQueue()
print("Cola creada")

imagen = ipl.createImage(cola,1200, 900)
print("Imagen creada")
imagenBilateral = ipl.createImage(cola, 1200, 900)
imagenBox = ipl.createImage(cola, 1200, 900)
imagenGaussian = ipl.createImage(cola, 1200, 900)
imagenConvolution = ipl.createImage(cola, 1200, 900)

ipl.loadBMP(imagen, b"lolita.bmp")
print("Imagen cargada")
ipl.saveBMP(imagen, b"lolitaCargada.bmp")
print("Imagen guardada")

ipl.bilateralFilter(cola, imagen, imagenBilateral, 9, 75, 75, ipl.Border.repl)
ipl.saveBMP(imagenBilateral, b"lolitaBilateral.bmp")
print("Imagen bilateral")

ipl.boxFilter(cola, imagen, imagenBox, 30, 30, ipl.Border.const_val)
ipl.saveBMP(imagenBox, b"lolitaBox.bmp")
print("Imagen box")

ipl.gaussianFilter(cola, imagen, imagenGaussian, 9, 75, 75, ipl.Border.const_val)
ipl.saveBMP(imagenGaussian, b"lolitaGaussiana.bmp")
print("imagen gaussian")


ipl.convolutionFilter(cola, imagen, imagenConvolution, 3, 3, [1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0], ipl.Border.repl)
ipl.saveBMP(imagenConvolution, b"lolitaConvolucion.bmp")
print("imagen Convolution")