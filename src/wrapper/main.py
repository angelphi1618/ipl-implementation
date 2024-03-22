import wrapper as ipl


print("Antes de todo")
cola = ipl.createQueue()
print("Cola creada")

imagen = ipl.createImage(cola, 466, 621)
print("Imagen creada")

ipl.loadPNG(imagen, b"carlos.png")
print("Imagen cargada")
ipl.savePNG(imagen, b"../../bin/images/wrapper/carlosWrapper.png")
print("Imagen guardada")

imagen2 = ipl.createImage(cola, 512, 512)
imagen3 = ipl.createImage(cola, 512, 512)
imagen4 = ipl.createImage(cola, 512, 512)

ipl.loadBMP(imagen2, b"../../bin/images/lena.bmp")
ipl.medianFilter(cola, imagen2, imagen3, 3)

ipl.saveBMP(imagen3, b"../../bin/images/wrapper/lenaMediana.bmp")

ipl.rgbToGray(cola, imagen2, imagen3)
ipl.saveBMP(imagen3, b"../../bin/images/wrapper/lenaGray.bmp")


ipl.separableFilter(cola, imagen2, imagen3, [1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,])
ipl.saveBMP(imagen3, b"../../bin/images/wrapper/lenaSeparada.bmp")
