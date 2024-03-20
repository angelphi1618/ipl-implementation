import wrapper as ipl


print("Antes de todo")
cola = ipl.createQueue()
print("Cola creada")

imagen = ipl.createImage(cola, 1200, 900)
print("Imagen creada")

ipl.loadBMP(imagen, b"lolita.bmp")
print("Imagen cargada")
ipl.saveBMP(imagen, b"lolitaWrapper.bmp")
print("Imagen guardada")

