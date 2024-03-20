import wrapper as ipl


print("Antes de todo")
cola = ipl.createQueue()
print("Cola creada")

imagen = ipl.createImage(cola, 466, 621)
print("Imagen creada")

ipl.loadPNG(imagen, b"carlos.png")
print("Imagen cargada")
ipl.savePNG(imagen, b"carlosWrapper.png")
print("Imagen guardada")

