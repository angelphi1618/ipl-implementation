import ctypes

clibrary = ctypes.CDLL("./sycl.so")
image = ctypes.POINTER(ctypes.c_char)
queue = ctypes.POINTER(ctypes.c_char)

# SYCL Queue create
clibrary.createQueue.argtypes = []
clibrary.createQueue.restype = queue

# Image constructor
clibrary.createImage.argtypes = [queue, ctypes.c_int, ctypes.c_int]
clibrary.createImage.restype = image
# Load bmp file
clibrary.loadBMP.argtypes = [image, ctypes.c_char_p]
clibrary.loadBMP.restype = ctypes.c_void_p
clibrary.saveBMP.argtypes = [image, ctypes.c_char_p]
clibrary.saveBMP.restype = ctypes.c_void_p

print("Antes de todo")
cola = clibrary.createQueue()
print("Cola creada")

imagen = clibrary.createImage(cola, 1200, 900)
print("Imagen creada")

clibrary.loadBMP(imagen, b"lolita.bmp")
print("Imagen cargada")
clibrary.saveBMP(imagen, b"lolitaWrapper.bmp")
print("Imagen guardada")

