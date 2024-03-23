import ctypes
from enum import Enum

clibrary = ctypes.CDLL("./ipl.so")
queue = ctypes.POINTER(ctypes.c_char)
image = ctypes.POINTER(ctypes.c_char)

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

# Load png file
clibrary.loadPNG.argtypes = [image, ctypes.c_char_p]
clibrary.loadPNG.restype = ctypes.c_void_p
clibrary.savePNG.argtypes = [image, ctypes.c_char_p]
clibrary.savePNG.restype = ctypes.c_void_p

#Bilateral Filter
clibrary.bilateral_filter.argtypes = [queue, image, image, ctypes.c_uint32, ctypes.c_double, ctypes.c_double, ctypes.c_int]
clibrary.bilateral_filter.restype = ctypes.c_void_p

#Box Filter
clibrary.box_filter.argtypes = [queue, image, image, ctypes.c_int, ctypes.c_int, ctypes.c_int]
clibrary.box_filter.restype = ctypes.c_void_p

Border = Enum('Border', ['repl', 'wrap', 'mirror', 'mirror_repl', 'default_val', 'const_val', 'transp'])

print("clibrary lista")

def createQueue():
	return clibrary.createQueue()

def createImage(cola, w, h):
	return clibrary.createImage(cola, w, h)

def loadBMP(imagen, ruta):
	clibrary.loadBMP(imagen, ruta)

def saveBMP(imagen, ruta):
	clibrary.saveBMP(imagen, ruta)

def loadPNG(imagen, ruta):
	clibrary.loadPNG(imagen, ruta)

def savePNG(imagen, ruta):
	clibrary.savePNG(imagen, ruta)

#Carlos

def bilateralFilter(cola, imgSrc, imgDst, kernel_size, sigma_intensity, sigma_distance, borde=Border.repl):
	clibrary.bilateral_filter(cola, imgSrc, imgDst, kernel_size, sigma_intensity, sigma_distance, borde.value - 1)


def boxFilter(cola, imgSrc, imgDst, w, h, borde=Border.repl):
	clibrary.box_filter(cola, imgSrc, imgDst, w, h, borde.value - 1)

#Angel
