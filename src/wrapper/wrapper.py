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

class border(Enum):
	repl = 0,
	wrap = 1,
	mirror = 2,
	mirror_repl = 3,
	default_val = 4, 
	const_val = 5,
	transp = 6

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

def bilateralFilter(cola, imgSrc, imgDst, kernel_size, sigma_intensity, sigma_distance, border = 0):
	clibrary.bilateral_filter(cola, imgSrc, imgDst, kernel_size, sigma_intensity, sigma_distance, border)




#Angel
