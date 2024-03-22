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


#Angel
# Median Filter
clibrary.medianFilter.argtypes = [queue, image, image, ctypes.c_uint32]
clibrary.medianFilter.restype = ctypes.c_void_p

def medianFilter(cola, src, dst, window_size):
	clibrary.medianFilter(cola, src, dst, window_size)

# RGB to Gray
clibrary.rgb_to_gray.argtypes = [queue, image, image]
clibrary.rgb_to_gray.restype = ctypes.c_void_p

def rgbToGray(cola, src, dst):
	clibrary.rgb_to_gray(cola, src, dst)

clibrary.separable_filter.argtypes = [queue, image, image,ctypes.c_int32, ctypes.c_int32 ,ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32) ]
clibrary.separable_filter.restype = ctypes.c_void_p

def separableFilter(cola, src, dst, kernel_x, kernel_y):
	width = len(kernel_x)
	height = len(kernel_y)
	kernel_x_c = (ctypes.c_int32 * width)(* kernel_x)
	kernel_y_c = (ctypes.c_int32 * height)(* kernel_y)
	clibrary.separable_filter(cola, src, dst, width, height, kernel_x_c, kernel_y_c)