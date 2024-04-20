import ctypes
import os
from enum import Enum

current_dir = os.path.dirname(__file__)
lib_path = os.path.join(current_dir, "ipl.so")
clibrary = ctypes.CDLL(lib_path)
image = ctypes.POINTER(ctypes.c_char)
kernelDataType = ctypes.POINTER(ctypes.c_float)

class Pixel(ctypes.Structure):
	_fields_ = [("R", ctypes.c_uint8),
			 	("G", ctypes.c_uint8),
				("B", ctypes.c_uint8),
				("A", ctypes.c_uint8)]
	
	def __init__(self, R=0, G=0, B=0, A=255):
		self.R = R
		self.G = G
		self.B = B
		self.A = A

	def set_color(self, R, G, B, A):
		self.R = R
		self.G = G
		self.B = B
		self.A = A

	def get_color(self):
		return self.R, self.G, self.B, self.A

# Image constructor
clibrary.createImage.argtypes = [ctypes.c_int, ctypes.c_int]
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
clibrary.bilateral_filter.argtypes = [image, image, ctypes.c_uint32, ctypes.c_double, ctypes.c_double, ctypes.c_int, Pixel]
clibrary.bilateral_filter.restype = ctypes.c_void_p

#Box Filter
clibrary.box_filter.argtypes = [image, image, ctypes.c_int, ctypes.c_int, ctypes.c_int, Pixel]
clibrary.box_filter.restype = ctypes.c_void_p

#Gaussian Filter
clibrary.gaussian_filter.argtypes = [image, image, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int, Pixel]
clibrary.gaussian_filter.restype = ctypes.c_void_p

#Convolution Filter
clibrary.filter_convolution.argtypes = [image, image,ctypes.c_int, ctypes.c_int, kernelDataType, ctypes.c_int, Pixel]
clibrary.filter_convolution.restype = ctypes.c_void_p

Border = Enum('Border', ['repl', 'wrap', 'mirror', 'mirror_repl', 'default_val', 'const_val', 'transp'])

print("clibrary lista")

def createQueue():
	return clibrary.createQueue()

def createImage( w, h):
	return clibrary.createImage( w, h)

def loadBMP(imagen, ruta):
	clibrary.loadBMP(imagen, ruta)

def saveBMP(imagen, ruta):
	clibrary.saveBMP(imagen, ruta)

def loadPNG(imagen, ruta):
	clibrary.loadPNG(imagen, ruta)

def savePNG(imagen, ruta):
	clibrary.savePNG(imagen, ruta)


def bilateralFilter( imgSrc, imgDst, kernel_size, sigma_intensity, sigma_distance, borde=Border.repl, defaultValue=Pixel()):
	clibrary.bilateral_filter( imgSrc, imgDst, kernel_size, sigma_intensity, sigma_distance, borde.value - 1, defaultValue)


def boxFilter( imgSrc, imgDst, w, h, borde=Border.repl, defaultValue=Pixel()):
	clibrary.box_filter( imgSrc, imgDst, w, h, borde.value - 1, defaultValue)

def gaussianFilter( imgSrc, imgDst, kernelSize, sigmaX, sigmaY, borde=Border.repl, defaultValue=Pixel()):
	clibrary.gaussian_filter( imgSrc, imgDst, kernelSize, sigmaX, sigmaY, borde.value - 1, defaultValue)

def convolutionFilter( imgSrc, imgDst, w, h, kernelData, borde=Border.repl, defaultValue=Pixel()):
	kernelData2 = (ctypes.c_float * len(kernelData))(*kernelData)
	clibrary.filter_convolution( imgSrc, imgDst, w, h, kernelData2, borde.value - 1, defaultValue)

# Median Filter
clibrary.medianFilter.argtypes = [image, image, ctypes.c_uint32, ctypes.c_int, Pixel]
clibrary.medianFilter.restype = ctypes.c_void_p

def medianFilter( src, dst, window_size, borde=Border.repl, defaultValue=Pixel()):
	clibrary.medianFilter( src, dst, window_size, borde.value - 1, defaultValue)

# RGB to Gray
clibrary.rgb_to_gray.argtypes = [image, image]
clibrary.rgb_to_gray.restype = ctypes.c_void_p

def rgbToGray( src, dst):
	clibrary.rgb_to_gray( src, dst)

clibrary.separable_filter.argtypes = [image, image, ctypes.c_int32, ctypes.c_int32 ,ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int, Pixel]
clibrary.separable_filter.restype = ctypes.c_void_p

def separableFilter( src, dst, kernel_x, kernel_y, borde=Border.repl, defaultValue=Pixel()):
	width = len(kernel_x)
	height = len(kernel_y)
	kernel_x_c = (ctypes.c_int32 * width)(* kernel_x)
	kernel_y_c = (ctypes.c_int32 * height)(* kernel_y)
	clibrary.separable_filter( src, dst, width, height, kernel_x_c, kernel_y_c, borde.value - 1, defaultValue)

clibrary.sobel_filter.argtypes = [image, image, ctypes.c_int32, ctypes.c_int, Pixel]
clibrary.sobel_filter.restype = ctypes.c_void_p

def sobelFilter( src, dst, kernel_size, borde=Border.repl, defaultValue=Pixel()):
	clibrary.sobel_filter( src, dst, kernel_size, borde.value - 1, defaultValue)