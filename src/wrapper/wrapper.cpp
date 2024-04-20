#include "../image.h"
#include "../allocators/device_usm_allocator_t.h"
#include "../image_persistance/bmp_persistance.h"
#include "../image_persistance/png_persistance.h"
#include "../algorithms/bilateral_filter.h"
#include "../algorithms/box_filter.h"
#include "../algorithms/gaussian_filter.h"
#include "../algorithms/filter_convolution.h"
#include "../border_generator/border_types.h"
#include "../algorithms/grayscale.h"
#include "../algorithms/median_filter.h"
#include "../algorithms/separable_filter.h"
#include "../algorithms/sobel_filter.h"

#include "singletoneQueue.h"

#include <cstdint>


extern "C" {

	using Image = ::image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>;
	using Pixel = ::pixel<uint8_t>;

	enum bordes{
		repl,
		mirror,
		constante
	};

	Image* createImage(int width, int height) {
		sycl::range<2> size(width, height);

		std::cout << "w=" << width << ", h=" << height << std::endl;

		std::cout << "pidiendo cola" << std::endl;

		sycl::queue* queue = SingletoneQueue::getInstance();

		std::cout << "cola recibida" << std::endl;

		return new Image(*queue, size);
	}

	void loadBMP(Image* img, char* filename) {
		std::string str(filename);
        std::cout << str << std::endl;
		bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imageLoader(*img);
		std::cout << "hola desde loadBMP"<< std::endl;
		imageLoader.loadImage(str);
	}

	void saveBMP(Image* img, char* filename) {
		std::string str(filename);
		bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(*img, str);
	}


	void loadPNG(Image* img, char* filename) {
		std::string str(filename);
        std::cout << str << std::endl;
		png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imageLoader(*img);
		std::cout << "hola desde loadBMP"<< std::endl;
		imageLoader.loadImage(str);
	}

	void savePNG(Image* img, char* filename) {
		std::string str(filename);
		png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(*img, str);
	}

	// void sobelFilter(sycl::queue* q, Image* src, Image* dst, int kernel_size) {
	// 	sobel_filter_spec kernel_spec(kernel_size);
	// 	sobel_filter(*q, *src, *dst, kernel_spec, border_types::repl).wait();
	// }

	void medianFilter(Image* src, Image* dst, unsigned int window_size, int border, Pixel default_value) {
		sycl::queue* q = SingletoneQueue::getInstance();

		median_spec window_spec = {window_size};
		median_filter(*q, *src, *dst, window_spec, static_cast<border_types>(border), default_value).wait();
	}

	void rgb_to_gray(Image* src, Image*dst) {
		sycl::queue* q = SingletoneQueue::getInstance();
		rgb_to_gray(*q, *src, *dst).wait();
	}

	void separable_filter(Image* src, Image* dst, int width, int height, int kernel_x[], int kernel_y[],
							int border, Pixel default_value) {
		sycl::queue* q = SingletoneQueue::getInstance();
		separable_spec<int> spec{sycl::range<2>(height, width), kernel_x, kernel_y};
		separable_filter(*q, *src, *dst, spec, static_cast<border_types>(border), default_value).wait();
	}

	void sobel_filter(Image* src, Image* dst, int kernel_size, int border, Pixel default_value) {
		sycl::queue* q = SingletoneQueue::getInstance();
		sobel_filter_spec spec(kernel_size);
		sobel_filter(*q, *src, *dst, spec, static_cast<border_types>(border), default_value).wait();
	}
	
	void bilateral_filter(  Image* imgOrg, 
							Image* imgDst,
							unsigned int kernel_size, double sigma_intensity, double sigma_distance, int border,
							Pixel default_value){
		sycl::queue* queue = SingletoneQueue::getInstance();
		bilateral_filter_spec<double> spec(kernel_size, sigma_intensity, sigma_distance);						
		bilateral_filter(*queue, *imgOrg, *imgDst, spec, static_cast<border_types>(border), default_value).wait();

	}
	
	void box_filter(Image* imgOrg, 
					Image* imgDst,
					int width, int height, int borde, Pixel default_value){
		sycl::queue* queue = SingletoneQueue::getInstance();
		sycl::range<2> rg(width, height);							
		box_filter_spec spec(rg);
		box_filter<double>(*queue, *imgOrg, *imgDst, spec, static_cast<border_types>(borde), default_value).wait();					
	}

	void gaussian_filter(	Image* imgOrg, 
							Image* imgDst,
							int kernel_size, double sigma_x, double sigma_y, int border, Pixel default_value){
		sycl::queue* queue = SingletoneQueue::getInstance();
		gaussian_filter_spec spec(kernel_size, sigma_x, sigma_y);
		gaussian_filter(*queue, *imgOrg, *imgDst, spec, static_cast<border_types>(border), default_value).wait();
	}

	void filter_convolution(Image* imgOrg, 
							Image* imgDst,
							int width, int height, float kernel_data[], int border, Pixel default_value){
		sycl::queue* queue = SingletoneQueue::getInstance();
		sycl::range<2> rg(width, height);
		filter_convolution_spec spec(rg, kernel_data);
		filter_convolution(*queue, *imgOrg, *imgDst, spec, static_cast<border_types>(border), default_value).wait();					
	}

}