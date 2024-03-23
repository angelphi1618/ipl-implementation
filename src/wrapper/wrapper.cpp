#include "../image.h"
#include "../allocators/device_usm_allocator_t.h"
#include "../image_persistance/bmp_persistance.h"
#include "../image_persistance/png_persistance.h"
#include "../algorithms/bilateral_filter.h"
#include "../algorithms/box_filter.h"
#include "../algorithms/gaussian_filter.h"
#include "../algorithms/filter_convolution.h"
#include "../border_generator/border_types.h"
#include <cstdint>


extern "C" {

	enum bordes{
		repl,
		mirror,
		constante
	};

	sycl::queue* createQueue() {
		sycl::device dev = sycl::device(sycl::cpu_selector());
        std::cout << "hola desde la cola" << std::endl;
		sycl::queue* Q = new sycl::queue(dev);
		return Q;
	}

	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>* createImage(sycl::queue* queue, int width, int height) {
		sycl::range<2> size(width, height);

		std::cout << "w=" << width << ", h=" << height << std::endl;

		return new image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>(*queue, size);
	}

	void loadBMP(image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>* img, char* filename) {
		std::string str(filename);
        std::cout << str << std::endl;
		bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imageLoader(*img);
		std::cout << "hola desde loadBMP"<< std::endl;
		imageLoader.loadImage(str);
	}

	void saveBMP(image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>* img, char* filename) {
		std::string str(filename);
		bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(*img, str);
	}


	void loadPNG(image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>* img, char* filename) {
		std::string str(filename);
        std::cout << str << std::endl;
		png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imageLoader(*img);
		std::cout << "hola desde loadBMP"<< std::endl;
		imageLoader.loadImage(str);
	}

	void savePNG(image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>* img, char* filename) {
		std::string str(filename);
		png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(*img, str);
	}

	
	void bilateral_filter(sycl::queue* queue, 
							image<uint8_t,device_usm_allocator_t<pixel<uint8_t>>>* imgOrg, 
							image<uint8_t,device_usm_allocator_t<pixel<uint8_t>>>* imgDst,
							unsigned int kernel_size, double sigma_intensity, double sigma_distance, int border = 0){

		bilateral_filter_spec<double> spec(kernel_size, sigma_intensity, sigma_distance);						
		bilateral_filter(*queue, *imgOrg, *imgDst, spec, static_cast<border_types>(border)).wait();

	}
	
	void box_filter(sycl::queue* queue, 
							image<uint8_t,device_usm_allocator_t<pixel<uint8_t>>>* imgOrg, 
							image<uint8_t,device_usm_allocator_t<pixel<uint8_t>>>* imgDst,
							int width, int height, int borde){

		sycl::range<2> rg(width, height);							
		box_filter_spec spec(rg);
		box_filter<double>(*queue, *imgOrg, *imgDst, spec, static_cast<border_types>(borde)).wait();					
	}

	void gaussian_filter(sycl::queue* queue, 
							image<uint8_t,device_usm_allocator_t<pixel<uint8_t>>>* imgOrg, 
							image<uint8_t,device_usm_allocator_t<pixel<uint8_t>>>* imgDst,
							int kernel_size, double sigma_x, double sigma_y, int border = 0){

		gaussian_filter_spec spec(kernel_size, sigma_x, sigma_y);
		gaussian_filter(*queue, *imgOrg, *imgDst, spec, static_cast<border_types>(border)).wait();
	}

}