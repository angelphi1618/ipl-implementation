#include "../image.h"
#include "../allocators/device_usm_allocator_t.h"
#include "../image_persistance/bmp_persistance.h"
#include "../image_persistance/png_persistance.h"
#include "../algorithms/grayscale.h"
#include "../algorithms/median_filter.h"
#include "../algorithms/separable_filter.h"

#include <cstdint>


extern "C" {

	using Image = ::image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>;

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

	// void sobelFilter(sycl::queue* q, Image* src, Image* dst, int kernel_size) {
	// 	sobel_filter_spec kernel_spec(kernel_size);
	// 	sobel_filter(*q, *src, *dst, kernel_spec, border_types::repl).wait();
	// }

	void medianFilter(sycl::queue* q, Image* src, Image* dst, unsigned int window_size) {
		median_spec window_spec = {window_size};
		median_filter(*q, *src, *dst, window_spec).wait();
	}

	void rgb_to_gray(sycl::queue* q, Image* src, Image*dst) {
		rgb_to_gray(*q, *src, *dst).wait();
	}

	void separable_filter(sycl::queue* q, Image* src, Image* dst, int width, int height, int kernel_x[], int kernel_y[]) {
		separable_spec<int> spec{sycl::range<2>(height, width), kernel_x, kernel_y};
		separable_filter(*q, *src, *dst, spec, border_types::repl).wait();
	}
    //Algotithms

    // void filter_convolution(sycl::queue* queue, image<uint8_t, 
    //                     device_usm_allocator_t<pixel<uint8_t>>>* imgOrg, image<uint8_t, 
    //                     device_usm_allocator_t<pixel<uint8_t>>>* imgDst){
                        
    // }
}