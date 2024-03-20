#include "../image.h"
#include "../allocators/device_usm_allocator_t.h"
#include "../image_persistance/bmp_persistance.h"
#include <cstdint>


extern "C" {

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


    //Algotithms

    // void filter_convolution(sycl::queue* queue, image<uint8_t, 
    //                     device_usm_allocator_t<pixel<uint8_t>>>* imgOrg, image<uint8_t, 
    //                     device_usm_allocator_t<pixel<uint8_t>>>* imgDst){
                        
    // }
}