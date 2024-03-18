#include "ipl.h"

int main(){


	sycl::device dev;
	dev = sycl::device(sycl::gpu_selector());
	sycl::queue Q(dev);

	device_usm_allocator_t<pixel<uint8_t>> loca(Q);
		roi_rect rectangulo1(sycl::range<2>(300,300), sycl::range<2>(496,60));

	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagen(Q, sycl::range(1200, 900), loca, rectangulo1);


	std::cout << "hola" << std::endl;
}