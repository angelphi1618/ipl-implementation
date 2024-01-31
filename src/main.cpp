#include "allocators/host_usm_allocator_t.h"
#include "allocators/device_usm_allocator_t.h"

#include <CL/sycl.hpp>
#include <cstdint>

#include "image.h"
#include "image_persistance.h"

int main() {

	sycl::device dev;
	dev = sycl::device(sycl::cpu_selector());
	sycl::queue Q(dev);

	device_usm_allocator_t<pixel<uint8_t>> loca(Q);	

	std::cout << "Running on "
		<< Q.get_device().get_info<sycl::info::device::name>()
		<< std::endl;

	
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagen(Q, sycl::range(1200, 900), loca);

	image_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imageloader(imagen);

	imageloader.loadImage("lolita.bmp");
	imageloader.saveImage("lolita2.bmp");

}
