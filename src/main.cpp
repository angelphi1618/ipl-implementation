#include "allocators/host_usm_allocator_t.h"
#include "allocators/device_usm_allocator_t.h"

#include <CL/sycl.hpp>
#include <cstdint>

#include "image.h"
#include "image_persistance/bmp_persistance.h"

int main() {

	sycl::device dev;
	dev = sycl::device(sycl::cpu_selector());
	sycl::queue Q(dev);

	device_usm_allocator_t<pixel<uint8_t>> loca(Q);	
	
	// imagendata = loca.allocate(...)
	// ev1 = Q.submit(...)
	// ev2 = Q.submit(...)
	// depencees (ev1, ev2)

	// imagen(imagendata, dependeces) 


	std::cout << "Running on "
		<< Q.get_device().get_info<sycl::info::device::name>()
		<< std::endl;

	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagen(Q, sycl::range(1200, 900), loca);

	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imageloader(imagen);

	imageloader.loadImage("lolita.bmp");
	imageloader.saveImage("lolita2.bmp");

	image<uint8_t> imagen2(Q, sycl::range(1200, 900));
	bmp_persistance<uint8_t> imageloader2(imagen2);

	imageloader2.loadImage("lolita.bmp");
	imageloader2.saveImage("lolita3.bmp");

}
