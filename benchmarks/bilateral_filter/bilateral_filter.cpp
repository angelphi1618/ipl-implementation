#include <CL/sycl.hpp>
#include <cstdint>

#include "../../src/image.h"
#include "../../src/allocators/device_usm_allocator_t.h"
#include "../../src/image_persistance/png_persistance.h"

#include "../../src/algorithms/bilateral_filter.h"


int main() {
	sycl::device dev;
	dev = sycl::device(sycl::gpu_selector());
	sycl::queue Q(dev);

	device_usm_allocator_t<pixel<uint8_t>> loca(Q);

	roi_rect rectangulo(sycl::range<2>(300,300), sycl::range<2>(496,60));

	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagen(Q,       sycl::range(1024, 683), loca, rectangulo);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenBox(Q,    sycl::range(1024, 683), loca);

	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imageLoader(imagen);
	imageLoader.loadImage("../../../figures/fdi.png");

	bilateral_filter_spec<double> bilateral_spec(36, 150, 150);

	for (size_t i = 0; i < 5; i++)
	{
		bilateral_filter<double>(Q, imagen, imagenBox, bilateral_spec, border_types::repl).wait();
		bilateral_filter<double>(Q, imagenBox, imagen, bilateral_spec, border_types::repl).wait();
	}
	

	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(imagenBox, "./bilateral_filter.png");

	return 0;

}