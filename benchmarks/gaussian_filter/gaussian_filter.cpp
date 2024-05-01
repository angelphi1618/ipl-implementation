#include <CL/sycl.hpp>
#include <cstdint>

#include "../../src/image.h"
#include "../../src/allocators/device_usm_allocator_t.h"
#include "../../src/image_persistance/png_persistance.h"

#include "../../src/algorithms/gaussian_filter.h"


int main() {
	sycl::device dev;
	dev = sycl::device(sycl::gpu_selector());
	sycl::queue Q(dev);

	device_usm_allocator_t<pixel<uint8_t>> loca(Q);

	roi_rect rectangulo(sycl::range<2>(300,300), sycl::range<2>(496,60));


	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagen(Q,       sycl::range(1024, 683), loca, rectangulo);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenBox(Q,    sycl::range(1024, 683), loca);

	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imageLoader(imagen);
	imageLoader.loadImage("../../../figures/fdi.png");;

	gaussian_filter_spec<double> gaussian_spec = {75, 6.4, 6.4};

	for (size_t i = 0; i < 5; i++)
	{
		gaussian_filter<double>(Q, imagen, imagenBox, gaussian_spec, border_types::repl).wait();
		gaussian_filter<double>(Q, imagenBox, imagen, gaussian_spec, border_types::repl).wait();
	}
	

	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(imagenBox, "./gaussian_filter.png");

	return 0;
}