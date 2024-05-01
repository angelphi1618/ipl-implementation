#include <CL/sycl.hpp>
#include <cstdint>

#include "../../src/image.h"
#include "../../src/allocators/device_usm_allocator_t.h"
#include "../../src/image_persistance/png_persistance.h"

#include "../../src/algorithms/median_filter.h"


int main() {
	sycl::device dev;
	dev = sycl::device(sycl::gpu_selector());
	sycl::queue Q(dev);

	device_usm_allocator_t<pixel<uint8_t>> loca(Q);

	roi_rect rectangulo(sycl::range<2>(200,200), sycl::range<2>(200, 200));


	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagen(Q, sycl::range(1024, 683), loca, rectangulo);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenBox(Q, sycl::range(1024, 683), loca);

	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imageLoader(imagen);
	imageLoader.loadImage("../../../figures/fdiNoise.png");

	median_spec median = {3};

	for (size_t i = 0; i < 5; i++)
	{
		median_filter<uint8_t>(Q, imagen, imagenBox, median).wait();
		median_filter<uint8_t>(Q, imagenBox, imagen, median).wait();
	}

	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(imagenBox, "./median_filter.png");

	return 0;

}