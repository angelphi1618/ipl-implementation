#include <CL/sycl.hpp>
#include <cstdint>

#include "../../image.h"
#include "../../allocators/device_usm_allocator_t.h"
#include "../../image_persistance/bmp_persistance.h"

#include "../../algorithms/median_filter.h"


int main() {
	sycl::device dev;
	dev = sycl::device(sycl::gpu_selector());
	sycl::queue Q(dev);

	device_usm_allocator_t<pixel<uint8_t>> loca(Q);

	roi_rect rectangulo(sycl::range<2>(200,200), sycl::range<2>(200, 200));


	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagen(Q, sycl::range(512, 512), loca, rectangulo);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenBox(Q, sycl::range(512, 512), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenBoxRoi(Q, sycl::range(512, 512), loca);

	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imageLoader(imagen);
	imageLoader.loadImage("../../images/prueba.bmp");

	median_spec median = {3};

	median_filter<uint8_t>(Q, imagen, imagenBox, median);
	median_filter_roi<uint8_t>(Q, imagen, imagenBoxRoi, median);

	Q.wait();

	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(imagenBox, "./median_filter.bmp");
	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(imagenBoxRoi, "./median_filter_roi.bmp");

	return 0;

}