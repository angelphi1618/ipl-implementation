#include <CL/sycl.hpp>
#include <cstdint>

#include "../../src/image.h"
#include "../../src/allocators/device_usm_allocator_t.h"
#include "../../src/image_persistance/bmp_persistance.h"

#include "../../src/algorithms/filter_convolution.h"


int main() {
	sycl::device dev;
	dev = sycl::device(sycl::cpu_selector());
	sycl::queue Q(dev);

	device_usm_allocator_t<pixel<uint8_t>> loca(Q);

	roi_rect rectangulo(sycl::range<2>(300,300), sycl::range<2>(496,60));

	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagen(Q, sycl::range(1200, 900), loca, rectangulo);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenBox(Q, sycl::range(1200, 900), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenBoxRoi(Q, sycl::range(1200, 900), loca);

	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imageLoader(imagen);
	imageLoader.loadImage("../../images/lolita.bmp");

	std::vector<double> kernel2{1, 0  -1,
							 2, 0, -2,
							 1, 0, -1};

	filter_convolution_spec<double> kernel_spec({3 ,3}, kernel2.data(),1, 1);

	filter_convolution<double, uint8_t, device_usm_allocator_t<pixel<uint8_t>>>(Q, imagen, imagenBox, kernel_spec, border_types::repl);
	filter_convolution_roi(Q, imagen, imagenBoxRoi, kernel_spec, border_types::repl);

	Q.wait();

	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(imagenBox, "./filter_convolution.bmp");
	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(imagenBoxRoi, "./filter_convolution_roi.bmp");

	return 0;

}