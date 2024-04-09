#include <CL/sycl.hpp>
#include <cstdint>

#include "../../image.h"
#include "../../allocators/device_usm_allocator_t.h"
#include "../../image_persistance/bmp_persistance.h"

#include "../../algorithms/bilateral_filter.h"


int main() {
	sycl::device dev;
	dev = sycl::device(sycl::gpu_selector());
	sycl::queue Q(dev);

	device_usm_allocator_t<pixel<uint8_t>> loca(Q);

	roi_rect rectangulo(sycl::range<2>(300,300), sycl::range<2>(496,60));


	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagen(Q, sycl::range(1200, 900), loca, rectangulo);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenBox(Q, sycl::range(1200, 900), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenBoxRoi(Q, sycl::range(1200, 900), loca);

	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imageLoader(imagen);
	imageLoader.loadImage("../../images/lolita.bmp");

	bilateral_filter_spec<double> bilateral_spec(9, 75, 75);

	bilateral_filter<double>(Q, imagen, imagenBox, bilateral_spec, border_types::repl);
	bilateral_filter_roi<double>(Q, imagen, imagenBoxRoi, bilateral_spec, border_types::repl);

	Q.wait();
		bilateral_filter<double>(Q, imagen, imagenBox, bilateral_spec, border_types::repl);
	bilateral_filter_roi<double>(Q, imagen, imagenBoxRoi, bilateral_spec, border_types::repl);

	Q.wait();
		bilateral_filter<double>(Q, imagen, imagenBox, bilateral_spec, border_types::repl);
	bilateral_filter_roi<double>(Q, imagen, imagenBoxRoi, bilateral_spec, border_types::repl);

	Q.wait();
		bilateral_filter<double>(Q, imagen, imagenBox, bilateral_spec, border_types::repl);
	bilateral_filter_roi<double>(Q, imagen, imagenBoxRoi, bilateral_spec, border_types::repl);

	Q.wait();
		bilateral_filter<double>(Q, imagen, imagenBox, bilateral_spec, border_types::repl);
	bilateral_filter_roi<double>(Q, imagen, imagenBoxRoi, bilateral_spec, border_types::repl);

	Q.wait();
		bilateral_filter<double>(Q, imagen, imagenBox, bilateral_spec, border_types::repl);
	bilateral_filter_roi<double>(Q, imagen, imagenBoxRoi, bilateral_spec, border_types::repl);

	Q.wait();

	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(imagenBox, "./bilateral_filter.bmp");
	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(imagenBoxRoi, "./bilateral_filter_roi.bmp");

	return 0;

}