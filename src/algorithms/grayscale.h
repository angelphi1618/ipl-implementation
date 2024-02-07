#pragma once

#include <CL/sycl.hpp>
#include "../pixel.h"
#include "../image.h"

template <typename ComputeT = pixel<>,
		typename DataT, typename AllocatorT>
sycl::event grayscale(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
						const std::vector<sycl::event>& dependencies = {}) {

	return q.submit([&](sycl::handler& cgh) {

		cgh.depends_on(dependencies);

		pixel<DataT>* image_src = src.get_data();
		pixel<DataT>* image_dst = dst.get_data();

		cgh.parallel_for(sycl::range<1>(src.get_linear_size()), [=](sycl::id<1> i){
			DataT gris = 0.299 * image_src[i].R + 0.587 * image_src[i].G + 0.114 * image_src[i].B;
			image_dst[i].R = gris;
			image_dst[i].G = gris;
			image_dst[i].B = gris;
		});
	});
}