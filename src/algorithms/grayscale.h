#pragma once

#include <CL/sycl.hpp>
#include "../pixel.h"
#include "../image.h"

template <typename ComputeT = pixel<>,
		typename DataT, typename AllocatorT>
sycl::event rgb_to_gray(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
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
			image_dst[i].A = image_src[i].A;
		});
	});
}

template <typename ComputeT = pixel<>,
		typename DataT, typename AllocatorT>
sycl::event rgb_to_gray_roi(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
						const std::vector<sycl::event>& dependencies = {}) {

	return q.submit([&](sycl::handler& cgh) {

		cgh.depends_on(dependencies);

		pixel<DataT>* image_src = src.get_data();
		pixel<DataT>* image_dst = dst.get_data();

		// Imagen entera copiada
		q.memcpy(image_dst, image_src, src.get_linear_size() * sizeof(pixel<DataT>)).wait();

		roi_rect roi = src.get_roi_rect();

		int width = src.get_size().get(0);
		int height = src.get_size().get(1);

		cgh.parallel_for(sycl::range<2>(roi.get_width(), roi.get_height()), [=](sycl::id<2> idx){

			int i = (roi.get_x_offset() + idx[0]) + (idx[1] + (height - roi.get_height() - roi.get_y_offset())) * width;
			DataT gris = 0.299 * image_src[i].R + 0.587 * image_src[i].G + 0.114 * image_src[i].B;

			image_dst[i].R = gris;
			image_dst[i].G = gris;
			image_dst[i].B = gris;
		});
	});
}