#pragma once

#include <CL/sycl.hpp>
#include "../pixel.h"
#include "../image.h"
#include "../border_generator/border_types.h"
#include "../border_generator/border_generator.h"
#include "../exceptions/invalid_argument.h"
#include "../exceptions/unimplemented.h"

#define MAX_WINDOW 5

template <typename DataT>
void buble_sort(pixel<DataT> array[], int size) {
    int i, j;
	pixel<DataT> tmp;

	for (i=1; i<size; i++)
		for (j=0 ; j<size - i; j++)
			if (array[j].value() > array[j+1].value()){
				tmp = array[j];
				array[j] = array[j+1];
				array[j+1] = tmp;
			}
}

struct median_spec{
	unsigned int window_size;
};

template <typename DataT, typename AllocatorT>
sycl::event median_filter(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
						const median_spec& spec,
						border_types border_type = border_types::repl,
						pixel<DataT> default_value = {},
						const std::vector<sycl::event>& dependencies = {}) {

	switch (border_type)
	{	
	case border_types::const_val:
	case border_types::repl:
		break;
	
	default:
		throw unimplemented("Tipo de no soportado");
	}
	
	if (spec.window_size > MAX_WINDOW)
		throw invalid_argument("Tamaño de ventana no permitido. Máx. 5.");

	return q.submit([&](sycl::handler& cgh) {

		cgh.depends_on(dependencies);
		pixel<DataT>* src_data = src.get_data();
		pixel<DataT>* dst_data = dst.get_data();

		int src_bordered_width  = src.get_size().get(0);
		int src_bordered_height = src.get_size().get(1);

		int dst_width = dst.get_size().get(0);

        int anchor = (spec.window_size - 1) / 2;
		cgh.parallel_for(dst.get_size(), [=](sycl::id<2> item){
            pixel<DataT> window[MAX_WINDOW * MAX_WINDOW];

			int i_destino = item.get(1);
			int j_destino = item.get(0);

			int i_src_bordered = i_destino; // + spec.window_size;
			int j_src_bordered = j_destino; // + spec.window_size;

			for (int ii = 0; ii < spec.window_size; ii++)
			{
				for (int jj = 0; jj < spec.window_size; jj++)
				{
					int ii_src_bordered = ii + i_src_bordered - anchor;
					int jj_src_bordered = jj + j_src_bordered - anchor;

					window[ii * spec.window_size + jj] = bordered_pixel_dispatcher(border_type, src_data, ii_src_bordered, jj_src_bordered, src_bordered_width, src_bordered_height, default_value);
				}
			}

            buble_sort(window, spec.window_size*spec.window_size);

			pixel<DataT> currentPixel = src_data[(i_src_bordered) * src_bordered_width + (j_src_bordered)];
			pixel<DataT> median = window[(spec.window_size * spec.window_size - 1)/2];

			double medianValue = median.value();
			double pixelValue =  currentPixel.value();

			dst_data[i_destino * dst_width + j_destino] = median;
		});
	});

}


template <typename DataT, typename AllocatorT>
sycl::event median_filter_roi(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
						const median_spec& spec,
						border_types border_type = border_types::repl,
						pixel<DataT> default_value = {},
						const std::vector<sycl::event>& dependencies = {}) {

	switch (border_type)
	{	
	case border_types::const_val:
	case border_types::repl:
		break;
	
	default:
		throw unimplemented("Tipo de no soportado");
	}
	
	if (spec.window_size > MAX_WINDOW)
		throw invalid_argument("Tamaño de ventana no permitido. Máx. 5.");

	image<DataT, AllocatorT>* bordered_image = generate_border<DataT, AllocatorT>(src, sycl::range<2>(spec.window_size, spec.window_size), border_type, default_value);


	return q.submit([&](sycl::handler& cgh) {

		cgh.depends_on(dependencies);
		pixel<DataT>* src_data = bordered_image->get_data();
		pixel<DataT>* dst_data = dst.get_data();

		int src_bordered_width = bordered_image->get_size().get(0);
		int dst_width = dst.get_size().get(0);

        int anchor = (spec.window_size - 1) / 2;
		
		roi_rect roi = src.get_roi_rect();
		
		int src_bordered_height = bordered_image->get_size().get(1);
		
		q.memcpy(dst_data, src.get_data(), src.get_linear_size() * sizeof(pixel<DataT>)).wait();

		cgh.parallel_for(sycl::range<2>(roi.get_height(), roi.get_width()), [=](sycl::id<2> item){
            pixel<DataT> window[MAX_WINDOW * MAX_WINDOW];

			int i_destino = item.get(1) + roi.get_x_offset();
			int j_destino = item.get(0) + (src_bordered_height - roi.get_y_offset() - roi.get_height());

			int i_src_bordered = i_destino + spec.window_size;
			int j_src_bordered = j_destino + spec.window_size;

			for (int ii = 0; ii < spec.window_size; ii++)
			{
				for (int jj = 0; jj < spec.window_size; jj++)
				{
					int ii_src_bordered = ii + i_src_bordered - anchor;
					int jj_src_bordered = jj + j_src_bordered - anchor;

					window[ii * spec.window_size + jj] = src_data[ii_src_bordered * src_bordered_width + jj_src_bordered];
				}
			}

            buble_sort(window, spec.window_size*spec.window_size);

			pixel<DataT> currentPixel = src_data[(i_src_bordered) * src_bordered_width + (j_src_bordered)];
			pixel<DataT> median = window[(spec.window_size * spec.window_size - 1)/2];

			double medianValue = median.value();
			double pixelValue =  currentPixel.value();

			dst_data[i_destino * dst_width + j_destino] = median;
		});
	});

}