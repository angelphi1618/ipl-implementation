#pragma once

#include <CL/sycl.hpp>
#include "../pixel.h"
#include "../image.h"
#include "../border_generator/border_types.h"
#include "../border_generator/border_generator.h"

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


template <typename DataT, typename AllocatorT>
sycl::event median_filter(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
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
	

	image<DataT, AllocatorT>* bordered_image = generate_border<DataT, AllocatorT>(src, sycl::range<2>(MAX_WINDOW, MAX_WINDOW), border_type, default_value);


	return q.submit([&](sycl::handler& cgh) {

		cgh.depends_on(dependencies);

		std::cout << "kernel copiado" << std::endl;

		//std::cout << typeid(*src.get_allocator()).name() << std::endl;

		pixel<DataT>* src_data = bordered_image->get_data();

		pixel<DataT>* dst_data = dst.get_data();

		int src_bordered_width = bordered_image->get_size().get(0);

		int dst_width = dst.get_size().get(0);

		sycl::stream os(1024*1024, 1024, cgh);

		std::cout << "lanzando parallel for" << std::endl;
        int anchor = (MAX_WINDOW - 1) / 2;
		cgh.parallel_for(dst.get_size(), [=](sycl::id<2> item){
			// os << "dentro del kernel" << sycl::endl;

			// os << "kernel usado" << sycl::endl;

            pixel<DataT> window[MAX_WINDOW * MAX_WINDOW];

			int i_destino = item.get(1);
			int j_destino = item.get(0);

			int i_src_bordered = i_destino + MAX_WINDOW;
			int j_src_bordered = j_destino + MAX_WINDOW;

			for (int ii = 0; ii < MAX_WINDOW; ii++)
			{
				for (int jj = 0; jj < MAX_WINDOW; jj++)
				{
					int ii_src_bordered = ii + i_src_bordered - anchor;
					int jj_src_bordered = jj + j_src_bordered - anchor;

					window[ii * MAX_WINDOW + jj] = src_data[ii_src_bordered * src_bordered_width + jj_src_bordered];
				}
			}

            buble_sort(window, MAX_WINDOW);
			pixel<DataT> currentPixel = src_data[(i_src_bordered) * src_bordered_width + (j_src_bordered)];
			pixel<DataT> median = window[(anchor * anchor - 1)/2];
			double medianValue = median.value();
			double pixelValue =  currentPixel.value();

			if (currentPixel.value() < 0.2){
				os << "pixel negro(" << median.R << ", " << median.G << ", " << median.B << ")"<< sycl::endl;
			}

			if (median.value() < 0.2){
				os << "mediana negra(" << median.R << ", " << median.G << ", " << median.B << ")"<< sycl::endl;
			}

			pixel<DataT> green(0, 255, 0, 255);
			//os << "mediana: " << (int) medianValue << "; pixel: " << (int) pixelValue << sycl::endl;

			dst_data[i_destino * dst_width + j_destino] = median;

			// if (fabs((median.value() - currentPixel.value()) / median.value()) <= 0.2) {
			// 	dst_data[i_destino * dst_width + j_destino] = currentPixel;
			// }else {
			// 	dst_data[i_destino * dst_width + j_destino] = median;
			// }
			//dst_data[i_destino * dst_width + j_destino] = window[MAX_WINDOW * MAX_WINDOW / 2];

			// os << "sumaR = " << suma.R << ", sumaG = " << suma.G << ", sumaB = " << suma.B << ", sumaA = " << suma.A << sycl::endl;
		});
	});

}






