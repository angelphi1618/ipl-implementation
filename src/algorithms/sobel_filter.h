#pragma once

#include <CL/sycl.hpp>
#include "../pixel.h"
#include "../image.h"
#include "../border_generator/border_types.h"
#include "../border_generator/border_generator.h"

struct sobel_filter_spec{
	size_t kernel_size;

	sobel_filter_spec(size_t kernel_size) : kernel_size(kernel_size) {}
};

template <typename DataT, typename AllocatorT>
sycl::event sobel_filter(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
						const sobel_filter_spec& kernel,
						border_types border_type = border_types::default_val,
						pixel<DataT> default_value = {},
						const std::vector<sycl::event>& dependencies = {})
{
	switch (border_type)
	{	
	case border_types::const_val:
	case border_types::repl:
		break;
	
	default:
		throw unimplemented("Tipo de borde no soportado");
	}

	if (kernel.kernel_size == 0)
		throw invalid_argument("El tamaño del kernel sobel no es válido");

	if (kernel.kernel_size != 3 && kernel.kernel_size != 5)
		throw unimplemented("El filtro sobel sólo soporta kernels de 3x3 o de 5x5");

	image<DataT, AllocatorT>* bordered_image = generate_border(src, sycl::range<2>(kernel.kernel_size, kernel.kernel_size) , border_type, default_value);

	int kernel_width = kernel.kernel_size;
	int kernel_height = kernel.kernel_size;

	return q.submit([&](sycl::handler& cgh) {
		cgh.depends_on(dependencies);
		
		pixel<DataT>* src_data = bordered_image->get_data();

		pixel<DataT>* dst_data = dst.get_data();

		int src_bordered_width = bordered_image->get_size().get(0);
		int dst_width = dst.get_size().get(0);
		int x_anchor = (kernel.kernel_size - 1) / 2;
		int y_anchor = (kernel.kernel_size - 1) / 2;


		cgh.parallel_for(dst.get_size(), [=](sycl::id<2> item){

			int kernel_data_x[5 * 5];
			int kernel_data_y[5 * 5];

			if (kernel.kernel_size == 3){
				int temp_kernel_data_x[3*3] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
				int temp_kernel_data_y[3*3] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

				for (int i = 0; i < 3 * 3; i++){
					kernel_data_x[i] = temp_kernel_data_x[i];
					kernel_data_y[i] = temp_kernel_data_y[i];
				}
			} else {
				int temp_kernel_data_x[5*5] = {5, 8, 10, 8, 5,
												4, 10, 20, 10, 4,
												0,0,0,0,0,
												-4,-10,-20,-10,-4,
												-5,-8,-10,-8,-5};

				int temp_kernel_data_y[5 * 5]= {-5, -4, 0, 4, 5,
												-8, -10,0,10,8,
												-10, -20, 0, 20, 10,
												-8, -10, 0, 10, 8,
												-5, -4, 0, 4, 5};
				for (int i = 0; i < 5 * 5; i++){
					kernel_data_x[i] = temp_kernel_data_x[i];
					kernel_data_y[i] = temp_kernel_data_y[i];
				}
			}

			int i_destino = item.get(1);
			int j_destino = item.get(0);

			int i_src_bordered = i_destino + kernel_height;
			int j_src_bordered = j_destino + kernel_width;

			int Gx_R = 0, Gy_R = 0;
			int Gx_G = 0, Gy_G = 0;
			int Gx_B = 0, Gy_B = 0;
			int Gx_A = 255, Gy_A = 255 ;

			for (int ii = 0; ii < kernel.kernel_size; ii++)
			{
				for (int jj = 0; jj < kernel.kernel_size; jj++)
				{
					int ii_src_bordered = ii + i_src_bordered - y_anchor;
					int jj_src_bordered = jj + j_src_bordered - x_anchor;

					Gx_R = Gx_R + (src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)].R * kernel_data_x[ii * kernel_width + jj]);
					Gx_G = Gx_G + (src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)].G * kernel_data_x[ii * kernel_width + jj]);
					Gx_B = Gx_B + (src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)].B * kernel_data_x[ii * kernel_width + jj]);

					Gy_R = Gy_R + (src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)].R * kernel_data_y[ii * kernel_width + jj]);
					Gy_G = Gy_G + (src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)].G * kernel_data_y[ii * kernel_width + jj]);
					Gy_B = Gy_B + (src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)].B * kernel_data_y[ii * kernel_width + jj]);
				}
			}

			dst_data[i_destino * dst_width + j_destino] = {
				(DataT)(sycl::abs<int>(Gx_R) + sycl::abs<int>(Gy_R)),
				(DataT)(sycl::abs<int>(Gx_G) + sycl::abs<int>(Gy_G)),
				(DataT)(sycl::abs<int>(Gx_B) + sycl::abs<int>(Gy_B)),
				static_cast<DataT>(Gx_A),
			};
		});
	});

}