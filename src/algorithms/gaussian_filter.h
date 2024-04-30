#pragma once

#include <CL/sycl.hpp>
#include "../pixel.h"
#include "../image.h"
#include "../border_generator/border_types.h"
#include "../border_generator/border_generator.h"
#include "../exceptions/unimplemented.h"

#include <cmath>
#include "filter_convolution.h"

template <typename ComputeT = float>
struct gaussian_filter_spec{
	unsigned int kernel_size;
	ComputeT sigma_x;
	ComputeT sigma_y;

	inline gaussian_filter_spec(int kernel_size, ComputeT sigma_x, ComputeT sigma_y) : 
			kernel_size(kernel_size), sigma_x(sigma_x), sigma_y(sigma_y) {}
};


template <typename ComputeT = double>
inline ComputeT gaussian_kernel_at(int i, int j, int kernel_size, int twice_sigma_x_sqrd, int twice_sigma_y_sqrd, ComputeT normalization_factor){
	int k = kernel_size;

	ComputeT x_axis_component = (k/2 - i) * (k/2 - i);
	ComputeT y_axis_component = (k/2 - j) * (k/2 - j);

	x_axis_component = x_axis_component / twice_sigma_x_sqrd;
	y_axis_component = y_axis_component / twice_sigma_y_sqrd;

	return exp(-(x_axis_component + y_axis_component)) * normalization_factor;
}


template <typename ComputeT = float, typename DataT, typename AllocatorT>
sycl::event gaussian_filter(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
						const gaussian_filter_spec<ComputeT>& kernel_spec,
						border_types border_type = border_types::const_val,
						pixel<DataT> default_value = {},
						const std::vector<sycl::event>& dependencies = {})
{
    switch (border_type)
	{	
	case border_types::const_val:
	case border_types::repl:
    case border_types::mirror:
		break;
	default:
		throw unimplemented("Tipo de borde no soportado");
	}

	// El kernel es cuadrado
	int kernel_width = kernel_spec.kernel_size;
	int kernel_height = kernel_spec.kernel_size;

	ComputeT* gaussian_kernel = static_cast<ComputeT*>(src.get_allocator()->allocate_bytes(kernel_width*kernel_height*sizeof(ComputeT)));

	// Inicializamos el kernel gaussiano
	return q.submit([&](sycl::handler& cgh) {

		cgh.depends_on(dependencies);

		pixel<DataT>* src_data = src.get_data();
		pixel<DataT>* dst_data = dst.get_data();


		ComputeT sigma_x = kernel_spec.sigma_x;
		ComputeT sigma_y = kernel_spec.sigma_y;
		ComputeT normalization_factor = 1 / (2 * M_PI * sigma_x * sigma_y);

		int k = kernel_width;

		int twice_sigma_x_sqrd = 2 * sigma_x * sigma_x;
		int twice_sigma_y_sqrd = 2 * sigma_y * sigma_y;

		int src_bordered_width  = src.get_size().get(0);
		int src_bordered_height = src.get_size().get(1);

		int dst_width = dst.get_size().get(0);
		int dst_height = dst.get_size().get(0);

		// Tamaño de la imagen destino
		int x_anchor = (kernel_width - 1) / 2;
		int y_anchor = (kernel_height - 1) / 2;

		int w = src_bordered_width;
		int h = src_bordered_height;

		// Obtenemos el tamaño máximo de work-group ...
		const int max_size = q.get_device().get_info<cl::sycl::info::device::max_work_group_size>();

		// ...fragmentamos ese tamaño en 2 dimensiones
		int work_group_w = floor(sqrt(max_size));
		int work_group_h = max_size / work_group_w;

		// Tamaño de cada work-group
		sycl::range<2> local(work_group_w, work_group_h);

		// Obtenemos la cantidad de los work-groups en cada dimensión
		int new_w = ceil(w / static_cast<ComputeT>(work_group_w)); // * work_group_w;
		int new_h = ceil(h / static_cast<ComputeT>(work_group_h)); // * work_group_h;

		sycl::range<2> global(new_w, new_h);

		// Preparamos las variables para hacer uso de la memoria local de cada work-group
		// Necesitamos un elemento por cada work-item y además los márgenes del work-group
		// Los márgenes están definidos por el anchor del kernel que vamos a procesar
		int left_padding = x_anchor ;
		int right_padding = kernel_width - x_anchor;

		int top_padding = y_anchor;
		int bottom_padding = kernel_height - y_anchor;

		// Tamaño de la memoria local por cada work-group
		int slm_width  = left_padding + work_group_w + right_padding ;
		int slm_height = top_padding  + work_group_h + bottom_padding;

		sycl::accessor<pixel<DataT>, 1, sycl::access::mode::read_write, sycl::access::target::local> slm(slm_width*slm_height, cgh);

		cgh.parallel_for_work_group(global, local, [=](sycl::group<2> grp){

			// Copiamos la memoria global a local teniendo en cuenta los bordes
			// Cada parallel_for_work_item actúa como barrera global del siguiente
			grp.parallel_for_work_item(sycl::range<2>(slm_width, slm_height), [=](sycl::h_item<2> it) {
				int i_destino = it.get_local_id(1); // 0 ... slm_height - 1
				int j_destino = it.get_local_id(0); // 0 ... slm_width - 1

				int i_src = grp.get_id(1) * work_group_h + i_destino - top_padding;
				int j_src = grp.get_id(0) * work_group_w + j_destino - left_padding;

				slm[i_destino*slm_width + j_destino] = 
					bordered_pixel_dispatcher(border_type, src_data, 
												i_src, j_src, 
												src_bordered_width,
												src_bordered_height, 
												default_value);
			});

			// Ejecución de la convolución
			grp.parallel_for_work_item([=](sycl::h_item<2> it) {
				int i_destino = grp.get_id(1) * work_group_h + it.get_local_id(1) ;
				int j_destino = grp.get_id(0) * work_group_w + it.get_local_id(0) ;

				j_destino = sycl::min<int>(j_destino, dst_width - 1);

				int i_src = it.get_local_id(1);// + top_padding;
				int j_src = it.get_local_id(0);// + left_padding;

				ComputeT R = 0;
				ComputeT G = 0;            
				ComputeT B = 0;
				ComputeT A = 0;
				pixel<DataT> current_pixel = slm[i_src * slm_width + j_src]; 

				for (int ii = 0; ii < kernel_height; ii++)
				{
					for (int jj = 0; jj < kernel_width; jj++)
					{
						int ii_src = ii + i_src;//  + y_anchor;
						int jj_src = jj + j_src;//  + x_anchor;

						pixel<DataT> current_pixel = slm[ii_src * slm_width + jj_src]; 
						ComputeT kernel_val = gaussian_kernel_at(ii, jj, kernel_width, twice_sigma_x_sqrd, twice_sigma_y_sqrd, normalization_factor);
						R = R + ((ComputeT)(current_pixel.R) * kernel_val);
						G = G + ((ComputeT)(current_pixel.G) * kernel_val);
						B = B + ((ComputeT)(current_pixel.B) * kernel_val);
						A = A + ((ComputeT)(current_pixel.A) * kernel_val);
					}
				}

				dst_data[i_destino * dst_width + j_destino] = {
					(DataT) R,
					(DataT) G,
					(DataT) B,
					(DataT) A,
				};
			});
		});
	});
}

template <typename ComputeT = float, typename DataT, typename AllocatorT>
sycl::event gaussian_filter_roi(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
						const gaussian_filter_spec<ComputeT>& kernel_spec,
						border_types border_type = border_types::default_val,
						pixel<DataT> default_value = {},
						const std::vector<sycl::event>& dependencies = {})
{
    switch (border_type)
	{	
	case border_types::const_val:
	case border_types::repl:
    case border_types::mirror:
		break;
	default:
		throw unimplemented("Tipo de borde no soportado");
	}

	// El kernel es cuadrado
	int kernel_width = kernel_spec.kernel_size;
	int kernel_height = kernel_spec.kernel_size;

	ComputeT* gaussian_kernel = static_cast<ComputeT*>(src.get_allocator()->allocate_bytes(kernel_width*kernel_height*sizeof(ComputeT)));

	// Inicializamos el kernel gaussiano
	q.submit([&](sycl::handler& cgh) {

		ComputeT sigma_x = kernel_spec.sigma_x;
		ComputeT sigma_y = kernel_spec.sigma_y;
		ComputeT normalization_factor = 1 / (2 * M_PI * sigma_x * sigma_y);

		int k = kernel_width;

		int twice_sigma_x_sqrd = 2 * sigma_x * sigma_x;
		int twice_sigma_y_sqrd = 2 * sigma_y * sigma_y;

		cgh.parallel_for(sycl::range<2>(kernel_height, kernel_width), [=](sycl::id<2> item){
			int i = item.get(0);
			int j = item.get(1);

			int index = (i * kernel_width) + j;

			ComputeT x_axis_component = (k/2 - i) * (k/2 - i);
			ComputeT y_axis_component = (k/2 - j) * (k/2 - j);

			x_axis_component = x_axis_component / twice_sigma_x_sqrd;
			y_axis_component = y_axis_component / twice_sigma_y_sqrd;

			ComputeT val = exp(-(x_axis_component + y_axis_component)) * normalization_factor;
			gaussian_kernel[index] = val;

		});
	}).wait();

	// Generamos el kernel tal y como espera filter_convolution
	filter_convolution_spec<ComputeT> kernel(sycl::range<2>(kernel_width, kernel_height), gaussian_kernel);

	return filter_convolution_roi(q, src, dst, kernel, border_type, default_value, dependencies);
}