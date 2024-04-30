#pragma once

#include <CL/sycl.hpp>
#include "../pixel.h"
#include "../image.h"
#include "../border_generator/border_types.h"
#include "../border_generator/border_generator.h"
#include "../exceptions/unimplemented.h"


template <typename ComputeT = float>
struct filter_convolution_spec{
	sycl::range<2> kernel_size;
	ComputeT* kernel_data;
	int x_anchor = 0;
	int y_anchor = 0;

	inline filter_convolution_spec(sycl::range<2> kernel_size, ComputeT* kernel_data, int x_anchor, int y_anchor) 
							: kernel_size(kernel_size), kernel_data(kernel_data), x_anchor(x_anchor), y_anchor(y_anchor) {}

	inline filter_convolution_spec(sycl::range<2> kernel_size, ComputeT* kernel_data) 
							: kernel_size(kernel_size), kernel_data(kernel_data) {
								this->x_anchor = (kernel_size.get(0) - 1) / 2;
								this->y_anchor = (kernel_size.get(1) - 1) / 2;
							}
};

template <typename ComputeT = float,
		typename DataT, typename AllocatorT>
sycl::event filter_convolution(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
						const filter_convolution_spec<ComputeT>& kernel,
						border_types border_type = border_types::const_val,
						pixel<DataT> default_value = {},
						const std::vector<sycl::event>& dependencies = {}) {

	switch (border_type)
	{	
	case border_types::const_val:
	case border_types::repl:
		break;
	
	default:
		throw unimplemented("Tipo de borde no soportado");
	}
	

	// image<DataT, AllocatorT>* bordered_image = generate_border(src, kernel.kernel_size, border_type, default_value);
	
	int kernel_width = kernel.kernel_size[0];
	int kernel_height = kernel.kernel_size[1];

	return q.submit([&](sycl::handler& cgh) {

		cgh.depends_on(dependencies);

		pixel<DataT>* src_data = src.get_data();
		ComputeT* kernel_data = static_cast<ComputeT*>(src.get_allocator()->allocate_bytes(kernel.kernel_size.size() * sizeof(ComputeT)));
		q.memcpy(kernel_data, kernel.kernel_data, kernel.kernel_size.size() * sizeof(ComputeT)).wait();

		pixel<DataT>* dst_data = dst.get_data();

		int src_bordered_width  = src.get_size().get(0);
		int src_bordered_height = src.get_size().get(1);

		int dst_width = dst.get_size().get(0);

		// Tamaño de la imagen destino
		int x_anchor = kernel.x_anchor;
		int y_anchor = kernel.y_anchor;

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

				int i_src = it.get_local_id(1) ; //+ top_padding;
				int j_src = it.get_local_id(0) ; //+ left_padding;

				pixel<DataT> suma(0, 0, 0, 255);
				ComputeT R = 0;
				ComputeT G = 0;            
				ComputeT B = 0;
				ComputeT A = 0;

				for (int ii = 0; ii < kernel_height; ii++)
				{
					for (int jj = 0; jj < kernel_width; jj++)
					{
						int ii_src = ii + i_src; // - y_anchor;
						int jj_src = jj + j_src; // - x_anchor;

						pixel<DataT> current_pixel = slm[ii_src * slm_width + jj_src]; 

						R = R + ((ComputeT)current_pixel.R * kernel_data[ii * kernel_width + jj]);
						G = G + ((ComputeT)current_pixel.G * kernel_data[ii * kernel_width + jj]);
						B = B + ((ComputeT)current_pixel.B * kernel_data[ii * kernel_width + jj]);
						A = A + ((ComputeT)current_pixel.A * kernel_data[ii * kernel_width + jj]);
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

template <typename ComputeT = float,
		typename DataT, typename AllocatorT>
sycl::event filter_convolution_roi(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
						const filter_convolution_spec<ComputeT>& kernel,
						border_types border_type = border_types::default_val,
						pixel<DataT> default_value = {},
						const std::vector<sycl::event>& dependencies = {}) {

	switch (border_type)
	{	
	case border_types::const_val:
	case border_types::repl:
		break;
	
	default:
		throw unimplemented("Tipo de borde no soportado");
	}
	

	image<DataT, AllocatorT>* bordered_image = generate_border(src, kernel.kernel_size, border_type, default_value);
	
	int kernel_width = kernel.kernel_size[0];
	int kernel_height = kernel.kernel_size[1];

	return q.submit([&](sycl::handler& cgh) {

		cgh.depends_on(dependencies);

		std::cout << "kernel copiado" << std::endl;

		//std::cout << typeid(*src.get_allocator()).name() << std::endl;

		pixel<DataT>* src_data = bordered_image->get_data();
		ComputeT* kernel_data = static_cast<ComputeT*>(src.get_allocator()->allocate_bytes(kernel.kernel_size.size() * sizeof(ComputeT)));
		q.memcpy(kernel_data, kernel.kernel_data, kernel.kernel_size.size() * sizeof(ComputeT));

		pixel<DataT>* dst_data = dst.get_data();

		int src_bordered_width = bordered_image->get_size().get(0);

		int dst_width = dst.get_size().get(0);

		sycl::stream os(1024*1024, 1024, cgh);

		std::cout << "lanzando parallel for" << std::endl;

		roi_rect roi = src.get_roi_rect();

		// Tamaño de la imagen destino
		int x_anchor = kernel.x_anchor;
		int y_anchor = kernel.y_anchor;

		int src_bordered_height = bordered_image->get_size().get(1);

		q.memcpy(dst_data, src.get_data(), src.get_linear_size() * sizeof(pixel<DataT>)).wait();

		cgh.parallel_for(sycl::range<2>(roi.get_height(), roi.get_width()), [=](sycl::id<2> item){
			// os << "dentro del kernel" << sycl::endl;

			// os << "kernel usado" << sycl::endl;

			int i_destino = item.get(1) + roi.get_x_offset();
			int j_destino = item.get(0) + (src_bordered_height - roi.get_y_offset() - roi.get_height());

			int i_src_bordered = i_destino + kernel_height;
			int j_src_bordered = j_destino + kernel_width;

			pixel<DataT> suma(0, 0, 0, 255);

			ComputeT R = 0;
            ComputeT G = 0;            
            ComputeT B = 0;
            ComputeT A = 0;

			for (int ii = 0; ii < kernel_height; ii++)
			{
				for (int jj = 0; jj < kernel_width; jj++)
				{
					int ii_src_bordered = ii + i_src_bordered - y_anchor;
					int jj_src_bordered = jj + j_src_bordered - x_anchor;

					R = R + ((ComputeT)src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)].R * kernel_data[ii * kernel_width + jj]);
					G = G + ((ComputeT)src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)].G * kernel_data[ii * kernel_width + jj]);
					B = B + ((ComputeT)src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)].B * kernel_data[ii * kernel_width + jj]);
					A = A + ((ComputeT)src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)].A * kernel_data[ii * kernel_width + jj]);
				}
			}

			dst_data[i_destino * dst_width + j_destino] = {
				(DataT) R,
				(DataT) G,
				(DataT) B,
				(DataT) A,
			};

			// os << "sumaR = " << suma.R << ", sumaG = " << suma.G << ", sumaB = " << suma.B << ", sumaA = " << suma.A << sycl::endl;
		});
	});
}