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
	

	image<DataT, AllocatorT>* bordered_image = generate_border(src, kernel.kernel_size, border_type, default_value);
	
	int kernel_width = kernel.kernel_size[0];
	int kernel_height = kernel.kernel_size[1];

	

	return q.submit([&](sycl::handler& cgh) {

		cgh.depends_on(dependencies);

		pixel<DataT>* src_data = bordered_image->get_data();
		ComputeT* kernel_data = static_cast<ComputeT*>(src.get_allocator()->allocate_bytes(kernel.kernel_size.size() * sizeof(ComputeT)));
		q.memcpy(kernel_data, kernel.kernel_data, kernel.kernel_size.size() * sizeof(ComputeT));

		pixel<DataT>* dst_data = dst.get_data();

		int src_bordered_width = bordered_image->get_size().get(0);
		int dst_width = dst.get_size().get(0);

		// Tamaño de la imagen destino
		int x_anchor = kernel.x_anchor;
		int y_anchor = kernel.y_anchor;
		cgh.parallel_for(dst.get_size(), [=](sycl::id<2> item){

			int i_destino = item.get(1);
			int j_destino = item.get(0);

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