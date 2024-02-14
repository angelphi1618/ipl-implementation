#pragma once

#include <CL/sycl.hpp>
#include "../pixel.h"
#include "../image.h"
#include "../border_generator/border_types.h"
#include "../border_generator/border_generator.h"


struct box_filter_spec{
	sycl::range<2> kernel_size;
    int x_anchor;
    int y_anchor;

    inline box_filter_spec(sycl::range<2> kernel_size) : kernel_size(kernel_size) {
        this->x_anchor = (kernel_size.get(0) - 1) / 2;
        this->y_anchor = (kernel_size.get(1) - 1) / 2;
    }
};


template <typename ComputeT, typename DataT, typename AllocatorT>
sycl::event box_filter(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
						const box_filter_spec& kernel,
						border_types border_type = border_types::default_val,
						pixel<DataT> default_value = {},
						const std::vector<sycl::event>& dependencies = {}) {



    switch (border_type)
	{	
	case border_types::const_val:
	case border_types::repl:
    case border_types::mirror:
		break;
	
	default:
		throw unimplemented("Tipo de no soportado");
	}

    image<DataT, AllocatorT>* bordered_image = generate_border(src, kernel.kernel_size, border_type, default_value);
	
	int kernel_width = kernel.kernel_size[0];
	int kernel_height = kernel.kernel_size[1];

	

	return q.submit([&](sycl::handler& cgh) {

		cgh.depends_on(dependencies);

		pixel<DataT>* src_data = bordered_image->get_data();

		pixel<DataT>* dst_data = dst.get_data();

		int src_bordered_width = bordered_image->get_size().get(0);

		int dst_width = dst.get_size().get(0);

        ComputeT alpha = 1 / (ComputeT) kernel.kernel_size.size();

        std::cout << "Valor de alpha " << alpha << std::endl;

		sycl::stream os(1024*1024, 1024, cgh);

		std::cout << "lanzando parallel for" << std::endl;
		// Tamaño de la imagen destino
		int x_anchor = kernel.x_anchor;
		int y_anchor = kernel.y_anchor;
		cgh.parallel_for(dst.get_size(), [=](sycl::id<2> item){
			// os << "dentro del kernel" << sycl::endl;

			// os << "kernel usado" << sycl::endl;

			int i_destino = item.get(1);
			int j_destino = item.get(0);

			int i_src_bordered = i_destino + kernel_height;
			int j_src_bordered = j_destino + kernel_width;

			pixel<ComputeT> suma(0, 0, 0, 255);

			for (int ii = 0; ii < kernel_height; ii++)
			{
				for (int jj = 0; jj < kernel_width; jj++)
				{
					int ii_src_bordered = ii + i_src_bordered - y_anchor;
					int jj_src_bordered = jj + j_src_bordered - x_anchor;

					suma = suma + (src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)] * alpha);
				}
			}

			dst_data[i_destino * dst_width + j_destino] =  (DataT)suma;

			os << "sumaR = " << suma.R << ", sumaG = " << suma.G << ", sumaB = " << suma.B << ", sumaA = " << suma.A << sycl::endl;
		});
	});
    

}