#pragma once

#include <CL/sycl.hpp>
#include "../pixel.h"
#include "../image.h"
#include "../border_generator/border_types.h"
#include "../border_generator/border_generator.h"
#include "../exceptions/unimplemented.h"
#include "../image_persistance/bmp_persistance.h"

template<typename ComputeT>
struct separable_spec
{
    sycl::range<2> window;
    ComputeT* kernel_x_ptr;
    ComputeT* kernel_y_ptr;
};

template <typename ComputeT = int,
		typename DataT, typename AllocatorT>
sycl::event separable_filter(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
						const separable_spec<ComputeT>& kernel,
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


    image<DataT, AllocatorT>* bordered_image = generate_border(src, {kernel.window.get(1), kernel.window.get(0)}, border_type, default_value);
    image<DataT, AllocatorT>* intermediate = generate_border(src, {kernel.window.get(1), kernel.window.get(0)}, border_type, default_value);

    pixel<DataT>* intermediate_data = intermediate->get_data();
    pixel<DataT>* src_data = bordered_image->get_data();
    pixel<DataT>* dst_data = dst.get_data();

    int bordered_width = bordered_image->get_size().get(0);
    int anchor_y = (kernel.window.get(0) - 1) / 2;
    int anchor_x = (kernel.window.get(1) - 1) / 2;
    int dst_width = dst.get_size().get(0);
    int inter_width = intermediate->get_size().get(0);

    ComputeT* kernel_x = static_cast<ComputeT*>(src.get_allocator()->allocate_bytes(kernel.window.get(1)*sizeof(ComputeT)));
    ComputeT* kernel_y = static_cast<ComputeT*>(src.get_allocator()->allocate_bytes(kernel.window.get(0) *sizeof(ComputeT)));

    q.memcpy(kernel_x, kernel.kernel_x_ptr, kernel.window.get(1) *sizeof(ComputeT)).wait();
    q.memcpy(kernel_y, kernel.kernel_y_ptr, kernel.window.get(0) *sizeof(ComputeT)).wait();

    q.submit([&](sycl::handler& cgh) {

		cgh.depends_on(dependencies);

		// Row
        cgh.parallel_for(dst.get_size(), [=](sycl::id<2> item){

            ComputeT R = 0;     
            ComputeT G = 0;                 
            ComputeT B = 0;     
            ComputeT A = 0;

            int i_dst = item.get(1);
            int j_dst = item.get(0);
            
            int j_bordered = item.get(0) + kernel.window.get(0);
			int i_bordered = i_dst + kernel.window.get(1);

            for (int j = 0; j < kernel.window.get(1); j++)
            {
                int jj_bordered = j + j_bordered - anchor_x;

                R = R + ((ComputeT)src_data[i_bordered * bordered_width + (jj_bordered)].R * kernel_x[j]);
                G = G + ((ComputeT)src_data[i_bordered * bordered_width + (jj_bordered)].G * kernel_x[j]);
                B = B + ((ComputeT)src_data[i_bordered * bordered_width + (jj_bordered)].B * kernel_x[j]);
                A = A + ((ComputeT)src_data[i_bordered * bordered_width + (jj_bordered)].A * kernel_x[j]);
            }

            R /= kernel.window.get(1);
            G /= kernel.window.get(1);
            B /= kernel.window.get(1);
            A /= kernel.window.get(1);
            
            intermediate_data[(i_dst + kernel.window.get(0)) * inter_width + j_dst + kernel.window.get(1)] = {
				(DataT) R,
				(DataT) G,
				(DataT) B,
				(DataT) A,
			};

        });
    }).wait();

    return q.submit([&](sycl::handler& cgh) {

        cgh.parallel_for(dst.get_size(), [=](sycl::id<2> item){

            ComputeT R = 0;     
            ComputeT G = 0;                 
            ComputeT B = 0;     
            ComputeT A = 0;

            int i_dst = item.get(1);
            int j_dst = item.get(0);
            int j = item.get(0) + kernel.window.get(1);

            for (int i = 0; i < kernel.window.get(0); i++)
            {
                int i_bordered = item.get(1) + i - anchor_y + kernel.window.get(0);

                R = R + ((ComputeT)intermediate_data[i_bordered * bordered_width + (j)].R * kernel_y[i]);
                G = G + ((ComputeT)intermediate_data[i_bordered * bordered_width + (j)].G * kernel_y[i]);
                B = B + ((ComputeT)intermediate_data[i_bordered * bordered_width + (j)].B * kernel_y[i]);
                A = A + ((ComputeT)intermediate_data[i_bordered * bordered_width + (j)].A * kernel_y[i]);
            }

            R /= kernel.window.get(0);
            G /= kernel.window.get(0);
            B /= kernel.window.get(0);
            A /= kernel.window.get(0);

            
            
            dst_data[i_dst * dst_width + j_dst] = {
				(DataT) R,
				(DataT) G,
				(DataT) B,
				(DataT) A,
			};

        });

    });


    

}
