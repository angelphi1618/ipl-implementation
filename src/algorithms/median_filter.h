#pragma once

#include <CL/sycl.hpp>
#include "../pixel.h"
#include "../image.h"
#include "../border_generator/border_types.h"
#include "../border_generator/border_generator.h"

template<typename DataT>
void swap(pixel<DataT>* a, pixel<DataT>* b){
    pixel<DataT> temp = *a;
    *a = *b;
    *b = temp;
}

template< typename DataT>
int partition(pixel<DataT>* window, int left, int right){
    pixel<DataT> last = window[right];
    int i = left, j = left;
    while (j < right)
    {
        if(window[i] < last){
            swap(&window[i], &window[j]);
            i++;
        }
        j++;
    }
    swap(&window[i], &window[right]);
    return i;
}

template< typename DataT>
int randomP(pixel<DataT>* window, int left, int right){
    int subSize = right - left + 1;
    int pivot = rand() % subSize;
    swap(&window[left + pivot], &window[right]);
    return partition(window, left, right);
}


template< typename DataT>
void sort(pixel<DataT>* window, int left, int right, int k, pixel<DataT>& a, pixel<DataT>& b, bool& aUsed, bool& bUsed){
    
    if(left <= right){
        int indexP = randomP(window, left, right);
        
        if(indexP == k){
            b = window[indexP];
            bUsed = true;
            if(aUsed) return;
        }
        else if(indexP == k - 1){
            a = window[indexP];
            aUsed = true;
            if(bUsed) return;
        }

        if(indexP >= k)
            return sort(window, left, indexP - 1, k, a, b, aUsed, bUsed);
        return sort(window, indexP + 1, right, k, a, b, aUsed, bUsed);
    }
    return;
}


template< typename DataT>
pixel<DataT> getMedian(pixel<DataT>* window, int size){
    int median;
    bool aUsed = false, bUsed = false;
    pixel<DataT> a, b;

    sort(window, 0, size - 1, size / 2, a, b, aUsed, bUsed);
    return b;
}



struct median_spec
{
    int radius;
};


template <typename DataT, typename AllocatorT>
sycl::event median_filter(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
						const median_spec& kernel,
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
	

	image<DataT, AllocatorT>* bordered_image = generate_border<DataT, AllocatorT>(src, kernel.radius, border_type, default_value);

	

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
        int anchor = (kernel.radius - 1) / 2;
		cgh.parallel_for(dst.get_size(), [=](sycl::id<2> item){
			// os << "dentro del kernel" << sycl::endl;

			// os << "kernel usado" << sycl::endl;

			int i_destino = item.get(1);
			int j_destino = item.get(0);

			int i_src_bordered = i_destino + kernel.radius;
			int j_src_bordered = j_destino + kernel.radius;

            pixel<DataT>* window = src.get_allocator()->allocate(kernel.radius * kernel.radius);
			for (int ii = 0; ii < kernel.radius; ii++)
			{
				for (int jj = 0; jj < kernel.radius; jj++)
				{
					int ii_src_bordered = ii + i_src_bordered - anchor;
					int jj_src_bordered = jj + j_src_bordered - anchor;

                    window[ii * kernel.radius + jj] = src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)];
				}
			}

			dst_data[i_destino * dst_width + j_destino] = getMedian(window, kernel.radius * kernel.radius);

			// os << "sumaR = " << suma.R << ", sumaG = " << suma.G << ", sumaB = " << suma.B << ", sumaA = " << suma.A << sycl::endl;
		});
	});

}






