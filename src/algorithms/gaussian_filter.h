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

	return filter_convolution(q, src, dst, kernel, border_type, default_value, dependencies);
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