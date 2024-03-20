#pragma once

#include <CL/sycl.hpp>
#include "../pixel.h"
#include "../image.h"
#include "../border_generator/border_types.h"
#include "../border_generator/border_generator.h"
#include "../exceptions/unimplemented.h"
#include <cmath>

template <typename ComputeT = float>
struct bilateral_filter_spec{
	unsigned int kernel_size;
	ComputeT sigma_intensity;
	ComputeT sigma_distance;

	bilateral_filter_spec(unsigned int kernel_size, ComputeT sigma_intensity, ComputeT sigma_distance)
	: kernel_size(kernel_size), sigma_intensity(sigma_intensity), sigma_distance(sigma_distance) {}
};

template <typename ComputeT = float, typename DataT>
inline ComputeT get_w(int i, int j, int k, int l, ComputeT twice_sigma_d_sqrd, ComputeT twice_sigma_i_sqrd, pixel<DataT>I_ij, pixel<DataT> I_kl){
	
	// L1 norm
	ComputeT first_term = (ComputeT) ((i - k)*(i - k) + (j - l)*(j - l));
	first_term = first_term / twice_sigma_d_sqrd;

	// L1 norm
	ComputeT second_term = static_cast<ComputeT>((I_ij.R - I_kl.R) + (I_ij.G - I_kl.G) + (I_ij.B - I_kl.B) + (I_ij.A - I_kl.A));
	second_term = second_term * second_term;
	second_term = second_term / twice_sigma_i_sqrd;

	return exp(0.0 - first_term - second_term);
}

template <typename ComputeT = float,
		typename DataT, typename AllocatorT>
sycl::event bilateral_filter(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
						const bilateral_filter_spec<ComputeT>& spec,
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
	
	int kernel_height = spec.kernel_size;
	int kernel_width = spec.kernel_size;

	image<DataT, AllocatorT>* bordered_image = generate_border(src, sycl::range<2>(kernel_width, kernel_height), border_type, default_value);
	

	return q.submit([&](sycl::handler& cgh) {

		cgh.depends_on(dependencies);

		pixel<DataT>* src_data = bordered_image->get_data();
		pixel<DataT>* dst_data = dst.get_data();

		int src_bordered_width = bordered_image->get_size().get(0);
		int dst_width = dst.get_size().get(0);

		// Tamaño de la imagen destino
		int x_anchor = (kernel_width - 1) / 2;
		int y_anchor = (kernel_height - 1) / 2;

		ComputeT twice_sigma_d_sqrd = 2 * spec.sigma_distance * spec.sigma_distance;
		ComputeT twice_sigma_i_sqrd = 2 * spec.sigma_intensity * spec.sigma_intensity;

		cgh.parallel_for(dst.get_size(), [=](sycl::id<2> item){
			int i_destino = item.get(1);
			int j_destino = item.get(0);

			int i_src_bordered = i_destino + kernel_height;
			int j_src_bordered = j_destino + kernel_width;

			pixel<DataT> I_ij = src_data[i_src_bordered * src_bordered_width + j_src_bordered];

			ComputeT sum_w = 0;
			ComputeT sum_Iw_R = 0;
			ComputeT sum_Iw_G = 0;
			ComputeT sum_Iw_B = 0;
			ComputeT sum_Iw_A = 0;

			for (int k = 0; k < kernel_height; k++)
			{
				for (int l = 0; l < kernel_width; l++)
				{
					int kk_src_bordered = k + i_src_bordered - y_anchor;
					int ll_src_bordered = l + j_src_bordered - x_anchor;

					pixel<DataT> I_kl = src_data[kk_src_bordered * src_bordered_width + (ll_src_bordered)];

					ComputeT w = get_w(i_src_bordered, j_src_bordered, kk_src_bordered, ll_src_bordered, twice_sigma_d_sqrd, twice_sigma_i_sqrd, I_ij, I_kl);

					sum_w += w;
					sum_Iw_R += I_kl.R * w;
					sum_Iw_G += I_kl.G * w;
					sum_Iw_B += I_kl.B * w;
					sum_Iw_A += I_kl.A * w;
				}
			}
			dst_data[i_destino * dst_width + j_destino] = {
				(DataT) (sum_Iw_R / sum_w),
				(DataT) (sum_Iw_G / sum_w),
				(DataT) (sum_Iw_B / sum_w),
				(DataT) (sum_Iw_A / sum_w),
			};
		});
	});
}