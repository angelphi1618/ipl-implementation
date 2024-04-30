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
	
	int kernel_height = spec.kernel_size;
	int kernel_width = spec.kernel_size;

	// image<DataT, AllocatorT>* bordered_image = generate_border(src, sycl::range<2>(kernel_width, kernel_height), border_type, default_value);
	

	return q.submit([&](sycl::handler& cgh) {

		cgh.depends_on(dependencies);

		pixel<DataT>* src_data = src.get_data();
		pixel<DataT>* dst_data = dst.get_data();

		int src_bordered_width = src.get_size().get(0);
		int src_bordered_height = src.get_size().get(1);
		
		int dst_width = dst.get_size().get(0);
		int dst_height = dst.get_size().get(1);

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

		int left_padding = x_anchor ;
		int right_padding = kernel_width - x_anchor;

		int top_padding = y_anchor;
		int bottom_padding = kernel_height - y_anchor;

		// Tamaño de la memoria local por cada work-group
		int slm_width  = left_padding + work_group_w + right_padding ;
		int slm_height = top_padding  + work_group_h + bottom_padding;

		sycl::accessor<pixel<DataT>, 1, sycl::access::mode::read_write, sycl::access::target::local> slm(slm_width*slm_height, cgh);

		ComputeT twice_sigma_d_sqrd = 2 * spec.sigma_distance * spec.sigma_distance;
		ComputeT twice_sigma_i_sqrd = 2 * spec.sigma_intensity * spec.sigma_intensity;

		cgh.parallel_for_work_group(global, local, [=](sycl::group<2> grp){

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

			grp.parallel_for_work_item([=](sycl::h_item<2> it) {
				int i_destino = grp.get_id(1) * work_group_h + it.get_local_id(1) ;
				int j_destino = grp.get_id(0) * work_group_w + it.get_local_id(0) ;

				j_destino = sycl::min<int>(j_destino, dst_width - 1);
				i_destino = sycl::min<int>(i_destino, dst_height - 1);

				int i_src_bordered = it.get_local_id(1); // + kernel_height;
				int j_src_bordered = it.get_local_id(0); // + kernel_width;

				pixel<DataT> I_ij = slm[(i_src_bordered + top_padding) * slm_width + j_src_bordered + left_padding];

				ComputeT sum_w = 0;
				ComputeT sum_Iw_R = 0;
				ComputeT sum_Iw_G = 0;
				ComputeT sum_Iw_B = 0;
				ComputeT sum_Iw_A = 0;

				for (int k = 0; k < kernel_height; k++)
				{
					for (int l = 0; l < kernel_width; l++)
					{
						int kk_src_bordered = k + i_src_bordered;
						int ll_src_bordered = l + j_src_bordered;

						pixel<DataT> I_kl = slm[(kk_src_bordered) * slm_width + ll_src_bordered];
						//bordered_pixel_dispatcher(border_type, src_data, kk_src_bordered, ll_src_bordered, src_bordered_width, src_bordered_height, default_value);

						ComputeT w = get_w(i_src_bordered + top_padding, j_src_bordered + left_padding, 
										  kk_src_bordered, ll_src_bordered, twice_sigma_d_sqrd, twice_sigma_i_sqrd, I_ij, I_kl);

						sum_w += w;
						sum_Iw_R += I_kl.R * w;
						sum_Iw_G += I_kl.G * w;
						sum_Iw_B += I_kl.B * w;
						sum_Iw_A += I_kl.A * w;
					}
				}

				dst_data[i_destino * dst_width + j_destino]
				//= I_ij; auto a 
				= {
					(DataT) (sum_Iw_R / sum_w),
					(DataT) (sum_Iw_G / sum_w),
					(DataT) (sum_Iw_B / sum_w),
					(DataT) (sum_Iw_A / sum_w),
				};

			});
		});
	});
}

template <typename ComputeT = float,
		typename DataT, typename AllocatorT>
sycl::event bilateral_filter_roi(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
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

		std::cout << "kernel copiado" << std::endl;

		//std::cout << typeid(*src.get_allocator()).name() << std::endl;

		pixel<DataT>* src_data = bordered_image->get_data();
		pixel<DataT>* dst_data = dst.get_data();

		int src_bordered_width = bordered_image->get_size().get(0);

		int dst_width = dst.get_size().get(0);

		// sycl::stream os(1024*1024, 1024, cgh);

		std::cout << "lanzando parallel for" << std::endl;
		// Tamaño de la imagen destino
		int x_anchor = (kernel_width - 1) / 2;
		int y_anchor = (kernel_height - 1) / 2;

		ComputeT twice_sigma_d_sqrd = 2 * spec.sigma_distance * spec.sigma_distance;
		ComputeT twice_sigma_i_sqrd = 2 * spec.sigma_intensity * spec.sigma_intensity;

		roi_rect roi = src.get_roi_rect();

		int src_bordered_height = bordered_image->get_size().get(1);

		q.memcpy(dst_data, src.get_data(), src.get_linear_size() * sizeof(pixel<DataT>)).wait();

		cgh.parallel_for(sycl::range<2>(roi.get_height(), roi.get_width()), [=](sycl::id<2> item){
			// os << "dentro del kernel bilateral" << sycl::endl;

			// os << "kernel usado" << sycl::endl;

			int i_destino = item.get(1) + roi.get_x_offset();
			int j_destino = item.get(0) + (src_bordered_height - roi.get_y_offset() - roi.get_height());

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


					//R = R + ((ComputeT)src_data[kk_src_bordered * src_bordered_width + (ll_src_bordered)].R * kernel_data[k * kernel_width + l]);
					//G = G + ((ComputeT)src_data[kk_src_bordered * src_bordered_width + (ll_src_bordered)].G * kernel_data[k * kernel_width + l]);
					//B = B + ((ComputeT)src_data[kk_src_bordered * src_bordered_width + (ll_src_bordered)].B * kernel_data[k * kernel_width + l]);
					//A = A + ((ComputeT)src_data[kk_src_bordered * src_bordered_width + (ll_src_bordered)].A * kernel_data[k * kernel_width + l]);
				}
			}
			
			// os << (int) sum_Iw_R << " / " << (int) sum_w << " = " << (int)(sum_Iw_R / sum_w) << sycl::endl;

			//os << "R=" << (int) (sum_Iw_R / sum_w) << sycl::endl;

			dst_data[i_destino * dst_width + j_destino] = {
				(DataT) (sum_Iw_R / sum_w),
				(DataT) (sum_Iw_G / sum_w),
				(DataT) (sum_Iw_B / sum_w),
				(DataT) (sum_Iw_A / sum_w),
			};
		});
	});
}