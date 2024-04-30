#pragma once

#include <CL/sycl.hpp>
#include "../pixel.h"
#include "../image.h"
#include "../border_generator/border_types.h"
#include "../border_generator/border_generator.h"
#include "../exceptions/unimplemented.h"
#include <vector>

struct box_filter_spec{
	sycl::range<2> kernel_size;
    int x_anchor;
    int y_anchor;

    inline box_filter_spec(sycl::range<2> kernel_size) : kernel_size(kernel_size) {
        this->x_anchor = (kernel_size.get(0) - 1) / 2;
        this->y_anchor = (kernel_size.get(1) - 1) / 2;
    }
};

unsigned int float4ToUint(sycl::float4 pixel, float scale);

template<typename DataT>
pixel<DataT> float4ToPixel(sycl::float4 pixel, float scale);

template<typename DataT>
sycl::float4 pixelToFloat4(pixel<DataT> p);

sycl::float4 uintToFloat4(unsigned int p);



template <typename ComputeT, typename DataT, typename AllocatorT>
sycl::event box_filter(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
						const box_filter_spec& kernel,
						border_types border_type = border_types::const_val,
						pixel<DataT> default_value = {},
						const std::vector<sycl::event>& dependencies = {}) {



    switch (border_type)
	{	
	case border_types::const_val:
	case border_types::repl:
    case border_types::mirror:
		break;
	
	default:
		throw unimplemented("Tipo de borde no soportado");
	}

	
	int kernel_width = kernel.kernel_size[0];
	int kernel_height = kernel.kernel_size[1];

	


		pixel<DataT>* src_data = src.get_data();
		pixel<DataT>* dst_data = dst.get_data();

		int src_bordered_width = src.get_size().get(0);
		int src_bordered_height = src.get_size().get(1);
		int image_height = src.get_size().get(1);
		int image_width = src.get_size().get(0);

		int dst_width = dst.get_size().get(0);

		

		unsigned int* medias = static_cast<unsigned int*>(src.get_allocator()->allocate_bytes(src.get_linear_size() *sizeof(unsigned int)));

		auto rowMean = q.submit([&](sycl::handler& cgh) {

			cgh.parallel_for(sycl::range<1>(image_height), [=](sycl::id<1> item){

				int y = item.get(0);
				int r = kernel_height / 2;
				float scale = 1.0f / (2 * r + 1);

			
				int cur = y*dst_width;
				sycl::float4 sum;

				pixel<DataT> aux = src_data[cur];

				sum = pixelToFloat4(aux) * r;

				//Calculamos la media del primer pixel de la fila
				for (int x = 0; x < r + 1; x++)
					sum += pixelToFloat4(src_data[cur + x]);

				medias[cur] = float4ToUint(sum, scale);		


				int right_index, left_index, k;
				for (int x = 1; x < dst_width; x++)
				{
					//K representa el pixel actual que estamos procesando 
					k = cur + x;

					right_index = sycl::min(k + r, cur + dst_width - 1);
					//Sumamos el nuevo pixel que entra al kernel
					sum += pixelToFloat4(src_data[right_index]);

					left_index = sycl::max(cur, k - r - 1);
					//Restamos el pixel que ha salido del kernel
					sum -= pixelToFloat4(src_data[left_index]);

					//Añadimos la media al vector intermedio de medias por filas
					medias[k] = float4ToUint(sum, scale);
				}
					
				
			});

		});

		
		return q.submit([&](sycl::handler& cgh) {
			
			//Esperamos la ejecución anterior
			cgh.depends_on(rowMean);
			//Calculamos ahora la medias de las columnas usando solo el vector "medias" calculadas anteriormente
			cgh.parallel_for(sycl::range<1>(image_width), [=](sycl::id<1> item){

				int y = item.get(0);
				int r = kernel_height / 2;
				float scale = 1.0f / (2 * r + 1);
				sycl::float4 sum;

				sum = uintToFloat4(medias[y]) * (float)r;

				//Calculamos la media del primer pixel de la columna
				for (int x = 0; x < r + 1; x++)
					sum += uintToFloat4(medias[x * dst_width + y]);
				
				dst_data[y] = float4ToPixel<DataT>(sum, scale);

				
				int top_index, bottom_index, k;
				for (int x = 1; x < image_height; x++)
				{
					k = x * dst_width + y;

					bottom_index = sycl::min((image_height - 1) * dst_width + y, k + dst_width * r);
					sum += uintToFloat4(medias[bottom_index]);

					top_index = sycl::max(k - (r * dst_width) - dst_width, y);
					sum -= uintToFloat4(medias[top_index]);

					dst_data[k] = float4ToPixel<DataT>(sum, scale);
				}
				

			});

		});
	
		
		
		
	
	

    

}



template <typename ComputeT, typename DataT, typename AllocatorT>
sycl::event box_filter_roi(sycl::queue& q, image<DataT, AllocatorT>& src, image<DataT, AllocatorT>& dst,
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
		throw unimplemented("Tipo de borde no soportado");
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

		roi_rect roi = src.get_roi_rect();

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

                    R += src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)].R * alpha;
                    G += src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)].G * alpha;
                    B += src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)].B * alpha;
                    A += src_data[ii_src_bordered * src_bordered_width + (jj_src_bordered)].A * alpha;
				}
			}

			dst_data[i_destino * dst_width + j_destino] = {(DataT) R, (DataT) G, (DataT) B, (DataT)A };

			//os << "sumaR = " << R << ", sumaG = " << G << ", sumaB = " << B << sycl::endl;
		});
	});
    

}

sycl::float4 uintToFloat4(unsigned int p){

	sycl::float4 f;
	f.x() = (p >> 0) & 0xFF;
	f.y() = (p >> 8) & 0xFF;
	f.z() = (p >> 16) & 0xFF;
	f.w() = (p >> 24) & 0xFF;
	return f;
	
}

unsigned int float4ToUint(sycl::float4 pixel, float scale){
	unsigned int aux = 0U;
	aux |= 0x000000FF & (unsigned int)(pixel.x() * scale);
	aux |= 0x0000FF00 & (((unsigned int)(pixel.y() * scale)) << 8);
	aux |= 0x00FF0000 & (((unsigned int)(pixel.z() * scale)) << 16);
	aux |= 0xFF000000 & (((unsigned int)(pixel.w() * scale)) << 24);
	return aux;
}

template<typename DataT>
pixel<DataT> float4ToPixel(sycl::float4 p, float scale){
	pixel<DataT> aux;

	aux.R = (DataT) (p.x() * scale);
	aux.G = (DataT) (p.y() * scale);
	aux.B = (DataT) (p.z() * scale);
	aux.A = (DataT) (p.w()* scale);	
	return aux;
}

template<typename DataT>
sycl::float4 pixelToFloat4(pixel<DataT> p){

	sycl::float4 f;
	f.x() = static_cast<float>(p.R);
	f.y() = static_cast<float>(p.G);
	f.z() = static_cast<float>(p.B);
	f.w() = static_cast<float>(p.A);
	return f;
	
}