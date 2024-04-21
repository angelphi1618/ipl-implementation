#include <CL/sycl.hpp>
#include "../image.h"
#include "border_types.h"

#pragma once

template<typename DataT, typename AllocatorT>
image<DataT, AllocatorT>* generate_border(image<DataT, AllocatorT>& img, sycl::range<2> borderSize, border_types borderType, pixel<DataT> value = {}){

    switch (borderType)
    {
    case border_types::default_val:
        value = {DataT(),DataT(), DataT(), DataT()};
    case border_types::const_val:
        return plain_border(img, borderSize, value);
    case border_types::repl:
		return repl_border(img, borderSize);
    default:
        break;
    }
}

template<typename DataT>
inline pixel<DataT> bordered_pixel_dispatcher(border_types borderType, pixel<DataT>* src, int i, int j, int w, int h, pixel<DataT> value = {}) {
	switch (borderType)
    {
    case border_types::default_val:
        value = {DataT(),DataT(), DataT(), DataT()};
    case border_types::const_val:
        return bordered_pixel_plain(src, i, j, w, h, value);
    case border_types::repl:
		return bordered_pixel_repl(src, i, j, w, h);
    default:
        break;
    }
}

template<typename DataT>
inline pixel<DataT> bordered_pixel_repl(pixel<DataT>* src, int i, int j, int w, int h) {
	int low_bound_width  = 0;
	int low_bound_height = 0;

	int upper_bound_width  = w;
	int upper_bound_height = h;
	
	// Nunca nos salimos de los bordes reales de la imagen. Nos mantenemos dentro siempre.
	int i_src = sycl::max<int>(sycl::min<int>(upper_bound_height - 1, i), low_bound_height);
	int j_src = sycl::max<int>(sycl::min<int>(upper_bound_width  - 1, j), low_bound_width);


	return src[i_src*w + j_src];
}

template<typename DataT>
inline pixel<DataT> bordered_pixel_plain(pixel<DataT>* src, int i, int j, int w, int h, pixel<DataT> value) {
	DataT fuera = static_cast<DataT>(i < 0 || i >= h || j < 0 || j >= w); // 0 ó 1
	
	int low_bound_width  = 0;
	int low_bound_height = 0;

	int upper_bound_width  = w;
	int upper_bound_height = h;

	// Ajustamos i y j a los límites aceptables de la imagen
	int i_src = sycl::max<int>(sycl::min<int>(upper_bound_height - 1, i), low_bound_height);
	int j_src = sycl::max<int>(sycl::min<int>(upper_bound_width  - 1, j), low_bound_width);

	return value * fuera + (src[i*w + j] * (1 - fuera)); // Escogemos uno u otro
}

template<typename DataT, typename AllocatorT>
image<DataT, AllocatorT>* plain_border(image<DataT, AllocatorT>& img, sycl::range<2> borderSize, pixel<DataT> value){

    sycl::queue* Q = img.get_queue();

    int originalSizeW = img.get_size().get(0), originalSizeH = img.get_size().get(1); 

    int newSizeW = originalSizeW + 2 * borderSize.get(0);
    int newSizeH = originalSizeH + 2 * borderSize.get(1);

    sycl::range<2> r = sycl::range<2>(newSizeW, newSizeH);

    image<DataT, AllocatorT>* borderedImage = new image<DataT, AllocatorT>(*Q, r);

    image<DataT, AllocatorT>* imgCopy = &img;
    
    pixel<DataT>* pixelImg = img.get_data();
    pixel<DataT>* pixelBorderedImg = borderedImage->get_data();

    std::cout << "Creando Borde" << std::endl;
    std::cout << "w = " << newSizeW << " h = " << newSizeH << std::endl;

    Q->submit([&](sycl::handler& cgh){
        cgh.parallel_for(sycl::range<2>(newSizeW, newSizeH), [=](sycl::id<2>indexBordered){


            int iOriginal = indexBordered[0] - borderSize.get(0);
            int jOriginal = indexBordered[1] - borderSize.get(1);
            
            //Estas en el borde
            if(iOriginal < 0 || iOriginal >= originalSizeW || jOriginal < 0 || jOriginal >= originalSizeH){
                pixelBorderedImg[indexBordered[0] + indexBordered[1] * newSizeW] = value;
            }
            else{
                int indexOriginalArray = (indexBordered[1] - borderSize.get(1)) * originalSizeW + indexBordered[0] - borderSize.get(0);
                int i = indexBordered[0] + indexBordered[1] * newSizeW;
                pixelBorderedImg[i] = pixelImg[indexOriginalArray];
            }

        });
    }).wait();

    std::cout << "Borde creado" << std::endl;

    return borderedImage;

}

template<typename DataT, typename AllocatorT>// TODO: LLAMAR A ESTO FILTRO MÁGICO
image<DataT, AllocatorT>* repl_border(image<DataT, AllocatorT>& img, sycl::range<2> borderSize){

	sycl::queue* Q = img.get_queue();

	int originalSizeW = img.get_size().get(0), originalSizeH = img.get_size().get(1); 

	int newSizeW = originalSizeW + 2 * borderSize.get(0);
	int newSizeH = originalSizeH + 2 * borderSize.get(1);

	sycl::range<2> r = sycl::range<2>(newSizeW, newSizeH);

	image<DataT, AllocatorT>* borderedImage = new image<DataT, AllocatorT>(*Q, r);

	image<DataT, AllocatorT>* imgCopy = &img;
	
	pixel<DataT>* pixelImg = img.get_data();
	pixel<DataT>* pixelBorderedImg = borderedImage->get_data();

	std::cout << "Creando Borde" << std::endl;
	// std::cout << "w = " << newSizeW << " h = " << newSizeH << std::endl;
	

	Q->submit([&](sycl::handler& cgh){

		int offset_x = borderSize.get(0);
		int offset_y = borderSize.get(1);

		sycl::stream os(1024*1024*1024, 1024, cgh);


		cgh.parallel_for(sycl::range<2>(newSizeW, newSizeH), [=](sycl::id<2>indexBordered){
			int i_destino = sycl::max<int>(sycl::min<int>(offset_x + originalSizeW - 1 , indexBordered.get(0)), offset_x);
			int j_destino = sycl::max<int>(sycl::min<int>(offset_y + originalSizeH - 1 , indexBordered.get(1)), offset_y);

			//os << "i = " << i_destino << ", j = " << j_destino << "i2 = " << indexBordered.get(0) << ", j2 = " << indexBordered.get(1) << sycl::endl;

			int indexOriginalArray = (j_destino - borderSize.get(1)) * originalSizeW + (i_destino - borderSize.get(0));
			int i = indexBordered[0] + indexBordered[1] * newSizeW;

			pixelBorderedImg[i] = pixelImg[indexOriginalArray];
		});
	}).wait();

	std::cout << "Borde creado" << std::endl;

	return borderedImage;

}