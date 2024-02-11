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
        break;
        
    default:
        break;
    }

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