#include <CL/sycl.hpp>
#include "../image.h"

#pragma once

template<typename DataT, typename AllocatorT>
image<DataT, AllocatorT>* border_default(image<DataT, AllocatorT>& img, int borderSize){

    sycl::queue* Q = img.get_queue();

    int originalSizeW = img.get_size().get(0), originalSizeH = img.get_size().get(1); 

    int newSizeW = originalSizeW + 2 * borderSize;
    int newSizeH = originalSizeH + 2 * borderSize;

    sycl::range<2> r = sycl::range<2>(newSizeW, newSizeH);

    image<DataT, AllocatorT>* borderedImage = new image<DataT, AllocatorT>(*Q, r);
    
    pixel<DataT>* pixelImg = img.get_data();
    pixel<DataT>* pixelBorderedImg = borderedImage->get_data();

    std::cout << "Creando Borde" << std::endl;
    std::cout << "w = " << newSizeW << " h = " << newSizeH << std::endl;
    

    Q->submit([&](sycl::handler& cgh){
            sycl::stream os(1024*1024, 1024, cgh);
        cgh.parallel_for(sycl::range<2>(newSizeW, newSizeH), [=](sycl::id<2>indexBordered){


            int iOriginal = indexBordered[0] - borderSize;
            int jOriginal = indexBordered[1] - borderSize;
            pixel<DataT> value;
            
            //Estas en el borde
            if(iOriginal < 0 || iOriginal >= originalSizeW || jOriginal < 0 || jOriginal >= originalSizeH){
                value = {0,0,0,0};
                pixelBorderedImg[indexBordered[0] + indexBordered[1] * newSizeW].R = 0;
                pixelBorderedImg[indexBordered[0] + indexBordered[1] * newSizeW].G = 170;
                pixelBorderedImg[indexBordered[0] + indexBordered[1] * newSizeW].B = 228;
                pixelBorderedImg[indexBordered[0] + indexBordered[1] * newSizeW].A = 255;
            }
            else{
                //int indexOriginalArray = - (indexBordered[0] - borderSize) + (indexBordered[1] - borderSize) * newSizeW - (indexBordered[1] - borderSize) * borderSize;
                int indexOriginalArray = (indexBordered[1] - borderSize) * originalSizeW + indexBordered[0] - borderSize;
                int i = indexBordered[0] + indexBordered[1] * newSizeW;
                pixelBorderedImg[i] = pixelImg[indexOriginalArray];
                if (indexOriginalArray >= 160000){

                os << ((indexBordered[0] - borderSize)) << " ";
                os << (((indexBordered[1] - borderSize) * newSizeW)) << " ";
                os << (((indexBordered[1] - borderSize) * borderSize)) << sycl::endl;
                }
                //pixelBorderedImg[indexBordered[0] + indexBordered[1] * newSizeW] = pixelImg[indexOriginalArray];
            }

        });
    }).wait();

    std::cout << "Borde creado" << std::endl;

    return borderedImage;

}