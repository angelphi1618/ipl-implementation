#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "../image.h"
#include "image_persistance.h"

#include <iostream>

#include <CL/sycl.hpp>

#pragma once

template <typename DataT, typename AllocatorT>
class png_persistance : public image_persistance<DataT, AllocatorT>{

    private:
        int height;
        int width;
        int channels;
        uint8_t* localData = NULL;

    public:

        ~png_persistance(){
            stbi_image_free(localData);
        }

        png_persistance(image<DataT, AllocatorT>& img) : image_persistance<DataT, AllocatorT>(img) {}

        void loadImage(std::string path){

            
            //c_str transforma un string en un const char*
            //Esta funcion le asigna el w, h y channels a la clase
            localData = stbi_load(path.c_str(), &width, &height, &channels, 0);

            if(localData == NULL){
                std::cout << "No se ha podido cargar la imagen" << std::endl;
                return;
            }

            std::cout << "w = " << width << " h = " << height << " channels = " << channels << std::endl;
            
            
            sycl::queue* Q = this->img->get_queue();

            uint8_t* image_device = sycl::malloc_device<uint8_t>(height * width * channels, *Q);

            Q->memcpy(image_device, localData, height * width * channels).wait();


            pixel<DataT>* data = this->img->get_data();
            Q->submit([&](sycl::handler& cgh){
                cgh.parallel_for(sycl::range<1>(height * width * channels), [=](sycl::id<1> index){
                    //En vez de 4 deberian ser channels
                    switch (index % 4)
				{
				case 0:
					data[index / 4].R = image_device[index];
					break;
				case 1:
					data[index / 4].G = image_device[index];
					break;
				case 2:
					data[index / 4].B = image_device[index];
					break;
                case 3:
                    data[index / 4].A = image_device[index];
				default:
					break;
				}
                });
            }).wait();

            sycl::free((void*)image_device, *Q);

            std::cout << "Imagen cargada" << std::endl;

        }

        void saveImage(std::string dest_path) {


            sycl::queue* Q = this->img->get_queue();

            uint8_t* image_device = sycl::malloc_device<uint8_t>(height * width * channels, *Q);

            pixel<DataT>* data = this->img->get_data();

            Q->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(height * width), [=](sycl::id<1> index) {
                    image_device[index * 4] =  data[index].R;
                    image_device[index * 4 + 1] =  data[index].G;
                    image_device[index * 4 + 2] =  data[index].B;
                    image_device[index * 4 + 3] = data[index].A;
                });
            }).wait();

            std::cout << "Convirtiendo imagen en array" << std::endl;

            std::vector<uint8_t> output(height * width * channels);

            Q->memcpy(output.data(), image_device, height * width * channels).wait();

            int s = stbi_write_png(dest_path.c_str(), width, height, channels, output.data(), width * channels);
            if (s == 0)
                std::cout << "No se ha podido guardar la imagen" << std::endl;

            sycl::free((void*)image_device, *Q);
        }

        static void saveImage(image<DataT, AllocatorT>& img, std::string dest_path){
            sycl::queue* Q = img.get_queue();
            int w = img.get_size().get(0);
            int h = img.get_size().get(1);

            std::cout << "h = " << h << " w = " << w << std::endl;

            uint8_t* image_device = sycl::malloc_device<uint8_t>(w * h * 4, *Q);
            pixel<DataT>* pixelData = img.get_data();
            Q->submit([&](sycl::handler& cgh){
                cgh.parallel_for(sycl::range<1>(w * h), [=](sycl::id<1>index){
                    image_device[index * 4] = pixelData[index].R;
                    image_device[index * 4 + 1] = pixelData[index].G;
                    image_device[index * 4 + 2] = pixelData[index].B;
                    image_device[index * 4 + 3] = pixelData[index].A;
                });
            }).wait();

            std::vector<uint8_t> local(w * h * 4);

            Q->memcpy(local.data(), image_device, w * h * 4).wait();

            int s = stbi_write_png(dest_path.c_str(), w, h, 4, local.data(), w * 4);
            if(s == 0)
                std::cout << "No se ha podido guardar la imagen" << std::endl;

            sycl::free((void*)image_device, *Q);

        }


};