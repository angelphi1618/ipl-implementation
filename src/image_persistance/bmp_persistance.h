#include "../image.h"
#include "image_persistance.h"

#include <string>

#include <fstream>
#include <iostream>

#include <CL/sycl.hpp>

#pragma once

#pragma pack(push, 1) // Para que esté continua en memoria igual que en disco
struct BMPHeader {
	uint16_t signature;
	uint32_t fileSize; // <-- Only this
	uint32_t reserved;
	uint32_t dataOffset; // ... and this
	uint32_t headerSize;
	uint32_t width;
	uint32_t height;
	uint16_t planes;
	uint16_t bitsPerPixel;
	uint32_t compression;
	uint32_t dataSize;
	uint32_t horizontalRes;
	uint32_t verticalRes;
	uint32_t colors;
	uint32_t importantColors;
};
#pragma pack(pop)

template <typename DataT, typename AllocatorT = base_allocator<pixel<DataT>>>
class bmp_persistance : public image_persistance<DataT, AllocatorT>
{
private:
	BMPHeader header;
	static BMPHeader generate_header(int width, int height);
public:

	bmp_persistance(image<DataT, AllocatorT>& img) : image_persistance<DataT, AllocatorT>(img) {}

	void loadImage(std::string path){
		sycl::queue* Q = this->img->get_queue();
		
		std::ifstream file(path, std::ios::binary);

		// Leemos el header del fichero
		file.read(reinterpret_cast<char*>(&header), sizeof(BMPHeader));

		// Reservamos en ram el espacio necesario
		std::vector<uint8_t> image_pixels(header.fileSize - header.dataOffset);

		// Leemos todo el contenido de la imagen
		file.seekg(header.dataOffset, std::ios::beg);
		file.read(reinterpret_cast<char*>(image_pixels.data()), image_pixels.size());

		file.close();

		// Reservamos en memoria del dispositivo espacio para la imagen lineal
		uint8_t* image_device = sycl::malloc_device<uint8_t>(header.fileSize - header.dataOffset, *Q);

		// Copiamos la imagen en ram al dispositivo
		Q->memcpy(image_device, image_pixels.data(), header.fileSize - header.dataOffset).wait();

		pixel<DataT>* data = this->img->get_data();
		Q->submit([&](sycl::handler& cgh) {
			// Convertimos la imagen lineal en un array de pixeles
			cgh.parallel_for(sycl::range<1>(image_pixels.size()), [=](sycl::id<1> index) {
				switch (index % 3)
				{
				case 0:
					data[index / 3].R = image_device[index];
					break;
				case 1:
					data[index / 3].G = image_device[index];
					break;
				case 2:
					data[index / 3].B = image_device[index];
					break;
				default:
					break;
				}
				data[index / 3].A = 255;
			});
		}).wait();

		sycl::free((void*)image_device, *Q);
	}

	void saveImage(std::string dest_path) {

		std::cout << "guardando imagen" << std::endl;

		sycl::queue* Q = this->img->get_queue();

		// Reservamos espacio en dispositivo para generar la imagen lineal
		uint8_t* image_device = sycl::malloc_device<uint8_t>(header.fileSize - header.dataOffset, *Q);

		pixel<DataT>* data = this->img->get_data();

		Q->submit([&](sycl::handler& cgh) {
			cgh.parallel_for(sycl::range<1>(this->header.width * this->header.height), [=](sycl::id<1> index) {
				// pixeles_device pixeles en gpu
				// image_device lineal en gpu
				image_device[index * 3] =  data[index].R;
				image_device[index * 3 + 1] =  data[index].G;
				image_device[index * 3 + 2] =  data[index].B;
			});
		}).wait();

		std::vector<uint8_t> output(this->header.width * this->header.height * 3);

		// Copiamos la imagen lineal del dispositivo a ram
		Q->memcpy(output.data(), image_device, this->header.width * this->header.height * 3).wait();

		std::ofstream outFile(dest_path, std::ios::binary);
		if (!outFile.is_open()) {
			std::cerr << "Failed to open output file!" << std::endl;
			return;
		}

		outFile.write(reinterpret_cast<char*>(&header), sizeof(BMPHeader));

		// Write the pixel data to the output file
		outFile.write(reinterpret_cast<char*>(output.data()), output.size());

		// Close the output file
		outFile.close();
		sycl::free((void*)image_device, *Q);

		std::cout << "BMP file copied successfully!" << std::endl;
	}

	static void saveImage(image<DataT, AllocatorT>& img, std::string dest_path) {
		// Debemos generar el header primero

		int width = img.get_size().get(0);
		int heigth = img.get_size().get(1);

		BMPHeader header = generate_header(width, heigth);

		sycl::queue* Q = img.get_queue();

		// Reservamos espacio en dispositivo para generar la imagen lineal
		uint8_t* image_device = sycl::malloc_device<uint8_t>(header.fileSize - header.dataOffset, *Q);

		pixel<DataT>* data = img.get_data();

		Q->submit([&](sycl::handler& cgh) {
			cgh.parallel_for(sycl::range<1>(header.width * header.height), [=](sycl::id<1> index) {
				// pixeles_device pixeles en gpu
				// image_device lineal en gpu
				image_device[index * 3] =  data[index].R;
				image_device[index * 3 + 1] =  data[index].G;
				image_device[index * 3 + 2] =  data[index].B;
			});
		}).wait();

		std::vector<uint8_t> output(header.width * header.height * 3);

		// Copiamos la imagen lineal del dispositivo a ram
		Q->memcpy(output.data(), image_device, header.width * header.height * 3).wait();

		std::ofstream outFile(dest_path, std::ios::binary);
		if (!outFile.is_open()) {
			std::cerr << "Failed to open output file!" << std::endl;
			return;
		}

		outFile.write(reinterpret_cast<char*>(&header), sizeof(BMPHeader));

		// Write the pixel data to the output file
		outFile.write(reinterpret_cast<char*>(output.data()), output.size());

		// Close the output file
		outFile.close();
		sycl::free((void*)image_device, *Q);

		std::cout << "BMP file copied successfully!" << std::endl;
	}

	~bmp_persistance()
	{
	};
};

template <typename DataT, typename AllocatorT>
BMPHeader bmp_persistance<DataT, AllocatorT>::generate_header(int width, int height){
		BMPHeader header;

		header.signature = 19778;
		header.fileSize = width * height * 3 + 54;
		header.reserved = 0;
		header.dataOffset = 54;
		header.headerSize = 40;
		header.width = width;
		header.height = height;
		header.planes = 1;
		header.bitsPerPixel = 24;
		header.compression = 0;
		header.dataSize = width * height * 3;
		header.horizontalRes = 3780;
		header.verticalRes = 3780;
		header.colors = 0;
		header.importantColors = 0;

		return header;
}