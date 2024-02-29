#include "allocators/host_usm_allocator_t.h"
#include "allocators/device_usm_allocator_t.h"

#include <CL/sycl.hpp>
#include <cstdint>

#include "image.h"
#include "image_persistance/bmp_persistance.h"
#include "roi_rect.h"
#include "image_persistance/png_persistance.h"
#include "border_generator/border_generator.h"
#include "border_generator/border_types.h"

#include "algorithms/grayscale.h"
#include "algorithms/filter_convolution.h"
#include "algorithms/box_filter.h"
#include "algorithms/gaussian_filter.h"
#include "algorithms/separable_filter.h"

#include "algorithms/sobel_filter.h"

int main() {

	sycl::device dev;
	dev = sycl::device(sycl::cpu_selector());
	sycl::queue Q(dev);

	device_usm_allocator_t<pixel<uint8_t>> loca(Q);
	
	
	// imagendata = loca.allocate(...)
	// ev1 = Q.submit(...)
	// ev2 = Q.submit(...)
	// depencees (ev1, ev2)

	// imagen(imagendata, dependeces) 


	std::cout << "Running on "
		<< Q.get_device().get_info<sycl::info::device::name>()
		<< std::endl;

	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagen(Q, sycl::range(1200, 900), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenLena(Q, sycl::range(512, 512), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> lenaSeparada(Q, sycl::range(512, 512), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenLenaOutput(Q, sycl::range(512, 512), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenLenaGris(Q, sycl::range(512, 512), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenLenaSobel(Q, sycl::range(512, 512), loca);


	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenLenaOutputBox(Q, sycl::range(512, 512), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> lolitaEnLaCaja(Q, sycl::range(1200, 900), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> lolitaGaussiana(Q, sycl::range(1200, 900), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> lolitaGris(Q, sycl::range(1200, 900), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> lolitaSobel(Q, sycl::range(1200, 900), loca);

	//median_spec median = {5};
	//image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenLenaMediana(Q, sycl::range(512, 512), loca);



	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imageloader(imagen);
	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imageloaderLena(imagenLena);


	imageloader.loadImage("images/lolita.bmp");
	imageloaderLena.loadImage("images/prueba.bmp");

	//auto aaa = median_filter(Q, imagenLena, imagenLenaMediana, median);

	//aaa.wait();
	//bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> ::saveImage(imagenLenaMediana, "images/lenaMediana.bmp");

	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>* lolitaBorder = generate_border(imagen, {52, 70}, border_types::const_val, {0, 0, 255, 255});
	imageloader.saveImage("images/lolitaLocal.bmp");
	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> ::saveImage(*lolitaBorder, "images/lolitaConBorde.bmp");
	imageloader.saveImage("images/lolita2.bmp");
	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> ::saveImage(imagen, "images/lolitaDesdeFuera.bmp");

	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenPika(Q, sycl::range(400,400), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenPikaGauss(Q, sycl::range(400,400), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenPikaGris(Q, sycl::range(400,400), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenPikaSobel(Q, sycl::range(400,400), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenPikaSobelProcesada(Q, sycl::range(400,400), loca);

	std::vector<int> ptr_y = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1,};
	std::vector<int> ptr_x = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1,};
	separable_spec<int> separada = {{20, 20}, ptr_x.data() , ptr_y.data()};
	separable_filter(Q, imagenLena, lenaSeparada, separada, border_types::repl).wait();
	std::cout << "Despues del 2 parallel" << std::endl;
	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> ::saveImage(lenaSeparada, "images/lenaseparada.bmp");


	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> pikaAtrapada(Q, sycl::range(400,400), loca);


	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>* lolitaBorderRepl = generate_border(imagen, {52, 70}, border_types::repl);
	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> ::saveImage(*lolitaBorderRepl, "images/lolitaConBordeRepl.bmp");

	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>* lenitaBorderRepl = generate_border(imagenLena, {100, 50}, border_types::repl);
	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> ::saveImage(*lenitaBorderRepl, "images/lenita.bmp");

	std::vector<int> kernel2{1, 0 -1,
							   2, 0, -2,
							   1, 0, -1};

	

	// std::vector<float> kernel2{ 1,  4,  7,  4, 1,
	// 							4, 20, 33, 20, 4,
	// 							7, 33, 55, 33, 7,
	// 							4, 20, 33, 20, 4,
	// 							1,  4,  7,  4, 1};

	std::vector<float> kernel1;

	for (auto& el : kernel2){
		kernel1.push_back(el);
	}

	float* kernel = device_usm_allocator_t<float>(Q).allocate(kernel1.size());

	Q.memcpy(kernel, kernel1.data(), kernel1.size() * sizeof(float));

	filter_convolution_spec<int> kernel_spec({3 ,3}, kernel2.data(),1, 1);

	

	std::cout << "filtrado convolucion " << std::endl;
	filter_convolution<int>(Q, imagenLena, imagenLenaOutput, kernel_spec, border_types::repl).wait();
	std::cout << "filtrado convolucion ok" << std::endl;

	box_filter_spec box_spec({30, 30});

	box_filter<float>(Q, imagen, lolitaEnLaCaja, box_spec, border_types::repl).wait();
	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> ::saveImage(lolitaEnLaCaja, "images/lolitaEnLaAtrapadaaa.bmp");

	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> ::saveImage(imagenLenaOutput, "images/lenitaFiltrada.bmp");

	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> png(imagenPika);
	png.loadImage("images/pika.png");

	box_filter<float>(Q, imagenPika, pikaAtrapada, box_spec, border_types::repl).wait();

	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> ::saveImage(pikaAtrapada, "images/pikaAtrapadaEnLaCaja.png");
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>* imagenConBorder = generate_border(imagenPika, {50, 20}, border_types::const_val, {0, 255, 0, 255});
	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(*imagenConBorder, "images/pikaConBorde.png");
	roi_rect rectangulo(sycl::range<2>(200,200), sycl::range<2>(40,60));
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>* imagenPikaRecortada = imagenPika.get_roi(rectangulo);

	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(*imagenPikaRecortada, "images/pikaRecortada.png");

	png.saveImage("images/pika3.png");


	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> ye(Q, sycl::range(1222, 2088), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> yeGris(Q, sycl::range(1222, 2088), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> yeSobel(Q, sycl::range(1222, 2088), loca);

	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> yeAtrapada(Q, sycl::range(1222, 2088), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> ye_filtrado(Q, sycl::range(1222, 2088), loca);
	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> ye_gaussiano(Q, sycl::range(1222, 2088), loca);

	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> png_ye(ye);


	png_ye.loadImage("images/ye.png");
	std::cout << "ye cargado" << std::endl;

	//filter_convolution<float>(Q, ye, ye_filtrado, kernel_spec, border_types::repl).wait();
	box_filter<float>(Q, ye, yeAtrapada, box_spec, border_types::repl).wait();
	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(yeAtrapada, "images/yeAtrapadaaa.png");


	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>* ye_borde = generate_border(ye, {1000, 500}, border_types::repl, {255,0,0,255});
	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(*ye_borde, "images/yeborde.png");
	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(ye_filtrado, "images/yefiltrado.png");



	gaussian_filter_spec<double> gaussian_spec(10, 1.4, 1.4);
	gaussian_filter<double>(Q, imagen, lolitaGaussiana, gaussian_spec, border_types::const_val, {255,128,0,255}).wait();
	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(lolitaGaussiana, "images/gauss/ye_gaussiano.bmp");
	std::cout << "--------------------------------------------" << std::endl;
	

	std::cout << "sobel" << std::endl;

	gaussian_filter<double>(Q, ye, ye_gaussiano, gaussian_spec, border_types::repl).wait();
	rgb_to_gray(Q, ye_gaussiano, yeGris).wait();
	sobel_filter(Q, yeGris, yeSobel, {3}, border_types::repl).wait();
	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(yeSobel, "images/gauss/yeSobel.png");
	std::cout << "--------------------------------------------" << std::endl;


	std::cout << "sobel" << std::endl;

	gaussian_filter<double>(Q, imagenPika, imagenPikaGauss, gaussian_spec, border_types::repl).wait();
	rgb_to_gray(Q, imagenPikaGauss, imagenPikaGris).wait();
	sobel_filter(Q, imagenPikaGris, imagenPikaSobel, {3}, border_types::repl).wait();
	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(imagenPikaGris, "images/gauss/pikaGris.png");

	png_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(imagenPikaSobel, "images/gauss/pikaSobel.png");
	std::cout << "--------------------------------------------" << std::endl;

	std::cout << "sobel" << std::endl;

	gaussian_filter<double>(Q, imagen, lolitaGaussiana, gaussian_spec, border_types::repl).wait();
	rgb_to_gray(Q, lolitaGaussiana, lolitaGris).wait();
	sobel_filter(Q, lolitaGris, lolitaSobel, {3}, border_types::repl).wait();
	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(lolitaSobel, "images/gauss/lolitaSobel.bmp");
	std::cout << "--------------------------------------------" << std::endl;


	/*

	image<uint8_t> imagen2(Q, sycl::range(1200, 900));
	bmp_persistance<uint8_t> imageloader2(imagen2);

	imageloader2.loadImage("lolita.bmp");
	imageloader2.saveImage("lolita3.bmp");

	bmp_persistance<uint8_t>::saveImage(imagen2, "lolitaestatica.bmp");

	roi_rect rectangulo(sycl::range<2>(300,300), sycl::range<2>(496,60));


	//image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>* imagen3 = imagen.get_roi({sycl::range<2>(500,500), sycl::range<2>(0, 0)});

	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>* imagen3 = imagen.get_roi(rectangulo);
	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>>::saveImage(*imagen3, "lolitaroi.bmp");


	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenColor(Q, sycl::range(1200, 900), loca);
	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imageloaderColor(imagenColor);
	imageloaderColor.loadImage("lolita.bmp");

	image<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> imagenGris(Q, sycl::range(1200, 900), loca);
	std::cout << "imagen cargada" << std::endl;

	rgb_to_gray_roi(Q, imagenColor, imagenGris).wait();

	std::cout << "A guardar la imagen" << std::endl;

	bmp_persistance<uint8_t, device_usm_allocator_t<pixel<uint8_t>>> ::saveImage(imagenGris, "lolitagris.bmp");

	std::cout<<"hola" << std::endl;
	*/


}
