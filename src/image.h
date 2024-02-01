#include <CL/sycl.hpp>
#include "allocators/host_usm_allocator_t.h"
#include "pixel.h"
#include "exceptions/invalid_argument.h"
#include "roi_rect.h"

#pragma once

template <typename DataT, typename AllocatorT = base_allocator<pixel<DataT>>>
class image {
private:
	const sycl::range<2> size;
	pixel<DataT>* data;
	sycl::queue* queue;

	base_allocator<pixel<DataT>>* allocator = nullptr;
	bool allocatorAllocated = false;

	const int defaultChannels = 4;
public:
	explicit image(sycl::queue &queue, const sycl::range<2> &image_size) : queue(&queue), size(image_size) {
		this->allocator = new host_usm_allocator_t<pixel<DataT>>(queue);
		this->allocatorAllocated = true;

		this->data = this->allocator->allocate(this->defaultChannels * image_size.size());
	}

	explicit image(sycl::queue &queue, const sycl::range<2> &image_size, AllocatorT allocator) 
	: queue(&queue), size(image_size), allocator(&allocator) {
		this->data = this->allocator->allocate(this->defaultChannels * image_size.size());
	}

	explicit image(sycl::queue &queue, pixel<DataT> *image_data, const sycl::range<2> &image_size, const std::vector<sycl::event> &dependencies = {})
	: queue(&queue), size(image_size), data(image_data) {

		if (image_data == nullptr)
			throw invalid_argument("image_data no puede ser null.");

		// Esperamos a que acaben todos los eventos de los que se nos indica dependencia
		this->queue.submit([&](sycl::handler& cgh) {
			cgh.depends_on(dependencies);
		}).wait();
	}

	explicit image(sycl::queue &queue, pixel<DataT> *image_data, const sycl::range<2> &image_size, AllocatorT allocator, const std::vector<sycl::event> &dependencies = {})
	: queue(&queue), size(image_size), allocator(&allocator), data(image_data) {

		if (image_data == nullptr)
			throw invalid_argument("image_data no puede ser null.");

		// Esperamos a que acaben todos los eventos de los que se nos indica dependencia
		this->queue.submit([&](sycl::handler& cgh) {
			cgh.depends_on(dependencies);
		}).wait();
	}

	image<DataT, AllocatorT>* get_roi(roi_rect rect) const {
		image<DataT, AllocatorT>* roi_image = new image(*this->queue, rect._size);

		pixel<DataT>* roi_image_data = roi_image->get_data();
		pixel<DataT>* current_image_data = this->data;
		int width = this->size.get(0);
		int height = this->size.get(1);

		std::cout << width << std::endl;

		this->queue->submit([&](sycl::handler& cgh) {
			cgh.parallel_for(sycl::range<2>(rect.get_width(), rect.get_height()), [=](sycl::id<2> idx){
				
				// algo pasa
				int i_origen = (rect.get_x_offset() + idx[0]) + (idx[1] + (height - rect.get_height())) * width;

				//ok
				int i_destino = idx[0] + idx[1] * rect.get_width();

				roi_image_data[i_destino] = current_image_data[i_origen];
			});
		}).wait();

		return roi_image;
	}

	std::size_t get_size() const {
		return this->size.size();
	}

	sycl::range<2> get_size() {
		return this->size;
	}	

	base_allocator<pixel<DataT>>* get_allocator() const {
		return this->allocator;
	}

	pixel<DataT>* get_data() {
		return this->data;
	}

	sycl::queue* get_queue(){
		return this->queue;
	}

	~image() {
		if (this->allocator != nullptr)
			this->allocator->deallocate(this->data);

		if (this->allocatorAllocated)
			delete this->allocator;
	}
};