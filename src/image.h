#include <CL/sycl.hpp>
#include "allocators/host_usm_allocator_t.h"
#include "pixel.h"

#pragma once

template <typename DataT, typename AllocatorT = base_allocator<pixel<DataT>>>
class image {
private:
	const sycl::range<2> size;
	pixel<DataT>* data;
	sycl::queue* queue;
	
	base_allocator<pixel<DataT>>* allocator;
	bool allocatorAllocated = false;

	const int defaultChannels = 3;
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

	std::size_t get_size() const {
		return this->size.size();
	}

	base_allocator<pixel<DataT>> get_allocator() const {
		return this->allocator;
	}

	pixel<DataT>* get_data() {
		return this->data;
	}

	sycl::queue* get_queue(){
		return this->queue;
	}

	~image() {
		this->allocator->deallocate(this->data);

		if (this->allocatorAllocated)
			delete this->allocator;
	}
};