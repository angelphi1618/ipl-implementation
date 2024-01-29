#include <CL/sycl.hpp>
#include "allocators/host_usm_allocator_t.h"

template <typename DataT, typename AllocatorT = base_allocator<DataT>>
class image {
private:
	const sycl::range<2> size;
	DataT* data;
	sycl::queue &queue;
	
	base_allocator<DataT>* allocator;
	bool allocatorAllocated = false;

	const int defaultChannels = 3;
public:
	explicit image(sycl::queue &queue, const sycl::range<2> &image_size) : queue(queue), size(image_size) {
		this->allocator = new host_usm_allocator_t<DataT>(queue);
		this->allocatorAllocated = true;

		this->data = this->allocator.allocate(this->defaultChannels * image_size.size());
	}

	explicit image(sycl::queue &queue, const sycl::range<2> &image_size, AllocatorT allocator) 
	: queue(queue), size(image_size), allocator(&allocator) {
		this->data = this->allocator->allocate(this->defaultChannels * image_size.size());
	}

	~image() {
		this->allocator.deallocate(this->data);

		if (this->allocatorAllocated)
			delete this->allocator;
	}
	

	

};