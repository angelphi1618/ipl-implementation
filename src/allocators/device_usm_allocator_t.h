#include "base_allocator.h"
#pragma once


template <typename T>
class device_usm_allocator_t : public base_allocator<T>
{
public:
	device_usm_allocator_t(sycl::queue &q) : base_allocator<T>(q) {
		std::cout << typeid(this).name() << std::endl;
	}

	T* allocate(std::size_t n) override {
		return sycl::malloc_device<T>(n, this->queue);
	}

	void* allocate_bytes(std::size_t n) override {
		return sycl::malloc_device<T>(n, this->queue);
	}
};
