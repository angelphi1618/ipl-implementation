#include "base_allocator.h"
#pragma once

template <typename T>
class host_usm_allocator_t : public base_allocator<T>
{
public:
	
	host_usm_allocator_t(sycl::queue &q) : base_allocator<T>(q) {}

	T* allocate(std::size_t n) override {
		return sycl::malloc_host<T>(n, this->queue);
	}
};