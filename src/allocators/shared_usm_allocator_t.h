#include "base_allocator.h"
#pragma once

template <typename T>
class shared_usm_allocator_t : public base_allocator<T>
{
public:

	shared_usm_allocator_t(sycl::queue &q) : base_allocator<T>(q) {}

	T* allocate(std::size_t n) override {
		return sycl::malloc_shared<T>(n, this->queue);
	};

	void* allocate_bytes(std::size_t n) override {
		return sycl::malloc_shared<T>(n, this->queue);
	}
};