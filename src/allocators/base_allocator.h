#include <CL/sycl.hpp>
#pragma once

template <typename T>
class base_allocator
{
protected:
	sycl::queue queue;
	base_allocator(sycl::queue &q) : queue(q) {};
public:
	
	~base_allocator() { return; };
	
	virtual T* allocate(std::size_t n) = 0;

	void deallocate(T* p) {
		sycl::free((void*) p, this->queue);
	};
};