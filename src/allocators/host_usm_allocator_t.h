#include <cstdint>
#include <CL/sycl.hpp>

template <typename T>
class host_usm_allocator_t
{
private:
	sycl::queue queue;
public:
	host_usm_allocator_t(sycl::queue &q) : queue(q) {};
	~host_usm_allocator_t();
	T* allocate(std::size_t n);
	void deallocate(T* p);
};

template <typename T>
T* host_usm_allocator_t<T>::allocate(std::size_t n) {
	return sycl::malloc_host<T>(n, this->queue);
}

template <typename T>
void host_usm_allocator_t<T>::deallocate(T* p) {
	sycl::free((void*) p, this->queue);
}

template <typename T>
host_usm_allocator_t<T>::~host_usm_allocator_t(){
	return;
}