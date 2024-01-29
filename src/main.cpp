#include "allocators/host_usm_allocator_t.h"
#include <CL/sycl.hpp>
#include <cstdint>


int main() {


	sycl::device dev;
	dev = sycl::device(sycl::cpu_selector());
	sycl::queue Q(dev);

	host_usm_allocator_t<uint8_t> loca(Q);

	uint8_t* puntero = loca.allocate(24);

	for (int i = 0; i < 24; i++)
		puntero[i] = i;

	for (int i = 0; i < 24; i++)
		printf("%d\n", puntero[i]);
	
	loca.deallocate(puntero);

}
