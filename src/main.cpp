#include "allocators/host_usm_allocator_t.h"
#include "allocators/device_usm_allocator_t.h"

#include <CL/sycl.hpp>
#include <cstdint>

#include "image.h"

int main() {

	sycl::device dev;
	dev = sycl::device(sycl::cpu_selector());
	sycl::queue Q(dev);

	device_usm_allocator_t<uint8_t> loca(Q);	

	std::cout << "Running on "
		<< Q.get_device().get_info<sycl::info::device::name>()
		<< std::endl;

	uint8_t* puntero = loca.allocate(1024);

	// for (int i = 0; i < 24; i++)
	// 	puntero[i] = i;

	sycl::range<1> numItems(1024);

	Q.submit([&](sycl::handler& handler) {
		handler.parallel_for(numItems, [=](sycl::id<1> idx) {
			puntero[idx] = idx;
		});
	}).wait();

	// for (int i = 0; i < 24; i++)
	// 	printf("%d\n", puntero[i]);
	
	loca.deallocate(puntero);

}
