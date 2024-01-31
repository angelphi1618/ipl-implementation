#include <CL/sycl.hpp>
#include <string>

template <typename DataT, typename AllocatorT = base_allocator<pixel<DataT>>>
class image_persistance {
protected:
	image<DataT, AllocatorT>* img;
	image_persistance(image<DataT, AllocatorT>& img) : img(&img) {} ;
public:

	virtual void loadImage(std::string path) = 0;
	virtual void saveImage(std::string path) = 0;
};