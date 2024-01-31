#include <CL/sycl.hpp>


struct roi_rect
{
	sycl::range<2> size, offset;

	inline explicit roi_rect(const sycl::range<2> size, const sycl::range<2> offset) : size(size), offset(offset) {}
	inline roi_rect(const sycl::range<2> size) : size(size), offset(0, 0) {}

	inline std::size_t get_x_offset() const { return offset.get(0); }
	inline std::size_t get_y_offset() const { return offset.get(1); }

	inline std::size_t get_width() const { return size.get(0); }
	inline std::size_t get_height() const { return size.get(1); }
};