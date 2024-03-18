// myLib.hpp

#ifndef MYLIB_HPP
#define MYLIB_HPP

// Include all necessary headers
#include "allocators/host_usm_allocator_t.h"
#include "allocators/device_usm_allocator_t.h"
#include "allocators/shared_usm_allocator_t.h"
#include "allocators/base_allocator.h"
#include "image_persistance/bmp_persistance.h"
#include "image_persistance/image_persistance.h"
#include "image_persistance/png_persistance.h"
#include "exceptions/invalid_argument.h"
#include "exceptions/unimplemented.h"
#include "roi_rect.h"
#include "algorithms/grayscale.h"
#include "algorithms/gaussian_filter.h"
#include "algorithms/filter_convolution.h"
#include "algorithms/sobel_filter.h"
#include "algorithms/separable_filter.h"
#include "algorithms/bilateral_filter.h"
#include "algorithms/box_filter.h"
#include "border_generator/border_generator.h"
#include "border_generator/border_types.h"
#include "image.h"
#include "pixel.h"

namespace myLib {
    // You may not need to define anything explicitly here
	template<typename DataT, typename AllocatorT = host_usm_allocator_t<pixel<char>>>
	using Image = ::image<DataT, AllocatorT>;
	// class image {
	// 	public:
	// 		explicit image(sycl::queue &queue, const sycl::range<2> &image_size) {
	// 			return image<DataT, AllocatorT>(queue, image_size);
	// 		};
	// 		image<DataT, AllocatorT>* get_roi() const;

	// 		image<DataT, AllocatorT>* get_roi(roi_rect rect) const ;

	// 		std::size_t get_linear_size() const ;

	// 		sycl::range<2> get_size() ;

	// 		base_allocator<pixel<DataT>>* get_allocator() const ;

	// 		pixel<DataT>* get_data() ;

	// 		sycl::queue* get_queue();

	// 		roi_rect get_roi_rect() const;

	// 		~image();
	// };
}

#endif // MYLIB_HPP
