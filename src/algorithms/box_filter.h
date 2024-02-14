#pragma once

#include <CL/sycl.hpp>
#include "../pixel.h"
#include "../image.h"
#include "../border_generator/border_types.h"
#include "../border_generator/border_generator.h"


struct box_filter_spec{
	sycl::range<2> kernel_size;
    int x_anchor;
    int y_anchor;

    inline box_filter_spec(sycl::range<2> kernel_size) : kernel_size(kernel_size) {
        this->x_anchor = (kernel_size.get(0) - 1) / 2;
        this->y_anchor = (kernel_size.get(1) - 1) / 2;
    }
};


