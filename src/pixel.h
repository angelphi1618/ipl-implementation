#pragma once

#include <cstdint>
template <typename DataT = uint8_t>
struct pixel{
	DataT R;
	DataT G;
	DataT B;
	DataT A;
};