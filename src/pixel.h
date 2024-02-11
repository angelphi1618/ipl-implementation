#pragma once

#include <cstdint>
template <typename DataT = uint8_t>
struct pixel{
	DataT R;
	DataT G;
	DataT B;
	DataT A;

	inline pixel(DataT R, DataT G, DataT B, DataT A): R(R), G(G), B(B), A(A) {};
	inline pixel(DataT R, DataT G, DataT B): R(R), G(G), B(B), A(255) {};
	inline pixel(): R(0), G(0), B(0), A(255) {};
};