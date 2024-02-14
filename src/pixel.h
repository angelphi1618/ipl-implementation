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

	template<typename ComputeT>
	pixel(const pixel<ComputeT>& other) : 
		
		R(static_cast<ComputeT> (other.R)),
		G(static_cast<ComputeT> (other.G)),
		B(static_cast<ComputeT> (other.B)),
		A(static_cast<ComputeT> (other.A)) {}
		
	

};

template<typename DataT, typename ComputeT>
pixel<DataT> operator*(pixel<DataT>& input_pixel, ComputeT scalar) {
	return {
		static_cast<DataT> (input_pixel.R * scalar),
		static_cast<DataT> (input_pixel.G * scalar),
		static_cast<DataT> (input_pixel.B * scalar),
		static_cast<DataT> (input_pixel.A)
	};
}

template<typename DataT>
pixel<DataT> operator+(pixel<DataT>& p1, pixel<DataT> p2) {
	return {
		static_cast<DataT>(p1.R + p2.R),
		static_cast<DataT>(p1.G + p2.G),
		static_cast<DataT>(p1.B + p2.B),
		static_cast<DataT>(p1.A)
	};
}
