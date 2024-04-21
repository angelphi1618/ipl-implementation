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
		
	
	inline bool operator<(const pixel<DataT>& other) const {
		return value() < other.value();
	}

	inline bool operator>(const pixel<DataT>& other) const {
		return value() > other.value();
	}

	inline float value() {
		return 0.299 * R + 0.587 * G + 0.114 * B;
	}
};

template<typename DataT, typename ComputeT>
pixel<DataT> operator*(pixel<DataT>& input_pixel, ComputeT scalar) {
	return {
		static_cast<DataT> (input_pixel.R * scalar),
		static_cast<DataT> (input_pixel.G * scalar),
		static_cast<DataT> (input_pixel.B * scalar),
		static_cast<DataT> (input_pixel.A * scalar)
	};
}

template<typename DataT>
pixel<DataT> operator+(pixel<DataT> p1, pixel<DataT> p2) {
	return {
		static_cast<DataT>(p1.R + p2.R),
		static_cast<DataT>(p1.G + p2.G),
		static_cast<DataT>(p1.B + p2.B),
		static_cast<DataT>(p1.A + p2.A)
	};
}


