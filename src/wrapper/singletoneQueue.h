#pragma once

#include <CL/sycl.hpp>

class SingletoneQueue{
	static sycl::queue* _Q;
public:
	static sycl::queue* getInstance() {
		if (_Q == nullptr) {
			sycl::device dev;
			// Intentamos seleccionar una gpu. Si no hay, escogemos el dispositivo por defecto
			// que probablemente sea la CPU
			try { dev = sycl::device(sycl::gpu_selector());}
			catch(const std::exception& e) {
				  dev = sycl::device(sycl::default_selector());
			}

			_Q = new sycl::queue(dev);
		}

		return _Q;
	}
};
sycl::queue* SingletoneQueue::_Q = nullptr;