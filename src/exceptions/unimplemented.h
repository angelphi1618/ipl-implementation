#pragma once
#include <exception>

class unimplemented : public std::exception {
private:
	const char* mensaje;
public:
	unimplemented(const char* mensaje) : mensaje(mensaje) {}

	const char* what() const noexcept override {
		return mensaje;
	}
};