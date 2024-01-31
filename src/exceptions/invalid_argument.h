#include <exception>

class invalid_argument : public std::exception {
private:
	const char* mensaje;
public:
	invalid_argument(const char* mensaje) : mensaje(mensaje) {}

	const char* what() const noexcept override {
		return mensaje;
	}
};