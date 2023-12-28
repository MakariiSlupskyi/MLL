#include "MLL/Linear/Vector.h"
#include <stdexcept>

ml::Vector::Vector() : Matrix(1, 1)
{}

ml::Vector::Vector(int size) : Matrix(size, 1)
{}

ml::Vector::Vector(int size, const std::vector<double>& data) : Matrix(size, 1, data)
{}

ml::Vector::Vector(const std::vector<double>& data) : Matrix(data.size(), 1, data)
{}

ml::Vector::Vector(const ml::Matrix& matrix) : Matrix(matrix)
{
	if (shape[1] != 1) { throw std::invalid_argument("Matrix with invalid shape was provided whan creating vector."); }
}

ml::Vector::Vector(const ml::Tensor& tensor) : Matrix(tensor)
{
	if (shape[1] != 1) { throw std::invalid_argument("Matrix with invalid shape was provided whan creating vector."); }
}

double ml::Vector::operator()(int index) const { return data.at(index); }

double& ml::Vector::operator()(int index) { return data.at(index); }