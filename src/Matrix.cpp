#include "MLL/Linear/Matrix.h"
#include <cmath>
#include <stdexcept>

ml::Matrix::Matrix() : Tensor({1, 1})
{}

ml::Matrix::Matrix(int columns, int rows) : Tensor({columns, rows})
{}

ml::Matrix::Matrix(int columns, int rows, const std::vector<double>& data)
	: Tensor({columns, rows}, data)
{}

ml::Matrix::Matrix(const ml::Tensor& tensor) : Tensor(tensor)
{
	if (shape.size() == 1) {
		shape.push_back(1);
	} else if (shape.size() != 2) {
		throw std::invalid_argument("Tensor with invalid shape was provided whan creating matrix.");
	}
}

ml::Matrix ml::Matrix::transpose() const {
	ml::Matrix res(shape[1], shape[0]);
	for (int i = 0; i < shape[0]; ++i) {
		for (int j = 0; j < shape[1]; ++j) {
			res(j, i) = this->operator()(i, j);
		}
	}
	return res;
}

ml::Matrix ml::Matrix::reshape(int columns, int rows) const {
	return Tensor::reshape({columns, rows});
}

double ml::Matrix::operator()(int x, int y) const {
	return data.at(x * shape.at(1) + y);
}

double& ml::Matrix::operator()(int x, int y) {
	return data.at(x * shape.at(1) + y);
}

ml::Matrix ml::Matrix::operator*(const ml::Matrix& other) const {
	if (shape[1] != other.getShape()[0]) {
		throw std::invalid_argument("Invalid shape of given matrix for multiplication.");
	}
	Matrix res(shape[0], other.getShape()[1]);
	for (int i = 0; i < res.getShape()[0]; ++i) {
		for (int j = 0; j < res.getShape()[1]; ++j) {
			double tmp = 0.0f, t;
			for (int k = 0; k < shape[1]; ++k) {
				t = this->operator()(i, k) * other(k, j);
				tmp += (std::isnan(t) || std::isinf(t)) ? 0 : t;
			}
			res(i, j) = tmp; 
		}
	}
	return res;
}

ml::Matrix& ml::Matrix::operator*=(const ml::Matrix& other) {
	if (shape[1] != other.getShape()[0]) {
		throw std::invalid_argument("Invalid shape of given matrix for multiplication.");
	}
	for (int i = 0; i < shape[0]; ++i) {
		for (int j = 0; j < other.getShape()[1]; ++j) {
			double tmp = 0.0f;
			for (int k = 0; k < shape[1]; ++k) { tmp += this->operator()(i, k) * other(k, j); }
			this->operator()(i, j) = tmp;
		}
	}
	return *this;
}

ml::Matrix ml::Matrix::operator*(double scalar) const {
	return ml::Tensor::operator*(scalar);
}

ml::Matrix& ml::Matrix::operator*=(double scalar) {
	Tensor::operator*=(scalar);
	return *this;
}