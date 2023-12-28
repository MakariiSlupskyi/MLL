#ifndef MLL_LINEAR_MATRIX_H_
#define MLL_LINEAR_MATRIX_H_

#include "MLL/Linear/Tensor.h"

namespace ml {
	class Matrix : public Tensor
	{
	public:
		Matrix();
		Matrix(int columns, int rows);
		Matrix(int columns, int rows, const std::vector<double>& data);
		Matrix(const ml::Tensor& tensor);

		Matrix transpose() const;
		Matrix reshape(int columns, int rows) const;

		double operator()(int x, int y) const;
		double& operator()(int x, int y);

		Matrix operator*(const Matrix& other) const;
		Matrix& operator*=(const Matrix& other);

		Matrix operator*(double scalar) const;
		Matrix& operator*=(double scalar);
	};
}

#endif