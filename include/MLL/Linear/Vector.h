#ifndef MLL_LINEAR_VECTOR_H_
#define MLL_LINEAR_VECTOR_H_

#include "MLL/Linear/Matrix.h"

namespace ml {
	class Vector : public Matrix
	{
	public:
		Vector();
		Vector(int size);
		Vector(int size, const std::vector<double>& data);
		Vector(const std::vector<double>& data);
		Vector(const ml::Matrix& matrix);
		Vector(const ml::Tensor& tensor);

		double operator()(int index) const;
		double& operator()(int index);

		Vector reshape(int size) const { return ml::Matrix::reshape(size, 1); }
		ml::Matrix transpose() const { return ml::Matrix(1, shape[0], data); }

		ml::Matrix operator*(const ml::Matrix& other) const { return ml::Matrix::operator*(other); }
		ml::Matrix& operator*=(const ml::Matrix& other) { ml::Matrix::operator*=(other); return *this; }
	};
}

#endif