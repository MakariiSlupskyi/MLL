#ifndef MLL_LINEAR_TENSOR_H_
#define MLL_LINEAR_TENSOR_H_

#include <vector>
#include <string>

namespace ml {
	class Tensor
	{
	public:
		Tensor();
		Tensor(const std::vector<int>& shape);
		Tensor(const std::vector<int>& shape, const std::vector<double>& data);

		std::vector<int> getShape() const { return shape; }
		std::vector<double> getData() const { return data; }
		int getDataSize() const { return dataSize; }

		Tensor chip(int axisInd, int index) const;
		Tensor slice(const std::vector<int>& indices) const;
		Tensor block(const std::vector<int>& start, const std::vector<int>& shape) const;
		Tensor reverse() const;
		Tensor reshape(const std::vector<int> shape) const;
		
		double sum() const;
		double max() const;
		double min() const;
		double average() const;

		Tensor& setValues(const std::vector<double>& values);
		Tensor& setConstant(double scalar);
		Tensor& setRandom(double scatter = 1);

		Tensor& setChip(int axisInd, int index, const Tensor& other);
		Tensor& setSlice(const std::vector<int>& indices, const Tensor& other);
		Tensor& setBlock(const std::vector<int>& start, const Tensor& block);

		Tensor applyFunc(double (*func)(double)) const;
		Tensor& applyFunc(double (*func)(double));

		double operator()(std::vector<int> indices) const;
		double& operator()(std::vector<int> indices);

		Tensor operator-() const;

		bool operator==(const Tensor& other) const;
		bool operator!=(const Tensor& other) const;

		Tensor operator+(const Tensor& other) const;
		Tensor operator-(const Tensor& other) const;
		Tensor operator*(const Tensor& other) const;
		Tensor operator/(const Tensor& other) const;

		Tensor& operator+=(const Tensor& other);
		Tensor& operator-=(const Tensor& other);
		Tensor& operator*=(const Tensor& other);
		Tensor& operator/=(const Tensor& other);

		Tensor operator+(double scalar) const;
		Tensor operator-(double scalar) const;
		Tensor operator*(double scalar) const;
		Tensor operator/(double scalar) const;

		Tensor& operator+=(double scalar);
		Tensor& operator-=(double scalar);
		Tensor& operator*=(double scalar);
		Tensor& operator/=(double scalar);

	protected:
		std::vector<int> shape;
		int dataSize;
		std::vector<double> data;
	
	private:
		int calcDataIndex(const std::vector<int>& indices) const;
		std::vector<int>& increaseIndices(std::vector<int>& indices) const;
		void checkForNan(const std::string& message);
	};
}

#endif