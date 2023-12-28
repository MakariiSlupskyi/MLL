#include "MLL/Linear/Tensor.h"
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <cmath>

ml::Tensor::Tensor() : shape({1}), dataSize(1), data(1)
{}

ml::Tensor::Tensor(const std::vector<int>& shape) : shape(shape), dataSize(1)
{
	for (int elem : shape) { dataSize *= elem; }
	data.resize(dataSize);
}

ml::Tensor::Tensor(const std::vector<int>& shape, const std::vector<double>& data) 
	: shape(shape), dataSize(1), data(data)
{
	for (int elem : shape) { dataSize *= elem; }
	if (data.size() != dataSize) {
		throw std::invalid_argument("Data with invalid size was provided when creating tensor.");
	}
}

ml::Tensor ml::Tensor::chip(int axisInd, int index) const {
	if (axisInd > shape.size() || index > shape[axisInd]) { throw std::invalid_argument("Invalid indices for chipping tensor."); }

	std::vector<int> resShape(shape);
	resShape.erase(resShape.begin() + axisInd);
	ml::Tensor res(resShape);
	std::vector<int> resInds(shape.size() - 1, 0), thisInds(shape.size(), 0);
	
	thisInds[axisInd] = index;
	for (int i = 0; i < res.dataSize; ++i) {
		for (int j = 0; j < resInds.size(); ++j) { thisInds.at((j < axisInd) ? j : j + 1) = resInds.at(j); }
		res(resInds) = this->operator()(thisInds);
		res.increaseIndices(resInds);
	}
	return res;
}

ml::Tensor ml::Tensor::slice(const std::vector<int>& indices) const {
	if (indices.size() >= shape.size()) { throw std::invalid_argument("Invalid indices for slicing tensor."); }

	if (indices.size() == 0) {
		return *this;
	} else {
		std::vector<int> indices_(indices.cbegin() + 1, indices.cend());
		return (this->chip(0, indices.at(0))).slice(indices_);
	}
}

ml::Tensor ml::Tensor::block(const std::vector<int>& start, const std::vector<int>& blockShape) const {
	if (start.size() != shape.size() || blockShape.size() != shape.size()) {
		throw std::invalid_argument("Invalid arguments for getting block of a tensor.");
	}

	Tensor res(blockShape);
	std::vector<int> resInds(shape.size()), thisInds(shape.size());
	for (int i = 0; i < res.dataSize; ++i) {
		for (int j = 0; j < blockShape.size(); ++j) {
			thisInds.at(j) = resInds.at(j) + start.at(j);
		}
		res(resInds) = this->operator()(thisInds);
		res.increaseIndices(resInds);
	}
	return res;
}

ml::Tensor ml::Tensor::reverse() const {
	ml::Tensor res(*this);
	std::reverse(res.data.begin(), res.data.end());
	return res;
}

ml::Tensor ml::Tensor::reshape(const std::vector<int> shape) const {
	ml::Tensor res(shape);
	auto data = this->data;
	data.resize(res.getDataSize());
	res.setValues(data);
	return res;
}

double ml::Tensor::sum() const {
	double res = 0.0;
	for (double elem : data) { res += elem; }
	return res;
}

double ml::Tensor::max() const {
	return *std::max_element(data.begin(), data.end());
}

double ml::Tensor::min() const {
	return *std::min_element(data.begin(), data.end());
}

double ml::Tensor::average() const {
	return this->sum() / data.size();
}

ml::Tensor& ml::Tensor::setValues(const std::vector<double>& values) {
	for (int i = 0; i < dataSize; ++i) {
		data.at(i) = values.at(i);
	}
	return *this;
}

ml::Tensor& ml::Tensor::setConstant(double scalar) {
	for (double& elem : data) { elem = scalar; }
	return *this;
}

ml::Tensor& ml::Tensor::setRandom(double scatter) {
	for (int i = 0; i < dataSize; ++i) {
		data.at(i) = (std::rand() % 1000 - 500) / 1000.0f * scatter;
	}
	return *this;
}

ml::Tensor& ml::Tensor::setChip(int axisInd, int index, const Tensor& other) {
	std::vector<int> thisInds(shape.size(), 0), otherInds(shape.size() - 1, 0);
	thisInds[axisInd] = index;
	for (int i = 0; i < other.dataSize; ++i) {
		for (int j = 0; j < otherInds.size(); ++j) { thisInds.at((j < axisInd) ? j : j + 1) = otherInds.at(j); }
		this->operator()(thisInds) = other(otherInds);
		other.increaseIndices(otherInds);
	}
	checkForNan("setting chip");
	return *this;
}

ml::Tensor& ml::Tensor::setSlice(const std::vector<int>& indices, const Tensor& other) {
	if (indices.size() == 0) {
		this->data = other.data;
	} else {
		std::vector<int> indices_(indices.cbegin() + 1, indices.cend());
		this->setChip(0, indices[0], this->chip(0, indices[0]).setSlice(indices_, other));
	}
	checkForNan("setting slice");
	return *this;
}

ml::Tensor& ml::Tensor::setBlock(const std::vector<int>& start, const Tensor& block) {
	std::vector<int> thisInds(shape.size(), 0), blockInds(shape.size(), 0);
	for (int i = 0; i < block.dataSize; ++i) {
		for (int j = 0; j < thisInds.size(); ++j) { thisInds.at(j) = start.at(j) + blockInds.at(j); }
		this->operator()(thisInds) = block(blockInds);
		block.increaseIndices(blockInds);
	}
	checkForNan("setting block");
	return *this;
}

ml::Tensor ml::Tensor::applyFunc(double (*func)(double)) const {
	ml::Tensor res = *this;
	std::transform(res.data.begin(), res.data.end(), res.data.begin(), func);
	res.checkForNan("applying function 1");
	return res;
}

ml::Tensor& ml::Tensor::applyFunc(double (*func)(double)) {
	std::transform(data.begin(), data.end(), data.begin(), func);
	checkForNan("applying function 2");
	return *this;
}

double ml::Tensor::operator()(std::vector<int> indices) const {
	return data.at(calcDataIndex(indices));

}

double& ml::Tensor::operator()(std::vector<int> indices) {
	return data.at(calcDataIndex(indices));
}

ml::Tensor ml::Tensor::operator-() const {
	ml::Tensor res(*this);
	std::transform(res.data.begin(), res.data.end(), res.data.begin(), [](double x) -> double { return -x; } );
	res.checkForNan("minus unary operator");
	return res;
}

bool ml::Tensor::operator==(const Tensor& other) const {
	if (shape.size() != other.shape.size()) { return false; }
	for (int i = 0; i < shape.size(); ++i) {
		if (shape[i] != other.shape[i]) { return false; }
	}
	return true;
}

bool ml::Tensor::operator!=(const Tensor& other) const {
	return !(this->operator==(other));
}

ml::Tensor ml::Tensor::operator+(const Tensor& other) const {
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid shape of given tensor for addition."); }
	ml::Tensor res(*this);
	for(int i = 0; i < dataSize; ++i) { res.data.at(i) += other.data.at(i); }
	res.checkForNan("adding tensor");
	return res;
}

ml::Tensor ml::Tensor::operator-(const Tensor& other) const {
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid shape of given tensor for subtraction."); }
	ml::Tensor res(*this);
	for(int i = 0; i < dataSize; ++i) { res.data.at(i) -= other.data.at(i); }
	res.checkForNan("subtracting tensor");
	return res;
}

ml::Tensor ml::Tensor::operator*(const Tensor& other) const {
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid shape of given tensor for multiplication."); }
	ml::Tensor res(*this);
	for(int i = 0; i < dataSize; ++i) {
		res.data[i] *= other.data[i];
		if (std::isnan(res.data[i]) || std::isinf(res.data[i])) { res.data[i] = 0; }
	}
	res.checkForNan("multiplying by tensor");
	return res;
}

ml::Tensor ml::Tensor::operator/(const Tensor& other) const {
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid shape of given tensor for division."); }
	ml::Tensor res(*this);
	for(int i = 0; i < dataSize; ++i) {
		res.data.at(i) = (other.data.at(i) == 0) ? 1000.0f : res.data.at(i) / other.data.at(i);
	}
	res.checkForNan("deviding by tensor");
	return res;
}

ml::Tensor& ml::Tensor::operator+=(const Tensor& other) {
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid shape of given tensor for addition."); }
	for(int i = 0; i < dataSize; ++i) { data.at(i) += other.data.at(i); }
	checkForNan("+= other tensor");
	return *this;
}

ml::Tensor& ml::Tensor::operator-=(const Tensor& other) {
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid shape of given tensor for subtraction."); }
	for(int i = 0; i < dataSize; ++i) { data.at(i) -= other.data.at(i); }
	checkForNan("-= other tensor");
	return *this;
}

ml::Tensor& ml::Tensor::operator*=(const Tensor& other) {
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid shape of given tensor for multiplication."); }
	for(int i = 0; i < dataSize; ++i) { data.at(i) *= other.data.at(i); }
	checkForNan("*= other tensor");
	return *this;
}

ml::Tensor& ml::Tensor::operator/=(const Tensor& other) {
	if (this->operator!=(other)) { throw std::invalid_argument("Invalid shape of given tensor for division."); }
	for(int i = 0; i < dataSize; ++i) {
		data.at(i) = (other.data.at(i) == 0) ? 1000.0f : data.at(i) / other.data.at(i);
	}
	checkForNan("/= other tensor");
	return *this;
}

ml::Tensor ml::Tensor::operator+(double scalar) const {
	ml::Tensor res(*this);
	for(double& elem : res.data) { elem += scalar; }
	res.checkForNan("adding scalar");
	return res;
}

ml::Tensor ml::Tensor::operator-(double scalar) const {
	ml::Tensor res(*this);
	for(double& elem : res.data) { elem -= scalar; }
	res.checkForNan("subtracting scalar");
	return res;
}

ml::Tensor ml::Tensor::operator*(double scalar) const {
	ml::Tensor res(*this);
	for(double& elem : res.data) { elem *= scalar; }
//		if (std::isnan(res.data[i]) || std::isinf(res.data[i])) { res.data[i] = 0; }
	res.checkForNan("multiplying by scalar");
	return res;
}

ml::Tensor ml::Tensor::operator/(double scalar) const {
	if (scalar == 0) { throw std::invalid_argument("Can't divide by zero."); }
	ml::Tensor res(*this);
	for(double& elem : res.data) { elem /= scalar; }
	res.checkForNan("deviding by scalar");
	return res;
}

ml::Tensor& ml::Tensor::operator+=(double scalar) {
	for(double& elem : data) { elem += scalar; }
	checkForNan("+= scalar");
	return *this;
}

ml::Tensor& ml::Tensor::operator-=(double scalar) {
	for(double& elem : data) { elem -= scalar; }
	checkForNan("-= scalar");
	return *this;
}

ml::Tensor& ml::Tensor::operator*=(double scalar) {
	for(double& elem : data) { elem *= scalar; }
	checkForNan("*= scalar");
	return *this;
}

ml::Tensor& ml::Tensor::operator/=(double scalar) {
	if (scalar == 0) { throw std::invalid_argument("Can't divide by zero."); }
	for(double& elem : data) { elem /= scalar; }
	checkForNan("/= scalar");
	return *this;
}

int ml::Tensor::calcDataIndex(const std::vector<int>& indices) const {
	int index = 0;
	int multiplier = 1;
	for (int i = 0; i < shape.size() ; ++i) {
		index += indices.at(i) * multiplier;
		multiplier *= shape.at(i);
	}
	return index;
}

std::vector<int>& ml::Tensor::increaseIndices(std::vector<int>& indices) const {
	indices.back() += 1;
	for (int i = int(shape.size()) - 1; i >= 0; --i) {
		if (indices.at(i) >= shape.at(i)) {
			indices.at(i) = 0;
			if (i != 0) { indices.at(i - 1) += 1; }
		} else {
			return indices;
		}
	}
	return indices;
}

void ml::Tensor::checkForNan(const std::string& message) {
	for (auto& elem : data) {
		if (std::isnan(elem)) {
			elem = std::numeric_limits<double>::min();
			throw std::invalid_argument("isnan when: " + message);
		}
		if (std::isinf(elem)) {
			elem = std::numeric_limits<double>::max();
			throw std::invalid_argument("isinf when: " + message);
		}
	}
}