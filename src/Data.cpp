#include "MLL/Data.h"
#include "MLL/Functions/Others.h"
#include <stdexcept>
#include <fstream>

ml::Data::Data(const std::vector<ml::Tensor>& data) : data(data), dataShape(data.at(0).getShape()) // FIXME
{
	for (int i = 0; i < data.size(); ++i) {
		if (dataShape.size() != data[i].getShape().size()) {
			throw std::invalid_argument("Invalid data was provided");
		}
	}
}

ml::Data::Data(const std::vector<int>& dataShape, const std::vector<std::vector<double>>& vecData) {
	this->dataShape = (dataShape.size() == 1) ? std::vector<int>{dataShape[0], 1} : dataShape;
	
	data.resize(vecData.size());
	for (int i = 0; i < vecData.size(); ++i) {
		data[i] = ml::Tensor(dataShape, vecData[i]);
	}
}

ml::Data::Data(const std::string& path) {
	this->read(path);
}

ml::Data ml::Data::merge(const ml::Data& other) {
	data.insert(data.end(), other.data.begin(), other.data.end());
	return *this;
}

ml::Data ml::Data::extract(int start, int size) const {
	return ml::Data(std::vector<ml::Tensor>(data.cbegin() + start, data.cbegin() + start + size));
}

int ml::Data::size() const { return data.size(); }


void ml::Data::write(const std::string& path) {
	std::ofstream file(path);

	if (!file.is_open()) {
		throw std::invalid_argument("Failed to open file " + path + "\n");
	} else {
		file << data.size() << std::endl;

		for (int t : dataShape) { file << t << ' '; }
		file << std::endl;

		for (auto tensor : data) {
			for (auto t : tensor.getData()) { file << t << ' '; }
			file << std::endl;
		}
	}
}

ml::Data ml::Data::read(const std::string& path) {
	std::ifstream file(path);

	if (!file.is_open()) {
		throw std::invalid_argument("Failed to open file " + path + "\n");
	} else {
		std::string line;
		
		// Read size of data
		std::getline(file, line);
		data.resize(std::stoi(line));

		// Read shape of data
		std::getline(file, line);
		std::vector<int> shape = ml::strToIntVec(line);

		// Read data
		for (auto& t : data) {
			std::getline(file, line);
			t = ml::Tensor(shape, ml::strToDoubleVec(line));
		}
	}

	return *this;
}

ml::Tensor ml::Data::operator[](int index) const { return data[index]; }

ml::Tensor ml::Data::at(int index) const { return data.at(index); }