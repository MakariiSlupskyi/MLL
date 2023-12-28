#ifndef MLL_DATA_H_
#define MLL_DATA_H_

#include "MLL/Linear/Tensor.h"
#include <string>
#include <vector>

namespace ml {
	class Data
	{
	public:
		Data(const std::vector<ml::Tensor>& data);
		Data(const std::vector<int>& dataShape, const std::vector<std::vector<double>>& vecData);
		Data(const std::string& path);

		Data merge(const ml::Data& other);
		Data extract(int start, int size) const;
		int size() const;

		void write(const std::string& path);
		Data read(const std::string& path);

		ml::Tensor operator[](int index) const;
		ml::Tensor at(int index) const;

	private:
		std::vector<int> dataShape;
		std::vector<ml::Tensor> data;
	};
}

#endif