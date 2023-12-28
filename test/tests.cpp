#include <gtest/gtest.h>
#include <MLL/Linear/Tensor.h>
#include <MLL/Linear/Matrix.h>
#include <MLL/Linear/Vector.h>

TEST(LinearAlgebra, TensorCreation1) {
	ml::Tensor t({3, 2}, {0, 1, 2, 3, 4, 5});
	EXPECT_EQ(t({1, 1}), 4) << "A mistake while indexing";

	ml::Tensor t1({2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
	EXPECT_EQ(t1({1, 1, 1}), 7) << "A mistake while indexing";
}

TEST(LinearAlgebra, TensorCreation2) {
	ml::Tensor t({2, 3, 3});
	EXPECT_EQ(t.getDataSize(), 18);
}

TEST(LinearAlgebra, TensorCreation3) {
	ml::Tensor t;
	EXPECT_EQ(t.getDataSize(), 1);
}

TEST(LinearAlgebra, TensorSetValues) {
	ml::Tensor tensor({3, 2});
	tensor.setValues({
		0, 1, 2,
		3, 4, 5
	});
	EXPECT_EQ(tensor({2, 1}), 5) << "A mistake while indexing";
	EXPECT_EQ(tensor({0, 1}), 3) << "A mistake while indexing";
}

TEST(LinearAlgebra, TensorSetConstant) {
	ml::Tensor tensor({3, 2});
	tensor.setConstant(10);
	EXPECT_EQ(tensor({0, 0}), 10.0f) << "A mistake while indexing";
}

TEST(LinearAlgebra, TensorSetRandom) {
	ml::Tensor tensor({3, 2});
	tensor.setRandom();
	EXPECT_TRUE(tensor({1, 1}) >= -1.0 && tensor({1, 1}) <= 1.0);
}

TEST(LinearAlgebra, TensorGetReversed) {
	ml::Tensor t({3, 2}, {0, 1, 2, 3, 4, 5});
	EXPECT_EQ(t({0, 0}), t.reverse()({2, 1}));
	EXPECT_EQ(t({1, 1}), t.reverse()({1, 0}));
}

TEST(LinearAlgebra, TensorGetSlice) {
	ml::Tensor t({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
	EXPECT_EQ(t.slice({0, 0}).getData().size(), 2);
}

TEST(LinearAlgebra, TensorGetBlock) {
	ml::Tensor t({3, 2}, {
		0, 1, 2,
		3, 4, 5
	});
	EXPECT_EQ(t.block({1, 0}, {2, 2}).getData().size(), 4) << "A mistake with shape";
	EXPECT_EQ(t.block({0, 0}, {2, 1}).getData()[1], 1) << "A mistake while indexing";
}

TEST(LinearAlgebra, TensorSetSlice) {
	ml::Tensor t1({2, 3, 2});
	t1.setConstant(0);
	t1.setChip(1, 1, t1.chip(1, 1).setConstant(1));
}

TEST(LinearAlgebra, TensorSetBlock) {
	ml::Tensor t1({3, 2});
	t1.setConstant(0);
	t1.setBlock({1, 0}, t1.block({1, 0}, {2, 2}).setConstant(1));
}

TEST(LinearAlgebra, TensorMultiplication1) {
	ml::Tensor t1({2, 2}, {0, 1, 2, 3});
	ml::Tensor t2({2, 2}, {2, 2, 2, 2});
	
	EXPECT_EQ((t1 * t2)({1, 0}), 2);
	t1 *= t2;
	EXPECT_EQ(t1({0, 1}), 4);
}

TEST(LinearAlgebra, TensorMultiplication2) {
	ml::Tensor t1({2, 2}, {0, 1, 2, 3});
	
	EXPECT_EQ((t1 * 4)({1, 0}), 4);
	t1 *= -1;
	EXPECT_EQ(t1({0, 1}), -2);
}

TEST(LinearAlgebra, MatrixMultiplication1) {
	ml::Matrix m(2, 3, {0, 1, 2, 3, 4, 5});
	std::vector<double> d = (m * m.reshape(3, 2)).getData();
	EXPECT_TRUE(d[0] == 10 && d[1] == 13 && d[2] == 28 && d[3] == 40);
}

TEST(LinearAlgebra, MatrixMultiplication2) {
	ml::Matrix m(2, 2);
	m.setConstant(0.0);
	m += 1;
	m *= 10;
	EXPECT_EQ(m(0, 0), 10);
	m /= 5;
	EXPECT_EQ(m(1, 1), 2);
}

TEST(LinearAlgebra, VectorCreation) {
	ml::Vector v1({1, 2, 3});
	ml::Vector v2(5);
	v2.setConstant(1);
	ml::Vector v3(1, {0});

	EXPECT_TRUE(v1.getShape()[0] == 3 && v1.getShape()[1] == 1 && v1.getData()[2] == 3);
	EXPECT_TRUE(v2.getShape()[0] == 5 && v2.getShape()[1] == 1 && v2.getData()[3] == 1);
	EXPECT_TRUE(v3.getShape()[0] == 1 && v3.getShape()[1] == 1 && v3.getData()[0] == 0);
}

TEST(LinearAlgebra, VectorMultiplitation) {
	ml::Vector v1({0, 1, 2, 3});
	ml::Vector v2(3);
	v2.setConstant(1);

	ml::Matrix m = v1 * v2.transpose();
	EXPECT_TRUE(m.getShape()[0] == 4 && m.getShape()[1] == 3);
}