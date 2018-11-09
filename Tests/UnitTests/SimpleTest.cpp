#include <gtest/gtest.h>

#include <Test.h>
#include <Backend/storage/backend.h>
#include <Backend/storage/baseOp.h>


using namespace cubby_dnn;
TEST(SimpleTest, Add) {
    EXPECT_EQ(5, Add(2, 3));
    Add(2,3);
    Tensor<int> a(std::vector<int>{0}, std::vector<int>{0});
    Tensor<int> b(std::vector<int>{0}, std::vector<int>{0});

    emptyOp<int> empty = emptyOp<int>();

    MatDot<int, int> op{a, empty, 1};
}

