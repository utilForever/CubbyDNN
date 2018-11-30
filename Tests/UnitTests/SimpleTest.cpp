#include <Test.h>
#include <gtest/gtest.h>

TEST(SimpleTest, Add)
{
    EXPECT_EQ(5, Add(2, 3));
    std::cout << "called Test" << std::endl;
}

TEST(Test1, Example1)
{
    std::vector<std::tuple<long, unsigned long, unsigned long, std::string>>
        ans{ { 0, 0, 0,
               "Empty operation" },  // operation_id, input_size, output_size
             { 1, 0, 3, "test placeHolder operation1" },
             { 2, 0, 3, "test weight operation1" },
             { 3, 2, 2, "test matMul operation1" },
             { 4, 1, 0, "test wrapper operation1" },
             { 5, 2, 1, "test matMul operation2" },
             { 6, 2, 1, "test matAdd operation1" },
             { 7, 1, 1, "test matDot operation1" },
             { 8, 1, 1, "test reshape operation1" },
             { 9, 1, 0, "test wrapper operation2" },
             { 10, 1, 0, "test wrapper operation3" },
             { 11, 1, 0, "test wrapper operation4" } };

    EXPECT_EQ(ans, Example1());
}

TEST(Test2, Example2)
{
    std::vector<std::tuple<long, unsigned long, unsigned long, std::string>>
        ans{ { 0, 0, 0, "Empty operation" },
             { 1, 0, 1, "test placeHolder operation1" },
             { 2, 0, 1, "test weight operation2" },
             { 3, 2, 1, "test matMul operation1" },
             { 4, 1, 0, "test wrapper operation1" } };

    EXPECT_EQ(ans, Example2());
}

TEST(Test2, Example3)
{
    std::vector<std::tuple<long, unsigned long, unsigned long, std::string>>
        ans{ { 0, 0, 0, "Empty operation" },
             { 1, 0, 1, "test placeHolder operation1" },
             { 2, 0, 1, "test weight operation1" },
             { 3, 0, 2, "test weight operation2" },
             { 4, 2, 1, "test matMul operation1" },
             { 5, 2, 1, "test matMul operation2" },
             { 6, 2, 1, "test matMul operation3" },
             { 7, 1, 1, "test reshape operation1" },
             { 8, 1, 0, "test wrapper operation1" } };

    EXPECT_EQ(ans, Example3());
}
