#include <Test.h>
#include <gtest/gtest.h>

TEST(SimpleTest, Add)
{
    EXPECT_EQ(5, Add(2, 3));
    std::cout << "called Test" << std::endl;
}

TEST(Test1, Example1)
{
    std::vector<std::tuple<long, unsigned long, unsigned long>> ans
            {
                    {0, 0, 0},//operation_id, input_size, output_size
                    {1, 0, 3},
                    {2, 0, 3},
                    {3, 2, 2},
                    {4, 1, 0},
                    {5, 2, 1},
                    {6, 2, 1},
                    {7, 1, 1},
                    {8, 1, 1},
                    {9, 1, 0},
                    {10, 1, 0},
                    {11, 1, 0}
            };

    EXPECT_EQ(ans, Example1());
}
