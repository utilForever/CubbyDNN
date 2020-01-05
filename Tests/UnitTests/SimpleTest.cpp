#include "gtest/gtest.h"
#include <SimpleTest.hpp>

TEST(SimpleTest, Add)
{
    EXPECT_EQ(5, Add(2, 3));
    std::cout << "called Test" << std::endl;
}