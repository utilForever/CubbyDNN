#include <gtest/gtest.h>

#include <Test.h>

TEST(SimpleTest, Add)
{
    EXPECT_EQ(5, Add(2, 3));
}
