#include <iostream>
#include <SimpleTest.hpp>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

TEST_CASE("Simple test")
{
    CHECK(5 == Add(2, 3));
    std::cout << "called Test" << std::endl;
}
