//
// Created by jwkim on 18. 12. 4.
//

#ifndef CUBBYDNN_SIMPLETEST_HPP
#define CUBBYDNN_SIMPLETEST_HPP
#include <iostream>
#include <vector>
#include "backend/graph/graph.hpp"

using namespace cubby_dnn;

int Add(int a, int b);

std::vector<operation_info> Example1();

std::vector<operation_info> Example2();

std::vector<operation_info> Example3();


#endif //CUBBYDNN_SIMPLETEST_HPP
