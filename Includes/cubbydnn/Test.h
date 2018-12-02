#ifndef CUBBYDNN_TEST_H
#define CUBBYDNN_TEST_H

#include <iostream>
#include <vector>
#include "Backend/util/generate_tensor.hpp"

using namespace cubby_dnn;

int Add(int a, int b);

std::vector<operation_info> Example1();

std::vector<operation_info> Example2();

std::vector<operation_info> Example3();


#endif