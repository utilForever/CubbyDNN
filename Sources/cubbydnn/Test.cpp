#include <Test.h>
#include <iostream>
#include "Backend/util/generate_tensor.hpp"

using namespace cubby_dnn;
int Add(int a, int b)
{
    // Example of multiplying two tensors of shape {1,1,1}
    return a + b;
}

void simple_Example()
{
    File_stream<int> file_stream;
    auto input_tensor1 = Generate<int>::placeHolder(
        std::vector<int>{ 1, 1, 1 }, file_stream, "test placeHolder operation");
    auto input_tensor2 = Generate<int>::weight(std::vector<int>{ 1, 1, 1 },
                                               true, "test weight operation");

    auto multiplied_tensor1 = Operate<int>::matMul(input_tensor1, input_tensor2,
                                                   "test matMul operation1");

    auto multiplied_tensor2 = Operate<int>::matMul(input_tensor1, input_tensor2,
            "test matMul operation2");

    Final<int>::wrapper(multiplied_tensor1, "test wrapper operation1");

    Final<int>::wrapper(multiplied_tensor2, "test wrapper operation2");
}
