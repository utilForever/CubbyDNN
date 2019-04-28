//
// Created by jwkim98 on 4/20/19.
//

#ifndef CUBBYDNN_UTIL_HPP
#define CUBBYDNN_UTIL_HPP

#include <cubbydnn/Tensors/Decl/TensorData.hpp>
#include <cubbydnn/Tensors/Decl/TensorPlug.hpp>
#include <cubbydnn/Operations/Decl/Operation.hpp>

namespace CubbyDNN{

    template<typename T>
    void ConnectOperations(Operation<T> from, Operation<T> to, const TensorShape& shape);
}

#endif //CUBBYDNN_UTIL_HPP
