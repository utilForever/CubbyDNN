//
// Created by Justin on 18. 11. 13.
//

#ifndef CUBBYDNN_GENERATOR_TENSOR_HPP
#define CUBBYDNN_GENERATOR_TENSOR_HPP

#include "Backend/util_decl/generate_tensor_decl.hpp"
#include "Backend/operations_decl/base_operations_decl.hpp"

namespace cubby_dnn
{
template <typename T>
Tensor<T> generate_ops<T>::placeHolder(const std::vector<int> &shape,
                                       Stream<T> stream)
{
    check_arguments(shape);

    auto id = static_cast<int>(Adj_management<T>::add_op_adj());

    Tensor<T> rtn(Tensor_type::placeHolder, shape, id, true);
    return rtn;
}

template <typename T>
Tensor<T> generate_ops<T>::placeHolder(const std::vector<int> &shape,
                                       Stream<T> stream,
                                       const std::string &name)
{
    check_arguments(shape);
    auto id = static_cast<int>(Adj_management<T>::add_op_adj());

    return Tensor<T>(Tensor_type::placeHolder, shape, id, true);
}

template <typename T>
Tensor<T> generate_ops<T>::weight(const std::vector<int> &shape,
                                  const std::string &name, bool trainable)
{
    check_arguments(shape);
    return Tensor<T>(placeHolder, shape, 0, false);
}

template <typename T>
Tensor<T> generate_ops<T>::weight(const std::vector<int> &shape, bool trainable)
{
    return Tensor<T>(placeHolder, shape, 0, false);
}

template <typename T>
Tensor<T> generate_ops<T>::filter(const std::vector<int> &shape,
                                  const std::string &name, bool trainable)
{
    return Tensor<T>(placeHolder, shape, 0, false);
}

template <typename T>
Tensor<T> generate_ops<T>::filter(const std::vector<int> &shape, bool trainable)
{
    return Tensor<T>(placeHolder, shape, 0, false);
}

template <typename T>
bool generate_ops<T>::check_arguments(const std::vector<int> &shape)
{
    if (shape.empty())
        std::cout << "Argument shape is empty" << std::endl;
    else if (shape.size() > max_dim)
        std::cout << "dimension of shape is over 3" << std::endl;

    long size = 1;
    for (auto elem : shape)
        size *= elem;

    if (size < 0)
        std::cout << "Invalid shape" << std::endl;
    return false;
}

}  // namespace cubby_dnn

#endif  // CUBBYDNN_GENERATOR_TENSOR_HPP
