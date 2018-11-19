//
// Created by Justin on 18. 11. 13.
//

#ifndef CUBBYDNN_GENERATOR_TENSOR_HPP
#define CUBBYDNN_GENERATOR_TENSOR_HPP

#include "Backend/operations_decl/base_operations_decl.hpp"
#include "Backend/util_decl/generate_tensor_decl.hpp"

namespace cubby_dnn
{
template <typename T>
Tensor<T> Generate<T>::placeHolder(const std::vector<int> &shape,
                                   Stream<T> stream, const std::string &name)
{
    check_arguments(shape);
    auto operation_id = static_cast<int>(Adj_management<T>::add_op_adj());
    Tensor<T> rtn_tensor(Tensor_type::placeHolder, shape, operation_id, true,
                         "tensor_from_op: " + name);
    auto new_op =
        placeHolder_op<T>(static_cast<unsigned long>(operation_id), name);
    Operation_management<T>::add_op(new_op, operation_id);
    return rtn_tensor;
}

template <typename T>
Tensor<T> Generate<T>::weight(const std::vector<int> &shape, bool trainable,
                              const std::string &name)
{
    check_arguments(shape);
    auto operation_id = static_cast<int>(Adj_management<T>::add_op_adj());
    Tensor<T> rtn_tensor(Tensor_type ::weight, shape, operation_id, true,
                         "tensor_from_op: " + name);
    auto new_op = weight_op<T>(static_cast<unsigned long>(operation_id), name);
    Operation_management<T>::add_op(new_op);
    return rtn_tensor;
}

// TODO : Add some special operations for filter to do
template <typename T>
Tensor<T> Generate<T>::filter(const std::vector<int> &shape, bool trainable,
                              const std::string &name)
{
    check_arguments(shape);
    auto operation_id = static_cast<int>(Adj_management<T>::add_op_adj());
    Tensor<T> rtn_tensor(Tensor_type ::filter, shape, operation_id, true,
                         "tensor_from_op: " + name);
    auto new_op = weight_op<T>(static_cast<unsigned long>(operation_id), name);
    Operation_management<T>::add_op(new_op);
    return rtn_tensor;
}

template <typename T>
bool Generate<T>::check_arguments(const std::vector<int> &shape)
{
    // TODO: find way to check if argument was verifiable
    static bool has_been_valid = true;

    if (!has_been_valid)
        return false;

    if (shape.empty())
    {
        has_been_valid = false;
        std::cout << "Argument shape is empty" << std::endl;
    }
    else if (shape.size() > max_dim)
    {
        has_been_valid = false;
        std::cout << "dimension of shape is over 3" << std::endl;
    }

    long size = 1;
    for (auto elem : shape)
        size *= elem;

    if (size < 0)
    {
        has_been_valid = false;
        std::cout << "Invalid shape" << std::endl;
    }

    return has_been_valid;
}

template <typename T>
Tensor<T> Operate<T>::matMul(Tensor<T> &tensor1, Tensor<T> &tensor2,
                             const std::string &name)
{
    // TODO check for validity
    auto id = static_cast<int>(Adj_management<T>::add_op_adj());
    tensor1.to = id;
    tensor2.to = id;
    // TODO: find way to initialize the default data
    // initialize(initialization_method)

    // set parameter tensor objects
    Tensor_object<T> tensor_object1(
        std::vector<T>(tensor1.get_data_size()), tensor1.shape, tensor1.type,
        "container of :" + name, tensor1.from, tensor1.to);

    auto tensor_object_ptr1 =
        std::make_shared<Tensor_object<T>>(tensor_object1);

    Tensor_object<T> tensor_object2(
        std::vector<T>(tensor2.get_data_size()), tensor2.shape, tensor2.type,
        "container_of :" + name, tensor2.from, tensor2.to);

    auto tensor_object_ptr2 =
        std::make_shared<Tensor_object<T>>(tensor_object2);

    tensor1.tensor_object_ptr = tensor_object_ptr1;
    tensor2.tensor_object_ptr = tensor_object_ptr2;

    // add parameter tensor_objects to new global adjacency matrix
    Adj_management<T>::add_edge(tensor1.from, tensor1.to, tensor_object_ptr1);
    Adj_management<T>::add_edge(tensor2.from, tensor2.to, tensor_object_ptr2);

    auto operation1 = Operation_management<T>::get_op(tensor1.from);
    operation1.add_output(tensor_object_ptr1);
    auto operation2 = Operation_management<T>::get_op(tensor2.from);
    operation2.add_output(tensor_object_ptr2);

    // setting the return tensor
    std::vector<int> new_shape{ tensor1.shape[0], tensor2.shape[1] };
    // row size of the first tensor * col size of the second tensor

    Tensor<T> rtn_tensor(Tensor_type ::other, new_shape, id, true,
                         "tensor_from_op" + name);
    Mat_mul_op<T> mat_mul_op(static_cast<unsigned long>(id), name);
    Operation_management<T>::add_op(mat_mul_op);
    return rtn_tensor;
}

template <typename T>
Tensor<T> Operate<T>::matAdd(Tensor<T> &tensor1, Tensor<T> &tensor2,
                             const std::string &name)
{
 //   Tensor<T> rtn(bias, std::vector(), 0);
}

template <typename T>
Tensor<T> Operate<T>::matDot(Tensor<T> &tensor1, T multiplier,
                             const std::string &name)
{
 //   return Tensor<T>(bias, std::vector(), 0);
}

template <typename T>
Tensor<T> Operate<T>::reshape(Tensor<T> &tensor1, const std::vector<int> &shape,
                              const std::string &name)
{
 //   return Tensor<T>(bias, std::vector(), 0);
}
}  // namespace cubby_dnn

#endif  // CUBBYDNN_GENERATOR_TENSOR_HPP
