//
// Created by Justin on 18. 11. 13.
//

#ifndef CUBBYDNN_GENERATOR_TENSOR_HPP
#define CUBBYDNN_GENERATOR_TENSOR_HPP

#include "Backend/operations_decl/base_operations_decl.hpp"
#include "Backend/util_decl/generate_tensor_decl.hpp"
#include "Backend/util_decl/shape.hpp"

namespace cubby_dnn
{
template <typename T>
Tensor<T> Generate<T>::placeHolder(const std::vector<int> &shape,
                                   Stream<T> stream, const std::string &name)
{
    if (!shape::check_shape(shape, name))
    {
        return get_default_tensor();  // check if shape is valid
    }

    auto operation_id = static_cast<int>(Adj_management<T>::add_op_adj());
    Tensor<T> rtn_tensor(Tensor_type::placeHolder, shape, operation_id, true,
                         "tensor_from_op: " + name);
    // declare empty operation
    auto new_op =
        placeHolder_op<T>(static_cast<unsigned long>(operation_id), name);
    // add the operation to the global operation list
    Operation_management<T>::add_op(new_op, operation_id);
    return rtn_tensor;
}

template <typename T>
Tensor<T> Generate<T>::weight(const std::vector<int> &shape, bool trainable,
                              const std::string &name)
{
    if (!shape::check_shape(shape, name))
    {
        return get_default_tensor();  // check if shape is valid
    }

    auto operation_id = static_cast<int>(Adj_management<T>::add_op_adj());
    Tensor<T> rtn_tensor(Tensor_type ::weight, shape, operation_id, true,
                         "tensor_from_op: " + name);
    // declare empty operation
    auto new_op = weight_op<T>(static_cast<unsigned long>(operation_id), name);
    // add the operation to the global operation list
    Operation_management<T>::add_op(new_op);
    return rtn_tensor;
}

// TODO : Add some special operations for filter to do
template <typename T>
Tensor<T> Generate<T>::filter(const std::vector<int> &shape, bool trainable,
                              const std::string &name)
{
    if (!shape::check_shape(shape, name))
    {
        return get_default_tensor();  // check if shape is valid
    }

    auto operation_id = static_cast<int>(Adj_management<T>::add_op_adj());
    Tensor<T> rtn_tensor(Tensor_type ::filter, shape, operation_id, true,
                         "tensor_from_op: " + name);
    // declare empty operation
    auto new_op = weight_op<T>(static_cast<unsigned long>(operation_id), name);
    // add the operation to the global operation list
    Operation_management<T>::add_op(new_op);
    return rtn_tensor;
}

template <typename T>
Tensor<T> Operate<T>::matMul(Tensor<T> &tensor1, Tensor<T> &tensor2,
                             const std::string &name)
{
    // validity checking
    if (!is_valid_tensor({ tensor1, tensor2 }))
    {
        return get_default_tensor();
    }

    if (tensor1.shape[1] != tensor2.shape[0] ||
        tensor1.shape[2] != tensor2.shape[2])
    {
        // number of rows of first tensor should be identical to number of
        // columns of second tensor
        std::cout << "tensor shapes doesn't match for multiplication"
                  << std::endl;
        std::cout << "This Error occurs from operation: " << name << std::endl;
        return get_default_tensor();
    }

    auto id = static_cast<int>(Adj_management<T>::add_op_adj());
    tensor1.to = id;
    tensor2.to = id;
    // TODO: find way to initialize the default data
    // initialize(initialization_method)

    auto tensor_object_ptr1 = std::make_shared<Tensor_object<T>>(
        std::vector<T>(tensor1.get_data_size()), tensor1.shape, tensor1.type,
        "container of :" + name, tensor1.from, tensor1.to);

    auto tensor_object_ptr2 = std::make_shared<Tensor_object<T>>(
        std::vector<T>(tensor2.get_data_size()), tensor2.shape, tensor2.type,
        "container_of :" + name, tensor2.from, tensor2.to);

    tensor1.set_tensor_object(tensor_object_ptr1);
    tensor2.set_tensor_object(tensor_object_ptr2);

    // add parameter tensor_objects to new global adjacency matrix
    Adj_management<T>::add_edge(tensor1.from, tensor1.to, tensor_object_ptr1);
    Adj_management<T>::add_edge(tensor2.from, tensor2.to, tensor_object_ptr2);

    Operation_management<T>::add_output_of(tensor1.from, tensor_object_ptr1);
    Operation_management<T>::add_output_of(tensor2.from, tensor_object_ptr2);

    // setting the return tensor
    // numCols of first tensor, numRows of second tensor and dimension of third
    // tensor
    std::vector<int> new_shape{ tensor1.shape[0], tensor2.shape[1],
                                tensor1.shape[2] };
    // row size of the first tensor * col size of the second tensor

    Tensor<T> rtn_tensor(Tensor_type ::normal, new_shape, id, true,
                         "tensor_from_op: " + name);
    Mat_mul_op<T> mat_mul_op(static_cast<unsigned long>(id), name);
    Operation_management<T>::add_op(mat_mul_op);
    return rtn_tensor;
}

template <typename T>
Tensor<T> Operate<T>::matAdd(Tensor<T> &tensor1, Tensor<T> &tensor2,
                             const std::string &name)
{
    if (!is_valid_tensor({ tensor1, tensor2 }))
    {
        return get_default_tensor();
    }

    if (tensor1.shape != tensor2.shape)
    {
        std::cout << "tensor shapes doesn't match for Addition" << std::endl;
        std::cout << "This Error occurs from operation: " << name << std::endl;
        return get_default_tensor();
    }

    auto id = static_cast<int>(Adj_management<T>::add_op_adj());
    tensor1.to = id;
    tensor2.to = id;
    // TODO: find way to initialize the default data
    // initialize(initialization_method)

    auto tensor_object_ptr1 = std::make_shared<Tensor_object<T>>(
        std::vector<T>(tensor1.get_data_size()), tensor1.shape, tensor1.type,
        "container of :" + name, tensor1.from, tensor1.to);

    auto tensor_object_ptr2 = std::make_shared<Tensor_object<T>>(
        std::vector<T>(tensor2.get_data_size()), tensor2.shape, tensor2.type,
        "container_of :" + name, tensor2.from, tensor2.to);

    tensor1.set_tensor_object(tensor_object_ptr1);
    tensor2.set_tensor_object(tensor_object_ptr2);

    // add parameter tensor_objects to new global adjacency matrix
    Adj_management<T>::add_edge(tensor1.from, tensor1.to, tensor_object_ptr1);
    Adj_management<T>::add_edge(tensor2.from, tensor2.to, tensor_object_ptr2);

    Operation_management<T>::add_output_of(tensor1.from, tensor_object_ptr1);
    Operation_management<T>::add_output_of(tensor2.from, tensor_object_ptr2);

    // setting the return tensor
    std::vector<int> new_shape{ tensor1.shape[0], tensor2.shape[1] };
    // row size of the first tensor * col size of the second tensor

    Tensor<T> rtn_tensor(Tensor_type ::normal, new_shape, id, true,
                         "tensor_from_op: " + name);
    Mat_mul_op<T> mat_mul_op(static_cast<unsigned long>(id), name);
    Operation_management<T>::add_op(mat_mul_op);
    return rtn_tensor;
}

template <typename T>
Tensor<T> Operate<T>::matDot(Tensor<T> &tensor1, T multiplier,
                             const std::string &name)
{
    if (!is_valid_tensor({ tensor1 }))
    {
        return get_default_tensor();
    }

    auto id = static_cast<int>(Adj_management<T>::add_op_adj());
    tensor1.to = id;
    // TODO: find way to initialize the default data
    // initialize(initialization_method)

    auto tensor_object_ptr1 = std::make_shared<Tensor_object<T>>(
        std::vector<T>(tensor1.get_data_size()), tensor1.shape, tensor1.type,
        "container of :" + name, tensor1.from, tensor1.to);

    tensor1.set_tensor_object(tensor_object_ptr1);

    // add parameter tensor_objects to new global adjacency matrix
    Adj_management<T>::add_edge(tensor1.from, tensor1.to, tensor_object_ptr1);

    Operation_management<T>::add_output_of(tensor1.from, tensor_object_ptr1);

    // setting the return tensor
    std::vector<int> new_shape = tensor1.shape;
    // row size of the first tensor * col size of the second tensor

    Tensor<T> rtn_tensor(Tensor_type ::normal, new_shape, id, true,
                         "tensor_from_op: " + name);
    Mat_mul_op<T> mat_mul_op(static_cast<unsigned long>(id), name);
    Operation_management<T>::add_op(mat_mul_op);
    return rtn_tensor;
}

template <typename T>
Tensor<T> Operate<T>::reshape(Tensor<T> &tensor1, const std::vector<int> &shape,
                              const std::string &name)
{
    if (!is_valid_tensor({ tensor1 }))
    {
        return get_default_tensor();
    }

    if (!shape::check_shape(tensor1.shape, name))
    {
        std::cout << "tensor shapes doesn't match for reshaping" << std::endl;
        std::cout << "This Error occurs from operation: " << name << std::endl;
        return get_default_tensor();
    }

    if (tensor1.get_data_size() != shape::get_shape_size(shape))
    {
        std::cout << "size of new shape doesn't match for reshaping"
                  << std::endl;
        std::cout << "new size: "
                  << std::to_string(shape::get_shape_size(shape))
                  << "original size: "
                  << std::to_string(tensor1.get_data_size()) << std::endl;
        std::cout << "This Error occurs from operation: " << name << std::endl;
    }

    auto id = static_cast<int>(Adj_management<T>::add_op_adj());
    tensor1.to = id;
    // TODO: find way to initialize the default data
    // initialize(initialization_method)

    auto tensor_object_ptr1 = std::make_shared<Tensor_object<T>>(
        std::vector<T>(tensor1.get_data_size()), tensor1.shape, tensor1.type,
        "container of :" + name, tensor1.from, tensor1.to);

    tensor1.set_tensor_object(tensor_object_ptr1);

    // add parameter tensor_objects to new global adjacency matrix
    Adj_management<T>::add_edge(tensor1.from, tensor1.to, tensor_object_ptr1);

    Operation_management<T>::add_output_of(tensor1.from, tensor_object_ptr1);

    // setting the return tensor
    std::vector<int> new_shape = tensor1.shape;
    // row size of the first tensor * col size of the second tensor

    Tensor<T> rtn_tensor(Tensor_type ::normal, new_shape, id, true,
                         "tensor_from_op: " + name);
    Mat_mul_op<T> mat_mul_op(static_cast<unsigned long>(id), name);
    Operation_management<T>::add_op(mat_mul_op);
    return rtn_tensor;
}
}  // namespace cubby_dnn

#endif  // CUBBYDNN_GENERATOR_TENSOR_HPP
