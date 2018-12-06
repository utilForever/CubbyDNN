//
// Created by Justin on 18. 11. 16.
//

#ifndef CUBBYDNN_BASE_OPERATIONS_HPP
#define CUBBYDNN_BASE_OPERATIONS_HPP
#include "backend/graph_decl/operations_decl.hpp"

namespace cubby_dnn
{
template <typename T>
operation<T>::operation() = default;

template <typename T>
void operation<T>::add_output_tensor(long tensor_id)
{
    output_tensor_id_vector.emplace_back(tensor_id);
}

template <typename T>
void operation<T>::add_input_tensor(long tensor_id)
{
    input_tensor_id_vector.emplace_back(tensor_id);
}

template <typename T>
size_t operation<T>::number_of_input_tensors()
{
    return input_tensor_id_vector.size();
}

template <typename T>
size_t operation<T>::number_of_output_tensors()
{
    return output_tensor_id_vector.size();
}

template <typename T>

const std::string &operation<T>::get_name()
{
    return name;
}

template <typename T>
long operation<T>::get_id()
{
    return operation_id;
}

template <typename T>
const std::string operation<T>::print_information()
{
    std::string info = name;
    info += "\noperation id: " + std::to_string(operation_id);
    info +=
        "\ninput tensor num: " + std::to_string(input_tensor_id_vector.size());
    info += "\noutput tensor num: " +
            std::to_string(output_tensor_id_vector.size());
    return info;
}

template <typename T>
operation_info operation<T>::get_info() const
{
    return operation_info(operation_id, input_tensor_id_vector.size(),
                          output_tensor_id_vector.size(), name);
}

template <typename T>
const std::vector<long> &operation<T>::get_input_tensor_vector() const
{
    return input_tensor_id_vector;
}

template <typename T>
const std::vector<long> &operation<T>::get_output_tensor_vector() const
{
    return output_tensor_id_vector;
}

template <typename T>
empty_op<T>::empty_op()
{
    this->op_type = operation_type::empty;
    this->name = "Empty operation";
}

template <typename T>
mat_mul_op<T>::mat_mul_op(long operation_id, const std::string &name)
{
    this->operation_id = operation_id;
    this->name = name;
    this->op_type = operation_type::mat_mul;
}

template <typename T>
mat_add_op<T>::mat_add_op(long operation_id, const std::string &name)
{
    this->operation_id = operation_id;
    this->name = name;
    this->op_type = operation_type::mat_add;
}

template <typename T>
mat_dot_op<T>::mat_dot_op(long operation_id, const std::string &name,
                          T multiplier)
{
    this->operation_id = operation_id;
    this->name = name;
    this->op_type = operation_type::mat_dot;
    this->multiplier = multiplier;
}

template <typename T>
reshape_op<T>::reshape_op(long operation_id, const std::string &name,
                          const tensor_shape &shape)
{
    this->operation_id = operation_id;
    this->name = name;
    this->op_type = operation_type::reshape;
    this->shape = shape;
}

template <typename T>
placeholder_op<T>::placeholder_op(long operation_id, const tensor_shape &shape,
                                  stream<T> &stream, const std::string &name)
{
    this->operation_id = operation_id;
    this->data_stream = stream;
    this->name = name;
    this->op_type = operation_type::placeholder;
    this->shape = shape;
}

template <typename T>
weight_op<T>::weight_op(long operation_id, const tensor_shape &shape,
                        const std::string &name)
{
    this->operation_id = operation_id;
    this->name = name;
    this->op_type = operation_type::weight;
    this->shape = shape;
}

template <typename T>
constant_op<T>::constant_op(long operation_id, const tensor_shape &shape,
                            const std::string &name)
{
    this->operation_id = operation_id;
    this->name = name;
    this->op_type = operation_type::constant;
    this->shape = shape;
}

template <typename T>
wrapper_op<T>::wrapper_op(long operation_id, const std::string &name)
{
    this->operation_id = operation_id;
    this->name = name;
    this->op_type = operation_type::wrapper;
}

}  // namespace cubby_dnn
#endif  // CUBBYDNN_BASE_OPERATIONS_HPP
