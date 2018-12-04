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
void operation_management<T>::add_op(operation<T> operation)
{
    std::lock_guard<std::mutex> guard(operation_list_mutex);
    operation_list.emplace_back(operation);
}

template <typename T>
void operation_management<T>::set_op(unsigned int id,
                                     const operation<T> &operation)
{
    std::lock_guard<std::mutex> guard(operation_list_mutex);
    operation_list[id] = operation;
}

template <typename T>
void operation_management<T>::add_output_of(
    long id, std::shared_ptr<tensor_object<T>> tensor_ptr)
{
    decltype(auto) operation = get_op(id);
    operation.add_output(tensor_ptr);
}

template <typename T>
void operation_management<T>::add_input_of(
    long id, std::shared_ptr<tensor_object<T>> tensor_ptr)
{
    decltype(auto) operation = get_op(id);
    operation.add_input(tensor_ptr);
}
}  // namespace cubby_dnn
#endif  // CUBBYDNN_BASE_OPERATIONS_HPP
