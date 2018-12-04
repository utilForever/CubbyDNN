//
// Created by jwkim on 18. 12. 4.
//

#ifndef CUBBYDNN_MANAGEMENT_DECL_HPP
#define CUBBYDNN_MANAGEMENT_DECL_HPP
#include "backend/management_decl/graph_management_decl.hpp"

namespace cubby_dnn
{
template <typename T>
long tensor_object_management<T>::add_tensor_object(
    const tensor_object<T>& object)
{
    tensor_object_vector.emplace_back(object);
    auto tensor_object_id = static_cast<long>(tensor_object_vector.size());
    return tensor_object_id;
}

template <typename T>
tensor_object<T>& tensor_object_management<T>::get_tensor_object(long id)
{
    return tensor_object_vector.at(static_cast<size_t>(id));
}

template <typename T>
long operation_management<T>::add_operation(
    const operation<T>& operation_to_add)
{
    operation_vector.emplace_back(operation_to_add);
    auto tensor_object_id = static_cast<long>(operation_vector.size());
    return tensor_object_id;
}

template <typename T>
operation<T>& operation_management<T>::get_operation(long id)
{
    return operation_vector.at(id);
}

template <typename T>
void operation_management<T>::print_operation_info()
{
    for (auto op : operation_vector)
    {
        std::cout << op.print_info() << std::endl;
    }
}

template <typename T>
const std::vector<operation_info> operation_management<T>::get_operation_info()
{
    std::vector<operation_info> op_vector;
    for (operation<T> operation : operation_vector)
    {
        op_vector.emplace_back(operation.get_info());
    }
    return op_vector;
}

template <typename T>
size_t operation_management<T>::operation_vector_size()
{
    return operation_vector.size();
}

template<typename T>
long operation_management<T>::get_next_operation_id() {
    return static_cast<long>(operation_vector.size());
}

template<typename T>
long adjacency_management<T>::add_operation_to_adjacency()
{
    auto graph_size = adjacency_matrix.size();

    size_t expected_row_size = (graph_size >= default_graph_size)
                                   ? graph_size + 1
                                   : default_graph_size;

    /// in order to save execution time,
    /// increment the graph size only if expected size exceeds default graph
    /// size
    if (expected_row_size > default_graph_size)
    {
        auto emplace_until_expected_row_size =
            [expected_row_size](std::deque<long>& arg) mutable {
                while (expected_row_size > arg.size())
                {
                    arg.emplace_back(
                        unallocated_state);  // copy-construct new temp
                }
            };

        std::for_each(adjacency_matrix.begin(), adjacency_matrix.end(),
                      emplace_until_expected_row_size);
    }

    adjacency_matrix.emplace_back(
        std::deque<long>(expected_row_size, unallocated_state));

    return static_cast<long>(adjacency_matrix.size());
}

template<typename T>
void adjacency_management<T>::print_adjacency_matrix()
{
    std::cout << "--Adjacency Matrix--" << std::endl;
    for (auto row : adjacency_matrix)
    {
        for (auto col : row)
        {
            std::cout << col;
        }
        std::cout << std::endl;
    }
}
}  // namespace cubby_dnn

#endif  // CUBBYDNN_MANAGEMENT_DECL_HPP
