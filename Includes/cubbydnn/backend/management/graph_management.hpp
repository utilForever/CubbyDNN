//
// Created by jwkim on 18. 12. 4.
//

#ifndef CUBBYDNN_MANAGEMENT_DECL_HPP
#define CUBBYDNN_MANAGEMENT_DECL_HPP
#include "backend/management_decl/graph_management_decl.hpp"

namespace cubby_dnn
{
template <typename T>
long tensor_data_management<T>::add_tensor_data(const tensor_data<T>& object)
{
    tensor_data_vector.emplace_back(object);
    auto tensor_data_id = static_cast<long>(tensor_data_vector.size());
    return tensor_data_id;
}

template <typename T>
long tensor_data_management<T>::add_tensor_data(tensor_data<T>&& object)
{
    tensor_data_vector.emplace_back(std::forward<tensor_data<T>>(object));
    auto tensor_data_id = static_cast<long>(tensor_data_vector.size());
    return tensor_data_id;
}

template <typename T>
tensor_data<T>& tensor_data_management<T>::get_tensor_data(long id)
{
    return tensor_data_vector.at(static_cast<size_t>(id));
}

template <typename T>
long operation_management<T>::add_operation(
    const operation<T>& operation_to_add)
{
    operation_vector.emplace_back(operation_to_add);
    auto tensor_data_id = static_cast<long>(operation_vector.size());
    return tensor_data_id;
}

template <typename T>
operation<T>& operation_management<T>::get_operation_by_id(long id)
{
    return operation_vector.at(id);
}

template <typename T>
void operation_management<T>::print_operation_info()
{
    for (auto op : operation_vector)
    {
        std::cout << op.print_information() << std::endl;
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

template <typename T>
long operation_management<T>::get_next_operation_id()
{
    return static_cast<long>(operation_vector.size());
}

template <typename T>
long adjacency_management<T>::add_operation_to_adjacency(long operation_id)
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

    auto input_vect = operation_management<T>::get_operation_by_id(operation_id)
                          .get_input_tensor_vector();

    for (long id : input_vect)
    {
        // TODO: put ID of corresponding tensor instead of 'id'
        adjacency_matrix[static_cast<size_t>(operation_id)][id] = 1;
    }
    return static_cast<long>(adjacency_matrix.size());
}

template <typename T>
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
