//
// Created by jwkim on 18. 12. 4.
//

#ifndef CUBBYDNN_MANAGEMENT_DECL_HPP
#define CUBBYDNN_MANAGEMENT_DECL_HPP
#include "backend/management_decl/graph_management_decl.hpp"

namespace cubby_dnn
{
template <typename T>
long tensor_data_management<T>::add_tensor_object(
    const tensor_object<T>& object)
{
    tensor_data_vector.emplace_back(object);
    auto tensor_data_id = static_cast<long>(tensor_data_vector.size()) - 1;
    return tensor_data_id;
}

template <typename T>
long tensor_data_management<T>::add_tensor_object(tensor_object<T>&& object)
{
    tensor_data_vector.emplace_back(std::forward<tensor_object<T>>(object));
    auto tensor_data_id = static_cast<long>(tensor_data_vector.size()) - 1;
    return tensor_data_id;
}

template <typename T>
tensor_object<T>& tensor_data_management<T>::get_tensor_object_without_data(
    long id)
{
    return tensor_data_vector.at(static_cast<size_t>(id));
}

template<typename T>
void tensor_data_management<T>::return_tensor_data_ptr(long id, std::unique_ptr<typename tensor_object<T>::data> rhs){
    tensor_data_vector.at(id) = std::move(rhs);
}

template <typename T>
std::unique_ptr<typename tensor_object<T>::data>
tensor_data_management<T>::get_tensor_data_ptr(long tensor_id)
{
    return std::move(tensor_data_vector.at(static_cast<size_t>(tensor_id)).get_data_ptr());
}

template <typename T>
void tensor_data_management<T>::clear()
{
    tensor_data_vector.clear();
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
void operation_management<T>::clear()
{
    operation_vector.clear();
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

    const std::vector<long>& input_tensor_id_vect =
        operation_management<T>::get_operation_by_id(operation_id)
            .get_input_tensor_vector();

    for (long tensor_id : input_tensor_id_vect)
    {
        tensor_object<T> tensor =
            tensor_data_management<T>::get_tensor_object_without_data(
                tensor_id);
        adjacency_matrix.at(static_cast<size_t>(tensor.comes_from()))
            .at(static_cast<size_t>(operation_id)) = tensor_id;
    }
    return static_cast<long>(adjacency_matrix.size());
}

template <typename T>
void adjacency_management<T>::print_number(long output_number)
{
    std::string output_string;
    if (output_number == -1)
        output_string = '*';
    else
        output_string = std::to_string(output_number);
    while (output_string.size() < default_gap)
        output_string += " ";
    std::cout << output_string;
}

template <typename T>
void adjacency_management<T>::print_adjacency_matrix()
{
    std::cout << "--Adjacency Matrix--" << std::endl;
    std::cout << "row: from  col: to" << std::endl;

    long index;

    for (index = -1; index < static_cast<long>(adjacency_matrix.size());
         index++)
    {
        print_number(index);
    }

    for (int i = 0; i < default_gap; i++)
        std::cout << std::endl;

    index = 0;
    for (const auto row : adjacency_matrix)
    {
        print_number(index++);

        for (auto col : row)
        {
            print_number(col);
        }
        for (int i = 0; i < default_gap; i++)
            std::cout << std::endl;
    }
}

template <typename T>
void adjacency_management<T>::clear()
{
    adjacency_matrix.clear();
}
}  // namespace cubby_dnn

#endif  // CUBBYDNN_MANAGEMENT_DECL_HPP
