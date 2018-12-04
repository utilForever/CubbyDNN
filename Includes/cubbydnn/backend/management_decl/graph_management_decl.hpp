//
// Created by Justin on 18. 12. 4.
//

#ifndef CUBBYDNN_GRAPH_MANAGEMENT_HPP
#define CUBBYDNN_GRAPH_MANAGEMENT_HPP

#include <deque>
#include <mutex>
#include "backend/graph/operations.hpp"
#include "backend/graph/tensor.hpp"

namespace cubby_dnn
{
template <typename T>
class tensor_object_management
{
 public:
    static long add_tensor_object(const tensor_object<T>& object);

    static tensor_object<T>& get_tensor_object(long id);

 private:
    static std::deque<tensor_object<T>> tensor_object_vector;
    static std::mutex tensor_object_vector_mutex;
};

template <typename T>
std::deque<tensor_object<T>> tensor_object_management<T>::tensor_object_vector;

template <typename T>
class operation_management
{
 public:
    static long add_operation(const operation<T>& operation_to_add);

    static operation<T>& get_operation(long id);

    static void print_operation_info();

    static const std::vector<operation_info> get_operation_info();

    static size_t operation_vector_size();

    static long get_next_operation_id();

 private:
    static std::deque<operation<T>> operation_vector;
};

template <typename T>
std::deque<operation<T>> operation_management<T>::operation_vector;

template <typename T>
class adjacency_management
{
 public:
    static long add_operation_to_adjacency();

    static void print_adjacency_matrix();

 private:
    static std::deque<std::deque<long>> adjacency_matrix;
    static const size_t default_graph_size;
    static const long unallocated_state;
};

template <typename T>
const size_t adjacency_management<T>::default_graph_size = 30;

template <typename T>
const long adjacency_management<T>::unallocated_state = -1;

template <typename T>
std::deque<std::deque<long>> adjacency_management<T>::adjacency_matrix;

};  // namespace cubby_dnn

#endif  // CUBBYDNN_GRAPH_MANAGEMENT_HPP
