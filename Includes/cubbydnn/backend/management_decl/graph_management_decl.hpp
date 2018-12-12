//
// Created by Justin on 18. 12. 4.
//

#ifndef CUBBYDNN_GRAPH_MANAGEMENT_HPP
#define CUBBYDNN_GRAPH_MANAGEMENT_HPP

#include <deque>
#include <mutex>
#include "backend/graph/operations.hpp"
#include "backend/graph/tensor.hpp"

namespace cubby_dnn {
/**
 * @brief contains, and manages operation objects used by graph
 * @tparam T
 */
    template<typename T>
    class operation_management {
    public:
        static long add_operation(const operation<T>& operation_to_add);

        static operation<T>& get_operation(long operation_id);

        static void print_operation_info();

        static const std::vector<operation_info> get_operation_info();

        static size_t operation_vector_size();

        static long get_next_operation_id();

        static void clear();

    private:
        static std::deque<operation<T>> operation_vector;
    };

    template<typename T>
    std::deque<operation<T>>
    operation_management<T>::operation_vector;

/**
 * @brief Contains, manages tensor_objects required for graph structure
 * @tparam T
 *
 */
    template<typename T>
    class tensor_object_management {
    public:
        friend operation_management<T>;

        static long add_tensor_object(const tensor_object<T>& object);

        static long add_tensor_object(tensor_object<T>&& object);

        static const typename tensor_object<T>::info& get_tensor_information(
                long tensor_id);

        /// returns unique pointer of corresponding tensor_object's data
        /// since data ptr is taken, it must be returned to original object
        /// by calling 'tensor_object_management<T>::return_tensor_data_ptr()'
        static std::unique_ptr<typename tensor_object<T>::data> get_tensor_data_ptr(
                long tensor_id);

        /// returns unique_ptr of tensor_object's data to corresponding tensor
        /// object
        static void return_tensor_data_ptr(
                long tensor_id, std::unique_ptr<typename tensor_object<T>::data> rhs);

        static void clear();

    private:
        static std::deque<tensor_object<T>> tensor_object_vector;
    };

    template<typename T>
    std::deque<tensor_object<T>>
    tensor_object_management<T>::tensor_object_vector;

/**
 * @brief manages adjacency matrix representing the graph
 * @tparam T
 */
    template<typename T>
    class adjacency_management {
    public:
        static long add_operation_to_adjacency(long operation_id);

        static void print_adjacency_matrix();

        static void clear();

    private:
        static const int default_gap = 2;

        static void print_number(long output_number);

        static std::deque<std::deque<long>> adjacency_matrix;  /// row: from, col:
        /// to
        static const size_t default_graph_size;

        static const long unallocated_state;
    };

    template<typename T>
    const size_t adjacency_management<T>::default_graph_size = 0;

    template<typename T>
    const long adjacency_management<T>::unallocated_state = -1;

    template<typename T>
    std::deque<std::deque<long>> adjacency_management<T>::adjacency_matrix;

};  // namespace cubby_dnn

#endif  // CUBBYDNN_GRAPH_MANAGEMENT_HPP
