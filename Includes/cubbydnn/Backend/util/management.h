//
// Created by jwkim on 18. 11. 10.
//

#ifndef CUBBYDNN_MANAGEMENT_H
#define CUBBYDNN_MANAGEMENT_H

#include "Tensor_object.h"
#include "exceptions.h"
#include <vector>
#include <deque>
#include <mutex>

namespace cubby_dnn{
    ///static class for graph management
    template<typename T>
    class Management{
    public:
        ///Adds new operation
        static int add_op() noexcept;
        ///Adds new edge between two
        static void add_edge(const int from, const int to, Tensor_object<T> &tensor) noexcept;

        static void add_placeHolder(std::unique_ptr<Tensor_object<T>> placeHolder) noexcept;

        static int get_graph_size(){
            return static_cast<int>(adj_forward.size());
        }

        static std::unique_ptr<Tensor_object<T>> get_tensor_ptr(const int from, const int to) noexcept;

    private:
        static std::deque<std::unique_ptr<Tensor_object<T>>> placeHolders;
        static std::deque<std::vector<std::unique_ptr<Tensor_object<T>>>> adj_forward;

        Management(){} ///disable the constructor

        static std::mutex adj_mutex;
    };
}


#endif //CUBBYDNN_MANAGEMENT_H
