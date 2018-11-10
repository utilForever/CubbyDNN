//
// Created by jwkim on 18. 11. 10.
//

#ifndef CUBBYDNN_MANAGEMENT_H
#define CUBBYDNN_MANAGEMENT_H

#include "backend.h"
#include "exceptions.h"
#include <vector>
#include <deque>

namespace cubby_dnn{
    ///static class for graph management
    template<typename T>
    class Management{
    public:
        ///Adds new operation
        static int add_op();
        ///Adds new edge between two
        static void add_edge(const int from, const int to, Tensor<T> &tensor) noexcept;


    private:
        static std::deque<std::vector<std::unique_ptr<Tensor<T>>>> adj;
    };

}


#endif //CUBBYDNN_MANAGEMENT_H
