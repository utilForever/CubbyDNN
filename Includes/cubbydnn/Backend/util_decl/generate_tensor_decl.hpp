//
// Created by Justin on 18. 11. 13.
//

#ifndef CUBBYDNN_GENERATE_TENSOR_HPP
#define CUBBYDNN_GENERATE_TENSOR_HPP

#include "Tensor_container_decl.hpp"
#include "stream_decl.hpp"

namespace cubby_dnn{
    template<typename T>
    class generate_Ops{
        static const int placeHolder_operation_index = -1;

        static const int max_dim = 3;

        //TODO: think about ways to put data stream through placeholders

        static Tensor <T> placeHolder(std::vector<int> shape, Stream<T> stream);

        static Tensor<T> placeHolder(std::vector<int> shape, std::string name);

        static Tensor<T> weight(std::vector<int> shape, std::string name, bool trainable = true);

        static Tensor<T> weight(std::vector<int> shape, bool trainable = true);

        static Tensor<T> filter(std::vector<int> shape, std::string name, bool trainable = true);

        static Tensor<T> filter(std::vector<int> shape, bool trainable = true);
    };
}

#endif //CUBBYDNN_GENERATE_TENSOR_HPP
