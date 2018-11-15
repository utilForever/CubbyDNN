//
// Created by Justin on 18. 11. 13.
//

#ifndef CUBBYDNN_GENERATOR_TENSOR_HPP
#define CUBBYDNN_GENERATOR_TENSOR_HPP

#include <Backend/util_decl/generate_tensor_decl.hpp>

#include "Backend/util_decl/generate_tensor_decl.hpp"

namespace cubby_dnn{

    template<typename T>
    Tensor <T> generate_tensor<T>::placeHolder(std::vector<int> shape, Stream<T> stream) {
        if(shape.empty())
            throw ArgumentException("argument 'shape' is empty");
        else if(shape.size() > max_dim)
            throw ArgumentException("dimension of shape is over 3");

        int this_num = Management<T>::get_plalceHolder_num();
        auto name = "placeHolder{" + std::to_string(this_num + 1 ) + "}";

        long size = 1;
        for(auto elem : shape){
            size *= elem;
        }

        if(size < 0) throw ArgumentException("invalid shape");

        Tensor<T> tensor(Tensor_type::placeHolder, placeHolder_operation_index);

        std::vector<T> data(static_cast<unsigned long>(size));

        Tensor_object<T> tensor_object(std::move(data), std::move(shape), Tensor_type::placeHolder, name, 0);
        std::unique_ptr<Tensor<T>> tensor_ptr = std::make_unique(tensor);
        Management<T>::add_placeHolder(tensor_ptr);
    }

    template<typename T>
    Tensor<T> generate_tensor<T>::placeHolder(std::vector<int> shape, std::string name){
        if(shape.empty())
            throw ArgumentException("argument 'shape' is empty");
        else if(shape.size() > max_dim)
            throw ArgumentException("dimension of shape is over 3");

        long size = 1;
        for(auto elem : shape)
            size *= elem;

        if(size < 0) throw ArgumentException("invalid shape");

        Tensor<T> tensor(Tensor_type::placeHolder, placeHolder_operation_index);

        std::vector<T> data(static_cast<unsigned long>(size));

        Tensor_object<T> tensor_object(data, shape, Tensor_type::placeHolder, name, 0);
        std::unique_ptr<Tensor<T>> tensor_ptr = std::make_unique(tensor);
        Management<T>::add_placeHolder(tensor_ptr);
    }

    template<typename T>
    Tensor<T> generate_tensor<T>::weight(std::vector<int> shape, std::string name, bool trainable) {
        return Tensor<T>(placeHolder, 0);
    }

    template<typename T>
    Tensor<T> generate_tensor<T>::weight(std::vector<int> shape, bool trainable) {
        return Tensor<T>(placeHolder, 0);
    }

    template<typename T>
    Tensor<T> generate_tensor<T>::filter(std::vector<int> shape, std::string name, bool trainable) {
        return Tensor<T>(placeHolder, 0);
    }

    template<typename T>
    Tensor<T> generate_tensor<T>::filter(std::vector<int> shape, bool trainable) {
        return Tensor<T>(placeHolder, 0);
    }


}

#endif //CUBBYDNN_GENERATOR_TENSOR_HPP
