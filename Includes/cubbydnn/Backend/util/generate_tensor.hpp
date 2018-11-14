//
// Created by jwkim on 18. 11. 13.
//

#ifndef CUBBYDNN_GENERATOR_TENSOR_HPP
#define CUBBYDNN_GENERATOR_TENSOR_HPP
#include "../util_decl/generate_tensor_decl.hpp"

namespace cubby_dnn{
    template<typename T>
    Tensor<T> generate_tensor<T>::placeHolder(std::vector<int> shape){
        if(shape.empty())
            throw ArgumentException("argument 'shape' is empty");
        else if(shape.size() > 3)
            throw ArgumentException("dimension of shape is over 3");

        int this_num = Management<T>::get_plalceHolder_num();
        std::string name = "placeHolder{" + std::to_string(this_num +1 ) + "}";

        Tensor<T> tensor(Tensor_type::placeHolder,
                         placeHolder_operation_index, false);

        unsigned long size = 1;
        for(auto elem : shape){
            size *= elem;
        }

        std::vector<T> data(size);

        Tensor_object<T> tensor_object(std::move(data), std::move(shape),
                                               Tensor_type::placeHolder, name, 0);
        std::unique_ptr<Tensor<T>> tensor_ptr = std::make_unique(tensor);
        Management<T>::add_placeHolder(tensor_ptr);
    }

    template<typename T>
    Tensor<T> generate_tensor<T>::placeHolder(std::vector<int> shape, std::string name){

        Tensor<T> tensor(Tensor_type::placeHolder,
                         placeHolder_operation_index, false);

        unsigned long size = 1;
        for(auto elem : shape)
            size *= elem;

        std::vector<T> data(size);

        Tensor_object<T> tensor_object(data, shape, Tensor_type::placeHolder, name, 0);
        std::unique_ptr<Tensor<T>> tensor_ptr = std::make_unique(tensor);
        Management<T>::add_placeHolder(tensor_ptr);
    }

}

#endif //CUBBYDNN_GENERATOR_TENSOR_HPP
