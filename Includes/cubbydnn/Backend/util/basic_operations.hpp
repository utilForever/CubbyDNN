//
// Created by jwkim on 18. 11. 13.
//

#ifndef CUBBYDNN_BASE_OPERATIONS_HPP
#define CUBBYDNN_BASE_OPERATIONS_HPP

#include <Backend/util_decl/Tensor_container_decl.hpp>
#include <Backend/util_decl/base_operations_decl.hpp>
#include <iostream>

namespace cubby_dnn{
    template<typename T>
    void emptyOp<T>::print(){
        std::cout<<"emptyOp"<<std::endl;
    }

    //TODO: implement these constructors! (Add new operation and tensor on adj matrix)

    template<typename T>
    MatMul<T>::MatMul(Tensor_container<T> &tensor1, Tensor_container<T> &tensor2){

        //TODO: make return tensor
    }

    template<typename T, typename U>
    MatDot<T,U>::MatDot(Tensor_container<T> &tensor1, U mul){

        //TODO: make return tensor
    }

    template<typename T>
    MatAdd<T>::MatAdd(Tensor_container<T> &tensor1){

        //TODO: make return tensor
    }

    template<typename T>
    MatSub<T>::MatSub(Tensor_container<T> &tensor1){

        //TODO: make return tensor
    }

    template<typename T>
    Reshape<T>::Reshape(Tensor_container<T> &tensor1) {

        //TODO: make return tensor
    }
}
#endif //CUBBYDNN_BASE_OPERATIONS_HPP
