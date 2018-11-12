//
// Created by Justin on 18. 11. 9.
//
#include <iostream>
#include "Backend/util/baseOp.hpp"

namespace cubby_dnn{

    template<typename T>
    emptyOp<T>::emptyOp() = default;

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