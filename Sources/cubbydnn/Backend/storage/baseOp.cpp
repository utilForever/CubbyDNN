//
// Created by Justin on 18. 11. 9.
//
#include <iostream>
#include "Backend/storage/baseOp.h"

namespace cubby_dnn{

    template<typename T>
    emptyOp<T>::emptyOp() = default;

    template<typename T>
    //from Operation<T>, to This operation
    MatMul<T>::MatMul(Tensor<T> &tensor1, const Operation<T> &from1,
            Tensor<T> &tensor2, const Operation<T>& from2){
        typename Operation<T>::edge info1{tensor1, from1, *this};
        typename Operation<T>::edge info2{tensor2, from2, *this};
        this->edgeIn.emplace_back(info1);
        this->edgeIn.emplace_back(info2);
        //TODO: make return tensor
    }

    template<typename T, typename U>
    MatDot<T,U>::MatDot(Tensor<T> &tensor1, const Operation<T> &from1, U mul){
        typename Operation<T>::edge info1{tensor1, from1, *this};
        this->edgeIn.emplace_back(info1);
        //TODO: make return tensor
    }

    template<typename T>
    MatAdd<T>::MatAdd(Tensor<T> &tensor1, const Operation<T> &from1){
        typename Operation<T>::Edge info1{tensor1, from1, *this};
        this->edgeIn.emplace_back(info1);
        //TODO: make return tensor
    }

    template<typename T>
    MatSub<T>::MatSub(Tensor<T> &tensor1, const Operation<T> &from1){
        typename Operation<T>::Edge info1{tensor1, from1, *this};
        this->edgeIn.emplace_back(info1);
        //TODO: make return tensor
    }

    template<typename T>
    Reshape<T>::Reshape(Tensor<T> &tensor1, const Operation<T> &from1){

        };
}