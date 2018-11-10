//
// Created by Justin on 18. 11. 8.
//

#ifndef CUBBYDNN_BASEOP_H
#define CUBBYDNN_BASEOP_H

#include "storage.h"
#include "Tensor_object.h"
#include "management.h"
#include <vector>
#include <variant>

namespace cubby_dnn{

    template<typename T>
    class Operation{
    protected:
        static void make_new_from(Tensor_object<T> tensor, const int from) {
            int this_id = Management<T>::add_op();
            Management<T>::add_edge(from, this_id, tensor)
        }

    private:
        Operation() {};///disable the constructor
    };

    template<typename T>
    class emptyOp: public Operation<T>{
    public:
        emptyOp();// do nothing
    };

    template<typename T>
    class MatMul: public Operation<T>{
    public:
        MatMul(Tensor_object<T> &tensor1, Tensor_object<T> &tensor2);
    };

    template<typename T, typename U>
    class MatDot: public Operation<T>{
    public:
        MatDot(Tensor_object<T> &tensor1, U mul);
    };

    template<typename T>
    class MatAdd: public Operation<T>{
    public:
        MatAdd(Tensor_object<T> &tensor1);
    };

    template<typename T>
    class MatSub: public Operation<T>{
    public:
        MatSub(Tensor_object<T> &tensor1);
    };

    template<typename T>
    class Reshape: public Operation<T>{
    public:
        Reshape(Tensor_object<T> &tensor1);
    };
}

#endif //CUBBYDNN_BASEOP_H
