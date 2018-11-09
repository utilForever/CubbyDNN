//
// Created by Justin on 18. 11. 8.
//

#ifndef CUBBYDNN_BASEOP_H
#define CUBBYDNN_BASEOP_H

#include "storage.h"
#include "backend.h"
#include <vector>
#include <variant>

namespace cubby_dnn{


    template<typename T>
    class Operation{
    protected:
        class edge{
        public:
            edge(Tensor<T> tensor, Operation<T> &from, Operation<T> &to) {
                this->tensor = tensor;
                this->from = from;
                this->to = to;
            }
            Operation<T> from;
            Operation<T> to;
            Tensor<T> tensor;
        };
        std::vector<edge> edgeIn; //edges as inputs of this operation
        std::vector<edge> edgeOut; //edges as outputs of this operation
    private:
        //disable copy constructor (NOT ALLOWED)
        Operation(Operation<T>& rhs) = default;
        //disable move operator (NOT ALLOWED)
        Operation<T>& operator=(Operation<T> &rhs) = default;
        int operationId;
    };

    template<typename T>
    class emptyOp: public Operation<T>{
    public:
        emptyOp();
    };

    template<typename T>
    class MatMul: public Operation<T>{
    public:
        MatMul(Tensor<T> tensor1, const Operation<T> &from1,
                Tensor<T> tensor2, const Operation<T>& from2);
    };

    template<typename T, typename U>
    class MatDot: public Operation<T>{
    public:
        MatDot(Tensor<T> tensor1, const Operation<T> &from1, U mul);
    };

    template<typename T>
    class MatAdd: public Operation<T>{
    public:
        MatAdd(Tensor<T> tensor1, const Operation<T> &from1);
    };

    template<typename T>
    class MatSub: public Operation<T>{
    public:
        MatSub(Tensor<T> tensor1, const Operation<T> &from1);
    };

    template<typename T>
    class Reshape: public Operation<T>{
    public:
        Reshape(Tensor<T> tensor1, const Operation<T> &from1);
    };
}

#endif //CUBBYDNN_BASEOP_H
