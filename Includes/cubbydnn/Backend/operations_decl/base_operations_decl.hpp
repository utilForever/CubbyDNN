
#ifndef CUBBYDNN_BASE_OPERATIONS_DECL_HPP
#define CUBBYDNN_BASE_OPERATIONS_DECL_HPP

#include "Backend/util/Tensor_container.hpp"

namespace cubby_dnn{
    enum class operation_type{
        generate,
        basic,
        final
    };

    template<typename T>
    class Operation{
    protected:
        explicit Operation();
    private:
        operation_type op_type;
        std::vector<std::shared_ptr<Tensor<T>>> input_tensor_vect;
        std::vector<std::shared_ptr<Tensor<T>>> output_tensor_vect;
    };

    template<typename T>
    class Mat_mul_op: public Operation<T>{
    public:
        explicit Mat_mul_op(std::shared_ptr<Tensor<T>> tensor1, std::shared_ptr<Tensor<T>> tensor2,
                std::shared_ptr<Tensor<T>> output_tensor);

        explicit Mat_mul_op(std::vector<std::shared_ptr<Tensor<T>>> tensor_vect,
                std::shared_ptr<Tensor<T>> output_tensor);
    };

    template<typename T>
    class Mat_add_op: public Operation<T>{
    public:
        explicit Mat_add_op(std::shared_ptr<Tensor<T>> tensor1, std::shared_ptr<Tensor<T>> tensor2,
                std::shared_ptr<Tensor<T>> output_tensor);

        explicit Mat_add_op(std::vector<std::shared_ptr<Tensor<T>>> tensor_vect, std::shared_ptr<Tensor<T>> output_tensor);
    };

    template<typename T>
    class Mat_dot_op: public Operation<T>{
    public:
        explicit Mat_dot_op(std::shared_ptr<Tensor<T>> tensor1, std::shared_ptr<Tensor<T>> identity_tensor,
                std::shared_ptr<Tensor<T>> output_tensor);
    };

    template<typename T>
    class Reshape_op: public Operation<T>{
    public:
        explicit Reshape_op(std::shared_ptr<Tensor<T>> tensor1, std::shared_ptr<Tensor<T>> output_tensor);
    private:

    };


    template<typename T>
    class placeHolder: public Operation<T>{
    public:
        explicit placeHolder(const std::vector<int>& shape, std::shared_ptr<Tensor<T>> output_tensor);
    private:
        std::vector<int> shape;
    };


    template<typename T>
    class weight: public Operation<T>{
    public:
        explicit weight(const std::vector<int>& shape, std::shared_ptr<Tensor<T>> output_tensor);
    private:
        std::vector<int> shape;
    };

}

#endif