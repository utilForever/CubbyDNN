//
// Created by Justin on 18. 11. 13.
//

#ifndef CUBBYDNN_GENERATE_TENSOR_HPP
#define CUBBYDNN_GENERATE_TENSOR_HPP

#include "Tensor_container_decl.hpp"
#include "stream_decl.hpp"

namespace cubby_dnn
{
template <typename T>
class Generate
{
    friend class Tensor<T>;

 public:
    // TODO: think about initialization methods
    enum class initializer
    {
        default_state
    };

    static const int placeHolder_operation_index = -1;

    static const int max_dim = 3;

    // TODO: think about ways to put data stream through placeholders

    static Tensor<T> placeHolder(const std::vector<int> &shape,
                                 Stream<T> stream,
                                 const std::string &name = "PlaceHolder");

    static Tensor<T> weight(const std::vector<int> &shape,
                            bool trainable = true,
                            const std::string &name = "weight");

    static Tensor<T> filter(const std::vector<int> &shape,
                            bool trainable = true,
                            const std::string &name = "filter");

 private:

    static Tensor<T> get_default_tensor()
    {
        // default tensor to return when error occurs
        return Tensor<T>(Tensor_type::None, std::vector<int>(), -1,
                         "default Tensor due to error");
    }
};

template <typename T>
class Operate
{
    friend class Tensor<T>;
    // friend class Operation_management;

 public:
    static Tensor<T> matMul(Tensor<T> &tensor1, Tensor<T> &tensor2,
                            const std::string &name = "matMul_op");

    static Tensor<T> matAdd(Tensor<T> &tensor1, Tensor<T> &tensor2,
                            const std::string &name = "matAdd_op");

    static Tensor<T> matDot(Tensor<T> &tensor1, T multiplier,
                            const std::string &name = "matDot_op");

    static Tensor<T> reshape(Tensor<T> &tensor1, const std::vector<int> &shape,
                             const std::string &name = "reshape_op");

 private:
    static Tensor<T> get_default_tensor()
    {
        // default tensor to return when error occurs
        return Tensor<T>(Tensor_type::None, std::vector<int>(), -1,
                         "default Tensor due to error");
    }

    static bool is_valid_tensor(const std::vector<Tensor<T>> &tensor_list){
        //Tensors should always have right shape, or empty shape
        for(Tensor<T> tensor : tensor_list){
            if(!tensor.is_valid())
                return false;
        }
        return true;
    }
};
}  // namespace cubby_dnn

#endif  // CUBBYDNN_GENERATE_TENSOR_HPP