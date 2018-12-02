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
class generate
{
 public:
    // TODO: think about initialization methods
    enum class initializer
    {
        default_state
    };

    // TODO: think about ways to put data stream through placeholders

    static tensor<T> placeholder(const tensor_shape &shape, stream<T> &stream,
                                 const std::string &name = "placeholder");

    static tensor<T> weight(const tensor_shape &shape, bool trainable = true,
                            const std::string &name = "weight");

    static tensor<T> filter(const tensor_shape &shape, bool trainable = true,
                            const std::string &name = "filter");

 private:
    static tensor<T> get_default_tensor()
    {
        // default tensor to return when error occurs
        return tensor<T>(tensor_type::None, tensor_shape(), -1,
                         "default Tensor due to error");
    }
};

template <typename T>
class operate : protected tensor<T>
{
    // friend class Operation_management;
 public:
    static tensor<T> mat_mul(tensor <T> &tensor1, tensor <T> &tensor2,
                             const std::string &name = "mat_mul");

    static tensor<T> mad_add(tensor <T> &tensor1, tensor <T> &tensor2,
                             const std::string &name = "mat_add");

    static tensor<T> mat_dot(tensor <T> &tensor1, T multiplier,
                             const std::string &name = "mat_dot");

    static tensor<T> reshape(tensor<T> &tensor1, const tensor_shape &shape,
                             const std::string &name = "reshape");

    static tensor<T> one_hot(tensor <T> &tensor1, unsigned long size,
                             const std::string &name = "one_hot");


 private:
    static tensor<T> get_default_tensor()
    {
        // default tensor to return when error occurs
        return tensor<T>(tensor_type::None, tensor_shape(), -1,
                         "default Tensor due to error");
    }
};

template <typename T>
class Final
{
    friend class tensor<T>;

 public:
    static void wrapper(tensor<T> &tensor1,
                        const std::string &name = "wrapper");
};
}  // namespace cubby_dnn

#endif  // CUBBYDNN_GENERATE_TENSOR_HPP