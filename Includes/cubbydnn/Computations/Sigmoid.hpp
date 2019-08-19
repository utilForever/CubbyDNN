//
// Created by jwkim98 on 8/15/19.
//

#ifndef CUBBYDNN_SIGMOID_HPP
#define CUBBYDNN_SIGMOID_HPP

//! This file contains different implementations depending on numberSystems

#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Utils/TypeCast.hpp>
#include <iostream>
#include <type_traits>
#include <unum/posit/posit>

#define exponent 2.718281828459045235360287471352662497757247093699959574966

namespace CubbyDNN
{
class ComputeLogistic
{
    template <typename T>
    static void Calculate(Tensor& tensor)
    {
        std::cout << "This type is not supported" << std::endl;
        assert(false);
    }

    //! We separate implementation of double
    //! since double wouldn't show good performance in some devices (such as
    //! GPU)
    template <>
    void Calculate<double>(Tensor& tensor)
    {
    }

    template <>
    void Calculate<int64_t>(Tensor& tensor)
    {
    }

    template <typename T, typename std::enable_if_t<
                              std::is_floating_point<T>::value, T>* = nullptr>
    void Calculate(Tensor& tensor)
    {
    }

    template <typename T, typename std::enable_if_t<std::is_integral<T>::value,
                                                    T>* = nullptr>
    void Calculate(Tensor& tensor)
    {
    }

    template <size_t nbits, size_t es>
    void Calculate(Tensor& destTensor, Tensor& sourceTensor)
    {
        using positType = typename sw::unum::posit<nbits, es>;

        auto size = sourceTensor.Info.Size();
        positType sourceValue[size];
        positType destValue[size];
        TypeCast<positType>::CastFromTensor<nbits, es>(sourceTensor, sourceValue);

        // TODO : parallelize this for loop if possible
        for (size_t count = 0; count < size; ++count)
        {
            destValue[count] =
                positType(1) /
                (positType(1) + sw::unum::exp<nbits, es>(-sourceValue[count]));
        }

        TypeCast<positType>::CastToTensor<nbits, es>(destTensor, destValue);
    }
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_SIGMOID_HPP