//
// Created by jwkim98 on 8/15/19.
//

#ifndef CUBBYDNN_SIGMOID_HPP
#define CUBBYDNN_SIGMOID_HPP

//! This file contains different implementations depending on numberSystems

#include <cubbydnn/Tensors/Tensor.hpp>
#include <iostream>
#include <type_traits>
#include <unum/posit/posit>

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

    template <size_t a, size_t b, size_t numberSystemByteSize>
    void CalculatePosit(Tensor& tensor)
    {
        if constexpr (numberSystemByteSize == 8)
        {
            auto* castPtr = (uint8_t*)tensor.DataPtr;

            for (size_t count = 0; count < tensor.Info.ByteSize(); ++count)
            {
            }

            sw::unum::posit<a, b> posit;
        }
        else if constexpr (numberSystemByteSize == 16)
        {
        }
        else if constexpr (numberSystemByteSize == 32)
        {
        }
    }
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_SIGMOID_HPP