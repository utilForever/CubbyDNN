//
// Created by jwkim98 on 8/18/19.
//

#ifndef CUBBYDNN_TYPECAST_HPP
#define CUBBYDNN_TYPECAST_HPP
#include <cubbydnn/Tensors/Tensor.hpp>
#include <type_traits>
#include <unum/posit/posit>
#include <vector>

namespace CubbyDNN
{
template <typename T>
class TypeCast
{
    template <typename std::enable_if_t<std::is_floating_point<T>::value, T>* =
                  nullptr>
    static T* Cast(Tensor& tensor, T* destination)
    {
        T* castedPtr = static_cast<T*>(tensor.DataPtr);
        return castedPtr;
    }

    template <
        typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
    static T* Cast(Tensor& tensor, T* destination)
    {
        T* castedPtr = static_cast<T*>(tensor.DataPtr);
        return castedPtr;
    }

    template <size_t nbits, size_t es>
    static T* cast(Tensor& tensor, T* destination)
    {
        auto size = tensor.Info.Size();
        if constexpr (nbits == 8)
        {
            auto* castPtr = static_cast<uint8_t*>(tensor.DataPtr);
            sw::unum::posit<nbits, es> posit;
            for (size_t count = 0; count < size; ++count)
            {
                posit.set_raw_bits(*(castPtr + count));
                *(destination + count) = posit;
            }
        }
        else if constexpr (nbits == 16)
        {
            auto* castPtr = static_cast<uint16_t*>(tensor.DataPtr);
            sw::unum::posit<nbits, es> posit;
            for (size_t count = 0; count < size; ++count)
            {
                posit.set_raw_bits(*(castPtr + count));
                *(destination + count) = posit;
            }
        }
        else if constexpr (nbits == 32)
        {
            auto* castPtr = static_cast<uint32_t*>(tensor.DataPtr);
            sw::unum::posit<nbits, es> posit;
            for (size_t count = 0; count < size; ++count)
            {
                posit.set_raw_bits(*(castPtr + count));
                *(destination + count) = posit;
            }
        }
        else if constexpr (nbits == 64)
        {
            auto* castPtr = static_cast<uint64_t*>(tensor.DataPtr);
            sw::unum::posit<nbits, es> posit;
            for (size_t count = 0; count < size; ++count)
            {
                posit.set_raw_bits(*(castPtr + count));
                *(destination + count) = posit;
            }
        }
        else
        {
            /// Unsupported data type
            assert(false);
        }
    }
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TYPECAST_HPP
