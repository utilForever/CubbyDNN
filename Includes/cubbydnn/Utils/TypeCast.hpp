// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TYPECAST_HPP
#define CUBBYDNN_TYPECAST_HPP
#include <cubbydnn/Tensors/Tensor.hpp>
#include <type_traits>
#include <unum/posit/posit>
#include <vector>

namespace CubbyDNN
{
//! Set of static functions for casting tensor(universal type wrapper)
//! to supported types
template <typename T>
class TypeCast
{
    /// Converts tensor to floating point types
    template <typename std::enable_if_t<std::is_floating_point<T>::value, T>* =
                  nullptr>
    static T* CastFromTensor(Tensor& tensor)
    {
        T* castedPtr = static_cast<T*>(tensor.DataPtr);
        return castedPtr;
    }

    /// Converts tensor to integer types
    template <
        typename std::enable_if_t<std::is_integral<T>::value, T>* = nullptr>
    static T* CastFromTensor(Tensor& tensor)
    {
        T* castedPtr = static_cast<T*>(tensor.DataPtr);
        return castedPtr;
    }

    /// Converts tensor to posit types and stores it in destination
    template <size_t nbits, size_t es>
    static void CastFromTensor(const Tensor& tensor,
                               sw::unum::posit<nbits, es>* destination)
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
            std::cout << "Unsupported type" << std::endl;
            assert(false);
        }
    }

    template <size_t nbits, size_t es>
    static void CastToTensor(Tensor& tensor, sw::unum::posit<nbits, es>* source)
    {
        auto numberSystemByteSize = tensor.Info.GetNumberSystemByteSize();

        if constexpr (nbits == 8)
        {
            auto destinationPtr = static_cast<uint8_t*>(tensor.DataPtr);
            for (size_t count = 0; count < numberSystemByteSize; ++count)
            {
                *(destinationPtr + count) = *(source + count).get();
            }
        }
        else if constexpr (nbits == 16)
        {
            auto destinationPtr = static_cast<uint16_t*>(tensor.DataPtr);
            for (size_t count = 0; count < numberSystemByteSize; ++count)
            {
                *(destinationPtr + count) = *(source + count).get();
            }
        }
        else if constexpr (nbits == 32)
        {
            auto destinationPtr = static_cast<uint32_t*>(tensor.DataPtr);
            for (size_t count = 0; count < numberSystemByteSize; ++count)
            {
                *(destinationPtr + count) = *(source + count).get();
            }
        }
        else if constexpr (nbits == 64)
        {
            auto destinationPtr = static_cast<uint64_t*>(tensor.DataPtr);
            for (size_t count = 0; count < numberSystemByteSize; ++count)
            {
                *(destinationPtr + count) = *(source + count).get();
            }
        }
    }
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_TYPECAST_HPP
