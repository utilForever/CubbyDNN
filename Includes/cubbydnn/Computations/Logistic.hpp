// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_LOGISTIC_HPP
#define CUBBYDNN_LOGISTIC_HPP

//! This file contains different implementations depending on numberSystems

#include <cubbydnn/Computations/Functions/ComputeTensor.hpp>
#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Units/HiddenComputableUnits/HiddenUnit.hpp>
#include <cubbydnn/Utils/GlobalHyperparams.hpp>
#include <cubbydnn/Utils/TypeCast.hpp>
#include <include/universal/posit/posit>
#include <iostream>
#include <type_traits>

namespace CubbyDNN
{
class LogisticBasic : public HiddenUnit
{
    void Compute() override
    {
        // TODO : separate types and call correct Calculate implementation
        assert(m_inputTensorVector.size() == 1 &&
               m_outputTensorVector.size() == 1);
        auto numberSystem = m_inputTensorVector.at(0).Info.GetNumberSystem();

        Tensor& sourceTensor = m_inputTensorVector.at(0);
        Tensor& destTensor = m_outputTensorVector.at(1);

        switch (numberSystem)
        {
            case NumberSystem::Float16:
                Calculate<float>(destTensor, sourceTensor);
                break;
            case NumberSystem::Float32:
                Calculate<float>(destTensor, sourceTensor);
                break;
            case NumberSystem::Float64:
                Calculate<double>(destTensor, sourceTensor);
                break;
            case NumberSystem::Int8:
                Calculate<int8_t>(destTensor, sourceTensor);
                break;
            case NumberSystem::Int16:
                Calculate<int16_t>(destTensor, sourceTensor);
                break;
            case NumberSystem::Int32:
                Calculate<int32_t>(destTensor, sourceTensor);
                break;
            case NumberSystem::Int64:
                Calculate<int64_t>(destTensor, sourceTensor);
                break;
            case NumberSystem::Posit8:
                Calculate<8, UnumBitConfig::Posit8_es>(destTensor,
                                                       sourceTensor);
                break;
            case NumberSystem::Posit16:
                Calculate<16, UnumBitConfig::Posit16_es>(destTensor,
                                                         sourceTensor);
                break;
            case NumberSystem::Posit32:
                Calculate<32, UnumBitConfig::Posit32_es>(destTensor,
                                                         sourceTensor);
                break;
            case NumberSystem::Posit64:
                Calculate<64, UnumBitConfig::Posit64_es>(destTensor,
                                                         sourceTensor);
                break;
        }
    }

    // TODO : separate implementations by bytes
    template <typename T, typename std::enable_if_t<
                              std::is_floating_point<T>::value, T>* = nullptr>
    void Calculate(Tensor& destTensor, Tensor& sourceTensor)
    {
        assert(destTensor.Info.Size() == sourceTensor.Info.Size());
        assert(destTensor.Info.GetNumberSystem() ==
               sourceTensor.Info.GetNumberSystem());
        auto size = destTensor.Info.Size();
        auto sourcePtr = TypeCast::CastFromTensor<T>(destTensor);
        auto destPtr = TypeCast::CastFromTensor<T>(sourceTensor);
        auto function = [](T& data) { return 1 / (1 + exp(data)); };
        ComputeTensor::BasicLoop<T>(destPtr, sourcePtr, function, size);
    }

    template <typename T, typename std::enable_if_t<std::is_integral<T>::value,
                                                    T>* = nullptr>
    void Calculate(Tensor& destTensor, Tensor& sourceTensor)
    {
        assert(destTensor.Info.Size() == sourceTensor.Info.Size());
        assert(destTensor.Info.GetNumberSystem() ==
               sourceTensor.Info.GetNumberSystem());
        auto size = destTensor.Info.Size();
        auto sourcePtr = TypeCast::CastFromTensor<T>(destTensor);
        auto destPtr = TypeCast::CastFromTensor<T>(sourceTensor);
        auto function = [](T& data) { return 1 / (1 + exp(data)); };
        ComputeTensor::BasicLoop<T>(destPtr, sourcePtr, function, size);
    }

    template <size_t nbits, size_t es>
    void Calculate(Tensor& destTensor, Tensor& sourceTensor)
    {
        assert(destTensor.Info.Size() == sourceTensor.Info.Size());
        assert(destTensor.Info.GetNumberSystem() ==
               sourceTensor.Info.GetNumberSystem());
        using positType = typename sw::unum::posit<nbits, es>;

        auto size = sourceTensor.Info.Size();
        positType sourceValue[size];
        positType destValue[size];
        TypeCast::CastFromTensor<nbits, es>(sourceTensor, sourceValue);

        auto function = [](positType& data){return positType(1) /
                (positType(1) + sw::unum::exp<nbits, es>(data));};
        // TODO : parallelize this for loop if possible

        ComputeTensor::BasicLoop<positType>(destValue, sourceValue, function, size);
        TypeCast::CastToTensor<nbits, es>(destTensor, destValue);
    }
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_LOGISTIC_HPP