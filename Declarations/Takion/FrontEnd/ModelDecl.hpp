// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_FRONTEND_MODEL_DECL_HPP
#define TAKION_FRONTEND_MODEL_DECL_HPP

#include <Takion/FrontEnd/AbsTensorDecl.hpp>
#include <Takion/Computations/Device.hpp>
#include <Takion/Computations/Initializers/InitializerType.hpp>
#include <Takion/Engine/UnitManager.hpp>
#include <Takion/Utils/Parameter.hpp>
#include <Takion/Utils/Shape.hpp>
#include <memory>
#include <vector>


namespace Takion::FrontEnd
{
template <typename T>
class Model
{
public:
    Model(Compute::Device device, std::size_t batchSize = 1);

    void SetDevice(Compute::Device device);

    AbsTensor<T> Constant(const Shape& shape, std::vector<T> data,
                          std::string name);

    AbsTensor<T> Dense(AbsTensor<T> source, unsigned numUnits,
                       std::unique_ptr<Compute::Initializer<T>>
                       weightInitializer = std::make_unique<Compute::HeNormal<T>
                       >(),
                       std::unique_ptr<Compute::Initializer<T>>
                       biasInitializer = std::make_unique<Compute::HeNormal<T>
                       >(),
                       std::string name = "");

    AbsTensor<T> ReLU(AbsTensor<T> source,
                      std::string name = "");

    AbsTensor<T> SoftMax(AbsTensor<T> source, std::string name = "");

    void MSE(AbsTensor<T> prediction, AbsTensor<T> label, std::string name);

    void Compile(std::string optimizer, Parameter optimizerParams);

    void Fit(std::size_t epochs);

private:
    Compute::Device m_device;
    Engine::UnitManager<T> m_unitManager;
    std::size_t m_batchSize;
    std::size_t m_id = 0;
};
}

#endif
