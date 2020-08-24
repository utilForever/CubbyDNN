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
#include <Takion/Utils/Loaders/Loader.hpp>
#include <Takion/Utils/TensorData.hpp>
#include <memory>
#include <vector>
#include <map>

namespace Takion::FrontEnd
{
template <typename T>
class Model
{
public:
    Model(Compute::Device device, std::size_t batchSize = 1);

    void SetDevice(Compute::Device device);

    AbsTensor<T> Fetcher(const Shape& shape,
                         std::unique_ptr<Util::Loader<T>> loaderFunction,
                         std::string name = "Fetcher");

    AbsTensor<T> Fetcher(const Shape& shape, std::string name = "Fetcher");

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

    AbsTensor<T> Sigmoid(AbsTensor<T> source, std::string name = "");

    AbsTensor<T> SoftMax(AbsTensor<T> source, std::string name = "");

    AbsTensor<T> MSE(AbsTensor<T> prediction, AbsTensor<T> label,
                     std::string name);

    AbsTensor<T> CrossEntropy(AbsTensor<T> prediction, AbsTensor<T> label,
                              std::string name);

    void Compile(std::string optimizer, Parameter optimizerParams);

    void Train();

    void Train(std::map<AbsTensor<T>, std::vector<T>> inputDataMap,
               AbsTensor<T> labelUnit, std::vector<T> label);

    void Predict();

    void Predict(std::map<AbsTensor<T>, std::vector<T>> inputDataMap);

    void Predict(std::map<AbsTensor<T>, std::vector<T>> inputDataMap,
                 AbsTensor<T> labelUnit, std::vector<T> label);

    void Fit(std::size_t epochs);

    [[nodiscard]] Util::TensorData<T> Output(
        AbsTensor<T> absTensor) const;

    [[nodiscard]] T GetLoss(AbsTensor<T> lossId);


    void ChangeBatchSize(std::size_t batchSize)
    {
        m_unitManager.ChangeBatchSize(batchSize);
        m_batchSize = batchSize;
    }

    void ChangeLoader(AbsTensor<T> loaderId,
                      std::function<std::vector<T>()> loaderFunction);

private:

    void m_appendSubjectUnitToPreviousOutput(const UnitId& subjectUnit,
                                             const UnitId& previousUnit);

    Compute::Device m_device;
    Engine::UnitManager<T> m_unitManager;
    std::size_t m_batchSize;
    std::size_t m_id = 0;
};
}

#endif
