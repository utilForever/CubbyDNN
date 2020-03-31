//! Copyright (c) 2019 Chris Ohk, Justin Kim

//! We are making my contributions/submissions to this project solely in our
//! personal capacity and are not conveying any rights to any intellectual
//! property of any third parties.

#ifndef CUBBYDNN_ENGINE_HPP
#define CUBBYDNN_ENGINE_HPP

#include <cubbydnn/Engine/TaskWrapper.hpp>
#include <cubbydnn/Units/CopyUnit.hpp>
#include <cubbydnn/Units/SinkComputableUnits/SinkUnit.hpp>
#include <cubbydnn/Units/SourceComputableUnits/SourceUnit.hpp>
#include <cubbydnn/Utils/SharedPtr.hpp>
#include <cubbydnn/Utils/SpinLockQueue.hpp>
#include <vector>

namespace CubbyDNN
{
//! Singleton class for maintaining threads that execute the program
class Graph
{
public:
    Graph(NumberSystem numberSystem);
    //! Execute the graph using single thread
    void Predict(std::size_t epochs);

    UnitId PlaceHolder(const Shape& shape);
    //! \param input : unit ID of previous unit
    //! \param units : size of output perceptrons
    //! \param activation : type of activation to use
    //! \param kernelInitializer : initializer of the kernel
    //! \param biasInitializer : initializer of the bias
    //! \param dropoutRate : percentage of units to dropout
    UnitId Dense(const UnitId& input, std::size_t units,
                 Activation activation, InitializerType kernelInitializer,
                 InitializerType biasInitializer, float dropoutRate = 0.0);

    UnitId Reshape(const UnitId& input, const Shape& shape);

    //! OptimizerType, Loss function
    void Compile(UnitId unitId, OptimizerType optimizer, Loss loss);

    //! Trains the graph with given optimizer and loss function
    void Fit(std::size_t epochs);

    void Predict(void* input, void* output, int workers);

private:
    void m_getExecutionOrder(UnitId subjectUnit,
                             std::vector<std::vector<UnitId>>& executionOrder,
                             int depth);

    /// True if this engine is active false otherwise
    bool m_active = true;
    std::vector<std::vector<UnitId>> m_executionOrder;

    std::vector<SharedPtr<ComputableUnit>> m_sourceUnitVector;
    SharedPtr<ComputableUnit> m_sinkUnit;
    std::vector<SharedPtr<ComputableUnit>> m_hiddenUnitVector;
    std::vector<SharedPtr<CopyUnit>> m_sourceCopyUnitVector;
    std::vector<SharedPtr<CopyUnit>> m_hiddenCopyUnitVector;

    /// number of epochs to run the graph
    /// If stateNum reaches this, that unit will be no longer computed
    std::size_t m_maxEpochs = 0;
    std::atomic_bool m_ready = false;
    NumberSystem m_numberSystem;
    OptimizerType m_optimizer = OptimizerType::Adam;
};
} // namespace CubbyDNN

#endif  // CAPTAIN_THREADMANAGER_HPP
