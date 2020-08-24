// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_COMPUTABLEUNIT_DECL_HPP
#define TAKION_GRAPH_COMPUTABLEUNIT_DECL_HPP

#include <Takion/Units/UnitType.hpp>
#include <Takion/Tensors/Tensor.hpp>
#include <future>

namespace Takion::Graph
{
template <typename T>
class ComputableUnit
{
public:
    //! \param subjectUnitId : id of the unit
    //! \param forwardInputMap : vector of input Tensor<T> for forward propagation
    //! \param backwardInputMap : vector of input Tensor<T> for back propagation
    //! \param forwardOutput : output of forward propagation
    //! \param backwardOutputMap : output of backward propagation
    //! \param internalTensorMap : map of the internally used tensors
    //! \param batchSize : batch size of the computableUnit
    ComputableUnit(UnitId subjectUnitId,
                   std::unordered_map<UnitId, Tensor<T>> forwardInputMap,
                   std::unordered_map<UnitId, Tensor<T>> backwardInputMap,
                   Tensor<T> forwardOutput,
                   std::unordered_map<UnitId, Tensor<T>> backwardOutputMap,
                   std::unordered_map<std::string, Tensor<T>> internalTensorMap,
                   std::size_t batchSize);
    virtual ~ComputableUnit() = default;

    ComputableUnit(const ComputableUnit<T>& computableUnit) = delete;
    ComputableUnit(ComputableUnit<T>&& computableUnit) noexcept;

    ComputableUnit<T>& operator=(const ComputableUnit<T>& computableUnit)
    = delete;
    ComputableUnit<T>& operator=(ComputableUnit<T>&& computableUnit) noexcept;

    [[nodiscard]] UnitId Id() const
    {
        return m_unitId;
    }

    //! Execute the Apply-propagating operation
    //! Throws runtime exception if unit is not ready to be executed
    //! This includes copying the result to input of next unit
    virtual void Forward() = 0;

    //! Executes the Forward activation asynchronously. Sets promise to 'true' if
    //! operation is completed
    //! \param promise : promise is set true internally if operation is completed
    virtual void AsyncForward(
        std::promise<bool> promise) = 0;
    //! Execute Backward-propagating operation
    //! Throws runtime exception if unit is not ready to be executed
    //! This includes copying the result to input of previous unit
    virtual void Backward() = 0;

    //! Executes the Backward  activation asynchronously. Sets promise to 'true'
    //! if operation is completed
    //! \param promise : promise is set true internally if operation is completed
    virtual void AsyncBackward(
        std::promise<bool> promise) = 0;

    //! Checks if forward propagation is ready
    //! \param cycle : cycle of current state
    //! \return : True if ready False if not
    [[nodiscard]] bool IsForwardReady(std::size_t cycle) const;

    //! Checks if forward propagation is ready
    //! \param cycle : cycle of current state
    //! \return : True if ready False if not
    [[nodiscard]] bool IsBackwardReady(std::size_t cycle) const;

    void UpdateForwardState();

    void UpdateBackwardState();

    void ResetState();

    virtual void ChangeBatchSize(std::size_t batchSize);

    T GetLoss()
    {
        return m_loss;
    }

    //! vector of input Tensor<T>s used to compute forward propagation
    std::unordered_map<UnitId, Tensor<T>> ForwardInputMap;
    //! vector of output Tensor<T>s used to compute back propagation
    std::unordered_map<UnitId, Tensor<T>> BackwardInputMap;
    //! single output Tensor<T> of forward propagation
    Tensor<T> ForwardOutput;
    //! single output Tensor<T> of back propagation
    std::unordered_map<UnitId, Tensor<T>> BackwardOutputMap;
    //! Tensors containing internal tensors
    std::unordered_map<std::string, Tensor<T>> InternalTensorMap;

    std::size_t BatchSize;

protected:

    UnitId m_unitId;
    /// UnitState m_objectPtr indicates execution state of ComputableUnit
    UnitState m_unitState;
    T m_loss = 0;
};
}; // namespace Takion

#endif  // takion_COMPUTABLEUNIT_HPP
