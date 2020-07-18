// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTABLEUNIT_HPP
#define CUBBYDNN_COMPUTABLEUNIT_HPP

#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Units/UnitType.hpp>
#include <cubbydnn/Units/UnitMetadata.hpp>
#include <future>

namespace CubbyDNN::Graph
{
class ComputableUnit
{
public:
    //! \param subjectUnitId : id of the unit
    //! \param numberSystem : number system to use
    //! \param forwardInputMap : vector of input tensor for forward propagation
    //! \param backwardInputMap : vector of input tensor for back propagation
    //! \param forwardOutput : output of forward propagation
    //! \param backwardOutputMap : output of backward propagation
    ComputableUnit(UnitId subjectUnitId, NumberSystem numberSystem,
                   std::unordered_map<UnitId, Tensor> forwardInputMap,
                   std::unordered_map<UnitId, Tensor> backwardInputMap,
                   Tensor forwardOutput,
                   std::unordered_map<UnitId, Tensor> backwardOutputMap
        );
    virtual ~ComputableUnit() = default;

    ComputableUnit(const ComputableUnit& computableUnit) = delete;
    ComputableUnit(ComputableUnit&& computableUnit) noexcept;

    ComputableUnit& operator=(const ComputableUnit& computableUnit) = delete;
    ComputableUnit& operator=(ComputableUnit&& computableUnit) noexcept;

    UnitId Id() const
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

    //! vector of input tensors used to compute forward propagation
    std::unordered_map<UnitId, Tensor> ForwardInputMap;
    //! vector of output tensors used to compute back propagation
    std::unordered_map<UnitId, Tensor> BackwardInputMap;
    //! single output tensor of forward propagation
    Tensor ForwardOutput;
    //! single output tensor of back propagation
    std::unordered_map<UnitId, Tensor> BackwardOutputMap;

protected:
    UnitId m_unitId;
    /// UnitState m_objectPtr indicates execution state of ComputableUnit
    UnitState m_unitState;
    //! Number system for this unit to use
    NumberSystem m_numericType;
};
}; // namespace CubbyDNN

#endif  // CUBBYDNN_COMPUTABLEUNIT_HPP
