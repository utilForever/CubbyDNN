// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTABLEUNIT_HPP
#define CUBBYDNN_COMPUTABLEUNIT_HPP

#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Tensors/TensorInfo.hpp>
#include <cubbydnn/Units/UnitType.hpp>
#include <cubbydnn/Computations/Initializers/InitializerType.hpp>
#include <future>

namespace CubbyDNN::Graph
{
class ComputableUnit
{
public:
    //! \param unitId : id of the unit
   //! \param numberSystem : number system to use
    //! \param forwardInputVector : vector of input tensor for forward propagation
    //! \param backwardInputVector : vector of input tensor for back propagation
    //! \param forwardOutput : output of forward propagation
    //! \param backwardOutputVector : output of backward propagation
    ComputableUnit(UnitId unitId, NumberSystem numberSystem,
                   std::vector<Tensor> forwardInputVector,
                   std::vector<Tensor> backwardInputVector,
                   Tensor forwardOutput,
                   std::vector<Tensor> backwardOutputVector
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

    //! Execute the Forward-propagating operation
    //! Throws runtime exception if unit is not ready to be executed
    //! This includes copying the result to input of next unit
    virtual void Forward() = 0;
    virtual void AsyncForward(
        std::promise<bool> promise) = 0;
    //! Execute Backward-propagating operation
    //! Throws runtime exception if unit is not ready to be executed
    //! This includes copying the result to input of previous unit
    virtual void Backward() = 0;
    virtual void AsyncBackward(
        std::promise<bool> promise) = 0;

    //! Initializes internal tensors
    virtual void Initialize(
        const std::vector<std::unique_ptr<Initializer>>& initializerVector) = 0;

    //! Checks if forward propagation is ready
    //! \param cycle : cycle of current state
    //! \return : True if ready False if not
    [[nodiscard]] bool IsForwardReady(std::size_t cycle) const;

    //! Checks if forward propagation is ready
    //! \param cycle : cycle of current state
    //! \return : True if ready False if not
    [[nodiscard]] bool IsBackwardReady(std::size_t cycle) const;

    //! vector of input tensors used to compute forward propagation
    std::vector<Tensor> ForwardInputVector;
    //! vector of output tensors used to compute back propagation
    std::vector<Tensor> BackwardInputVector;
    //! single output tensor of forward propagation
    Tensor ForwardOutput;
    //! single output tensor of back propagation
    std::vector<Tensor> BackwardOutputVector;

protected:
    void m_updateForwardState();

    void m_updateBackwardState();

    UnitId m_unitId;
    /// UnitState m_objectPtr indicates execution state of ComputableUnit
    UnitState m_unitState;
    //! Number system for this unit to use
    NumberSystem m_numberSystem;
};
}; // namespace CubbyDNN

#endif  // CUBBYDNN_COMPUTABLEUNIT_HPP
