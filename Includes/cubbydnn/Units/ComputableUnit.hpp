// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTABLEUNIT_HPP
#define CUBBYDNN_COMPUTABLEUNIT_HPP

#include <atomic>
#include <cubbydnn/Tensors/Tensor.hpp>
#include <cubbydnn/Tensors/TensorInfo.hpp>

namespace CubbyDNN
{
class ComputableUnit
{
 public:
    //! \param unitType : type of the unit
    ComputableUnit(UnitType unitType);

    //! Constructor
    virtual ~ComputableUnit() = default;

    ComputableUnit(const ComputableUnit& computableUnit) = delete;
    ComputableUnit(ComputableUnit&& other) noexcept;

    ComputableUnit& operator=(ComputableUnit&& other) noexcept;
    ComputableUnit& operator=(const ComputableUnit& computableUnit) = delete;

    //! Gets whether if executableUnit is ready to be executed
    //! \return : whether corresponding unit is ready to be executed
    virtual bool IsReady() = 0;

    //! Method that is executed on the engine
    //! This method must be called after checking computation is ready
    virtual void Compute() = 0;

    //! Called after computation for releasing the unit after computation
    //! Increments the stateNum and marks IsBusy as false
    void ReleaseUnit();

    //! Gets reference of the atomic state counter for atomic comparison
    //! of state counter
    //! \return : reference of the state counter
    std::size_t GetStateNum() const
    {
        return m_unitState.StateNum.load(std::memory_order_acquire);
    }

    UnitType Type = UnitType::Undefined;

 protected:
    /// UnitState m_objectPtr indicates execution state of ComputableUnit
    UnitState m_unitState;

};
};  // namespace CubbyDNN

#endif  // CUBBYDNN_COMPUTABLEUNIT_HPP
