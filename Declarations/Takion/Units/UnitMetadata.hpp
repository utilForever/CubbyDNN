// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_UNITMETADATA_HPP
#define TAKION_UNITMETADATA_HPP

#include <Takion/Units/UnitType.hpp>
#include <Takion/Utils/Shape.hpp>
#include <Takion/Utils/Declarations.hpp>
#include <Takion/Utils/Parameter.hpp>
#include <Takion/Computations/Initializers/InitializerType.hpp>
#include <vector>
#include <memory>
#include <unordered_map>

namespace takion::Graph
{
template <typename T>
class UnitMetaData
{
public:
    UnitMetaData(
        UnitId unitId,
        std::unordered_map<std::string, Shape> internalVariableShapeMap,
        std::unordered_map<std::string, std::unique_ptr<Initializer>>
        initializerMap,
        std::unordered_map<std::string, Shape> inputShapeMap, Shape outputShape,
        std::unordered_map<std::string, UnitId> inputUnitIdMap,
        NumberSystem numericType, Compute::Device device,
        Parameter params = Parameter());

    ~UnitMetaData() = default;
    UnitMetaData(const UnitMetaData& unitMetaData) = delete;

    UnitMetaData(UnitMetaData&& unitMetaData) noexcept;
    UnitMetaData& operator=(const UnitMetaData& unitMetaData) = delete;
    UnitMetaData& operator=(UnitMetaData&& unitMetaData) noexcept;

    void AppendOutputUnitId(UnitId unitId);

    void SetOutputUnitIdVector(std::vector<UnitId> unitIdVector);

    //! Used to add internal tensor if required
    //! \param key : key to store the tensor
    //! \param tensor: tensor to store
    void AddInternalTensor(const std::string& key, Tensor tensor);

    [[nodiscard]] UnitId Id() const;

    [[nodiscard]] const Tensor& GetInternalTensor(const std::string& key) const;

    [[nodiscard]] Shape GetInputShape(const std::string& key) const;

    [[nodiscard]] UnitId GetInputUnitId(const std::string& key) const;

    [[nodiscard]] Shape OutputShape() const;

    [[nodiscard]] std::unordered_map<std::string, UnitId>
    InputUnitMap() const;

    [[nodiscard]] std::vector<UnitId> OutputUnitVector() const;

    [[nodiscard]] const std::unique_ptr<Initializer>& GetInitializer(
        const std::string& name) const;

    [[nodiscard]] Shape GetInternalVariableShape(const std::string& name) const;

private:
    UnitId m_unitId;
    std::unordered_map<std::string, Shape> m_internalVariableShapeMap;
    std::unordered_map<std::string, std::unique_ptr<Initializer>>
    m_initializerMap;
    std::unordered_map<std::string, Tensor<T>> m_internalTensorMap;

    //! Input names and their shape
    std::unordered_map<std::string, Shape> m_inputShapeMap;
    Shape m_outputShape;
    //! Input names and their unitId
    //! key of inputShapeMap and inputUnitIdMap must be identical
    std::unordered_map<std::string, UnitId> m_inputUnitMap;
    std::vector<UnitId> m_outputUnitIdVector;

public:
    Compute::Device Device;
    Parameter Params = Parameter();
};
}
#endif
