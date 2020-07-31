// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_UNITMETADATA_HPP
#define CUBBYDNN_UNITMETADATA_HPP

#include <cubbydnn/Units/UnitType.hpp>
#include <cubbydnn/Utils/Shape.hpp>
#include <cubbydnn/Utils/Declarations.hpp>
#include <cubbydnn/Computations/Initializers/InitializerType.hpp>
#include <vector>
#include <memory>
#include <unordered_map>

namespace CubbyDNN::Graph
{
class Parameter
{
public:
    Parameter() = default;
    ~Parameter() = default;

    Parameter(const Parameter& parameterPack) = default;
    Parameter(Parameter&& parameterPack) noexcept = default;
    Parameter& operator=(const Parameter& parameterPack) = default;
    Parameter& operator=(Parameter&& parameterPack) noexcept = default;

    Parameter(std::unordered_map<std::string, int> integerParams,
              std::unordered_map<std::string, float> floatingPointParams,
              std::unordered_map<std::string, std::string> stringParams);

    Parameter(std::unordered_map<std::string, int> integerParams);
    Parameter(std::unordered_map<std::string, float> floatingPointParams);
    Parameter(std::unordered_map<std::string, std::string> stringParams);

    [[nodiscard]] int GetIntegerParam(const std::string& name) const;

    [[nodiscard]] float GetFloatingPointParam(const std::string& name) const;

    [[nodiscard]] std::string GetStringParam(const std::string& name) const;

private:
    std::unordered_map<std::string, int> m_integerParameters;
    std::unordered_map<std::string, float> m_floatingPointParameters;
    std::unordered_map<std::string, std::string> m_stringParameters;
};

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
    std::unordered_map<std::string, Tensor> m_internalTensorMap;

    //! Input names and their shape
    std::unordered_map<std::string, Shape> m_inputShapeMap;
    Shape m_outputShape;
    //! Input names and their unitId
    //! key of inputShapeMap and inputUnitIdMap must be identical
    std::unordered_map<std::string, UnitId> m_inputUnitMap;
    std::vector<UnitId> m_outputUnitIdVector;

public:
    NumberSystem NumericType;
    Compute::Device Device;
    Parameter Params = Parameter();
};
}
#endif
