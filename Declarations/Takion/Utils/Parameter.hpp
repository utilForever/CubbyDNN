// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_PARAMETER_HPP
#define CUBBYDNN_PARAMETER_HPP

#include <unordered_map>

namespace Takion
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
}

#endif
