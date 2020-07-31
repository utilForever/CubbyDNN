// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Takion/Utils/Parameter.hpp>

namespace Takion
{
Parameter::Parameter(std::unordered_map<std::string, int> integerParams,
                     std::unordered_map<std::string, float> floatingPointParams,
                     std::unordered_map<std::string, std::string> stringParams)
    : m_integerParameters(std::move(integerParams)),
      m_floatingPointParameters(std::move(floatingPointParams)),
      m_stringParameters(std::move(stringParams))
{
}

Parameter::Parameter(std::unordered_map<std::string, int> integerParams)
    : m_integerParameters(std::move(integerParams))
{
}

Parameter::Parameter(std::unordered_map<std::string, float> floatingPointParams)
    : m_floatingPointParameters(std::move(floatingPointParams))
{
}

Parameter::Parameter(std::unordered_map<std::string, std::string> stringParams)
    : m_stringParameters(std::move(stringParams))
{
}

int Parameter::GetIntegerParam(const std::string& name) const
{
    return m_integerParameters.at(name);
}

float Parameter::GetFloatingPointParam(const std::string& name) const
{
    return m_floatingPointParameters.at(name);
}

std::string Parameter::GetStringParam(const std::string& name) const
{
    return m_stringParameters.at(name);
}
}
