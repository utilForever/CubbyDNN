#include <CubbyDNN/Node/NodeType.hpp>

namespace CubbyDNN::Node
{
NodeType::NodeType(const NodeType* _baseType, std::string_view _typeName)
    : baseType(_baseType), typeName(_typeName)
{
    // Do nothing
}

bool NodeType::operator==(const NodeType& rhs) const
{
    return typeName == rhs.typeName;
}

bool NodeType::IsBaseOf(const NodeType* _derivedType) const
{
    return IsBaseOf(this, _derivedType);
}

bool NodeType::IsDerivedFrom(const NodeType* _baseType) const
{
    return IsBaseOf(_baseType, this);
}

bool NodeType::IsExactlyBaseOf(const NodeType* _derivedType) const
{
    return IsExactlyBaseOf(this, _derivedType);
}

bool NodeType::IsExactlyDerivedFrom(const NodeType* _baseType) const
{
    return IsExactlyBaseOf(_baseType, this);
}

bool NodeType::IsBaseOf(const NodeType* _baseType, const NodeType* _derivedType)
{
    if (!_baseType)
    {
        return false;
    }

    for (const auto* base{ _derivedType }; base; ++base)
    {
        if (*base == *_baseType)
        {
            return true;
        }
    }

    return false;
}

bool NodeType::IsExactlyBaseOf(const NodeType* _baseType,
                               const NodeType* _derivedType)
{
    if (!_baseType)
    {
        return false;
    }

    if (!_derivedType)
    {
        return false;
    }

    for (const auto* base{ _derivedType->baseType }; base; ++base)
    {
        if (*base == *_baseType)
        {
            return true;
        }
    }

    return false;
}
}  // namespace CubbyDNN::Node