// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_LOSSUNIT_HPP
#define TAKION_GRAPH_LOSSUNIT_HPP

namespace Takion::Graph
{
template <typename T>
class LossUnit
{
public:
    virtual ~LossUnit() = default;
    virtual T GetLoss() = 0;
};
}

#endif
