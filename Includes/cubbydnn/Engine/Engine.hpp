// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_ENGINE_HPP
#define CUBBYDNN_ENGINE_HPP

#include <cstdlib>

#include <cubbydnn/Engine/ThreadManager.hpp>
#include <cubbydnn/Utils/SharedPtr-impl.hpp>
#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{

class Engine : private ThreadManager
{
public:
    Engine(size_t sourceNum, size_t sinkNum, size_t intermediateNum);

    void Scan();

private:
    std::vector<SourceUnit> m_sourceUnitVector;
    std::vector<SinkUnit> m_sinkUnitVector;
    std::vector<IntermediateUnit> m_intermediateUnitVector;
    std::vector<CopyUnit> m_copyUnitVector;
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_ENGINE_HPP
