//
// Created by jwkim98 on 7/8/19.
//

#include <cubbydnn/Engine/Engine.hpp>

namespace CubbyDNN
{
Engine::Engine(size_t sourceNum, size_t sinkNum, size_t intermediateNum)
{
    m_sourceUnitVector.reserve(sourceNum);
    m_sinkUnitVector.reserve(sinkNum);
    m_intermediateUnitVector.reserve(intermediateNum);
}

void Engine::Scan()
{
    for (auto& sourceUnit : m_sourceUnitVector)
    {
        if (sourceUnit.IsReady())
        {
            auto func = [&sourceUnit]() { return sourceUnit.Compute(); };
            EnqueueTask(TaskWrapper(TaskType::ComputeSource, func));
        }
    }

    for (auto& sinkUnit : m_sinkUnitVector)
    {
        if (sinkUnit.IsReady())
        {
            auto func = [&sinkUnit]() { return sinkUnit.Compute(); };
            EnqueueTask(TaskWrapper(TaskType::ComputeSink, func));
        }
    }

    for (auto& intermediateUnit : m_intermediateUnitVector)
    {
        if (intermediateUnit.IsReady())
        {
            auto func = [&intermediateUnit]() {
                return intermediateUnit.Compute();
            };
            EnqueueTask(TaskWrapper(TaskType::ComputeIntermediate, func));
        }
    }
}

}  // namespace CubbyDNN