// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_SHAREDPTRTESTS_HPP
#define TAKION_SHAREDPTRTESTS_HPP

#include <Takion/Utils/SharedPtr.hpp>

namespace Takion::Test
{
void CopyAndDestruct(const SharedPtr<int>& sharedPtr, int numCopy, bool* stop);

void ConcurrentCopy(int spawnNum, int numCopy);

}  // namespace Takion

#endif  // CUBBYDNN_SHAREDPTRTESTS_HPP
