// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SHAREDPTRTESTS_HPP
#define CUBBYDNN_SHAREDPTRTESTS_HPP

#include <cubbydnn/Utils/SharedPtr.hpp>

namespace Takion::Test
{
void CopyAndDestruct(const SharedPtr<int>& sharedPtr, int numCopy, bool* stop);

void ConcurrentCopy(int spawnNum, int numCopy);

}  // namespace Takion

#endif  // CUBBYDNN_SHAREDPTRTESTS_HPP
