// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SHAREDPTRTESTS_HPP
#define CUBBYDNN_SHAREDPTRTESTS_HPP

#include <cubbydnn/Utils/SharedPtr-impl.hpp>

namespace CubbyDNN
{
    void CopyAndDestruct(SharedPtr<int>& sharedPtr);

    void ConcurrentCopy(int spawnNum);

}

#endif //CUBBYDNN_SHAREDPTRTESTS_HPP
