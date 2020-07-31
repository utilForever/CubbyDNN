// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "WeakPtrTests.hpp"
#include <doctest/doctest.h>
#include <cubbydnn/Utils/WeakPtr.hpp>

namespace Takion::Test
{
void SimpleOwnershipTransfer()
{
    const auto sharedPtr = SharedPtr<int>::Make(1);
    const WeakPtr<int> weakPtr = sharedPtr;
    const auto ptr2 = weakPtr.Lock();
    *ptr2.operator->() += 1;

    CHECK(*sharedPtr.operator->() == 2);
}

}
