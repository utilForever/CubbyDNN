// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#include "WeakPtrTests.hpp"
#include <doctest/doctest.h>
#include <Takion/Utils/WeakPtr.hpp>
#include <Takion/Utils/SharedPtr.hpp>

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
