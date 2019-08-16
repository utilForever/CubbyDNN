// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_GRAPHSYNCHRONIZATIONTEST_HPP
#define CUBBYDNN_GRAPHSYNCHRONIZATIONTEST_HPP

#include <cubbydnn/Engine/Engine.hpp>
#include <cubbydnn/Units/CopyUnit.hpp>
#include <cubbydnn/Units/SourceComputableUnits/SourceUnit.hpp>
#include <cubbydnn/Units/SinkComputableUnits/SinkUnit.hpp>
#include <cubbydnn/Units/HiddenComputableUnits/HiddenUnit.hpp>
#include "gtest/gtest.h"


#include <vector>

namespace GraphTest{

    void GraphConstruction();

}

#endif //CUBBYDNN_GRAPHSYNCHRONIZATIONTEST_HPP