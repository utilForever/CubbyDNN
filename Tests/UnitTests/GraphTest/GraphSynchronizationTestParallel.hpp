// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_GRAPHSYNCHRONIZATIONTESTPARALLEL_HPP
#define CUBBYDNN_GRAPHSYNCHRONIZATIONTESTPARALLEL_HPP

namespace CubbyDNN
{
void SimpleGraphTestParallel(int numMainThreads, int numCopyThreads,
                             size_t epochs);

void MultiplyGraphTestParallel(int numMainThreads, int numCopyThreads,
                               size_t epochs);
}

#endif