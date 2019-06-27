/** Copyright (c) 2019 Chris Ohk, Justin Kim
 *
* We are making my contributions/submissions to this project solely in our
* personal capacity and are not conveying any rights to any intellectual
* property of any third parties.
*/

#include <cubbydnn/Tensors/TensorData.hpp>

namespace CubbyDNN
{
    TensorData::TensorData(void* Data, NumberSystem numberSystem, const TensorDataInfo& info)
    :DataPtr(Data), numberSystem(numberSystem), Info(info)
    {

    }
}