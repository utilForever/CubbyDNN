//
// Created by jwkim98 on 4/20/19.
//

#ifndef CUBBYDNN_UTIL_IMPL_HPP
#define CUBBYDNN_UTIL_IMPL_HPP

#include <cubbydnn/Tensors/Decl/Util.hpp>

namespace CubbyDNN{
    template<typename T>
    void ConnectOperations(Operation<T> from, Operation<T> to, const TensorShape& shape)
    {
        TensorSocketPtr<T> socketPtr = std::make_unique<TensorSocket<T>>();
        TensorPlugPtr<T> objectPtr = std::make_unique<TensorPlug<T>>(shape, socketPtr);

        from.AddOutput(objectPtr);
        //to.AddInput(socketPtr);
    }


}
#endif //CUBBYDNN_UTIL_IMPL_HPP
