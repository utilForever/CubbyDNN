//
// Created by jwkim98 on 8/15/19.
//

#ifndef CUBBYDNN_ACTIVATIONFUNCTIONS_HPP
#define CUBBYDNN_ACTIVATIONFUNCTIONS_HPP

#include <cubbydnn/Units/HiddenComputableUnits/HiddenUnit.hpp>
#include <cmath>

namespace CubbyDNN
{
class Logistic : public HiddenUnit
{
    void Compute() final
    {
#ifdef CubbyMath
        auto& tensor = m_inputTensorVector.at(0);
#else
        auto& tensor = m_inputTensorVector.at(0);
        switch(tensor.Info.GetNumberSystem()){
        case NumberSystem::Float16 :


        }

#endif
    }

private:

};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_ACTIVATIONFUNCTIONS_HPP
