//
// Created by jwkim on 18. 11. 20.
//
#include "../../../../Includes/cubbydnn/Backend/util_decl/shape.hpp"

bool shape::check_shape(const std::vector<int> &shape,
                        const std::string &op_name)
{
    // TODO: find way to check if argument was verifiable
    static bool has_been_valid = true;

    if (!has_been_valid)
        return false;

    if (shape.empty())
    {
        has_been_valid = false;
        std::cout << "Argument shape is empty" << std::endl;
    }
    else if (shape.size() != 3)
    {
        //dimension over 3 is not allowed
        has_been_valid = false;
        std::cout << "shape is not 3-dimensional" << std::endl;
    }

    for (auto elem : shape){
        //every dimension should be at least 1
        if (elem <= 0)
        {
            has_been_valid = false;
            std::cout << "Invalid shape" << std::endl;
        }
    }

    if (!has_been_valid)
    {
        std::cout << "This Error occurs from operation: " << op_name
                  << std::endl;
    }
    return has_been_valid;
}

unsigned long shape::get_shape_size(const std::vector<int> &shape) {
    unsigned long size = 1;
    for(auto element : shape){
        size *= element;
    }
    return size;
}