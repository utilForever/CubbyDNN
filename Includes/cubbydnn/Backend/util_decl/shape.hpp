//
// Created by jwkim on 18. 11. 20.
//

#ifndef CUBBYDNN_SHAPE_HPP
#define CUBBYDNN_SHAPE_HPP

#include <iostream>
#include <string>
#include <vector>

struct shape
{
 public:
    static bool check_shape(const std::vector<int> &shape,
                            const std::string &op_name);

    static unsigned long get_shape_size(const std::vector<int> &shape);

 private:
    std::vector<int> shape;
};

#endif  // CUBBYDNN_SHAPE_HPP
