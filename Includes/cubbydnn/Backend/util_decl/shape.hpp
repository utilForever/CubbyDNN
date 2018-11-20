//
// Created by jwkim on 18. 11. 20.
//

#ifndef CUBBYDNN_SHAPE_HPP
#define CUBBYDNN_SHAPE_HPP

#include <vector>
#include <string>
#include <iostream>

struct shape{
    static bool check_shape(const std::vector<int> &shape,
                            const std::string &op_name);

    static unsigned long get_shape_size(const std::vector<int> &shape);

    static unsigned long max_dim ;
};

unsigned long shape::max_dim = 3;



#endif //CUBBYDNN_SHAPE_HPP
