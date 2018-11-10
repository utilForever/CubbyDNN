//
// Created by jwkim on 18. 11. 10.
//

#include "../../../../Includes/cubbydnn/Backend/storage/management.h"

namespace cubby_dnn{
    template<typename T>
    int Management<T>::add_op(){
        unsigned long graph_size = adj.size();
        std::vector<std::unique_ptr<Tensor<T>>> temp(graph_size + 1, nullptr);
        adj.emplace_back(temp); // graph_size += 1
        return static_cast<int>(graph_size);
    }

    template<typename T>
    void Management<T>::add_edge(const int from, const int to, Tensor<T> &tensor) noexcept{

        try{
            int graph_size = static_cast<int>(adj.size());
            if(from == to){
                std::string error_msg = "cannot connect to operation itself";
                throw ArgumentException(error_msg);
            }
            if(graph_size + 1 < from or graph_size+ 1 < to){
                std::string error_msg = "pointing to operation that doesn't exist";
                error_msg += ("graph size: " + std::to_string(adj.size()) + "from: "
                              + std::to_string(from) + "to: " + std::to_string(to));
                throw ArgumentException(error_msg);
            }
        }
        catch(TensorException e){
            return; ///do nothing, and return
        }
        adj[from][to] = make_unique(tensor);
    }

    ///initialization of adjacency matrix
    template<typename T>
    std::deque<std::vector<std::unique_ptr<Tensor<T>>>> Management<T>::adj =
            std::deque<std::vector<std::unique_ptr<Tensor<T>>>>();
}