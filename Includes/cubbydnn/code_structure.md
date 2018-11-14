
#Code Structure

Graph generation will be single-threaded
However, analyzing the graph and executing the graph can be multi-threaded

* Management class
    * This class is able to manage resources used in 
    _graph-construction process_
    
    contains: 
    * adjacency matrix for representing graph structure
    * mutex for modifying adj matrix
    * getters and setters for adj matrix
    * abilities to manage placeholders (which can stream data into graph)
    * declaration of private resources:
    
    '''
            static std::deque<std::unique_ptr<Tensor_container<T>>> placeHolders;
    
            static std::deque<std::vector<std::unique_ptr<Tensor_container<T>>>> adj_forward;
    
            Management(){} ///disable the constructor
    
            static std::mutex adj_mutex;
    '''
    