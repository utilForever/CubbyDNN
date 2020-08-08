<h4>How to use</h4>
_CubbyDNN_ computes operations using directed graph structure. Each vertex corresponds to an computation and each edge corresponds to copy operation. 

We call vertices _main unit_ and edges _copy unit_. 

Main units can perform any kind of computations. Such as reading and saving data or any mathematical calculation that's necessary.

On the other hand, Copy units can only _copy_ results from previous main unit to next corresponding main unit.

Therefore, Each main unit passes data structure called _tensor_ to next unit via copy units. 

Copy units and main units can communicate together so they can run asynchronously without violating dependencies.



__Main Units__

Main units play role of vertex in computation graph.

There are 3 types of main units



* __Source Unit__

  This unit can only have outputs. Therefore, they must generate some kind of tensor to pass every epoch. They are always located at the begging of the graph. Each graph should have at least one source unit. Source unit can only have one output.

  ex) Image loader, Data loaders



* __Hidden Unit__

  Hidden unit has both input and outputs. Therefore, they produce outputs using given inputs. They are always located between source unit and hidden unit. Mathematical computations usually locate in hidden units. Hidden unit can have multiple inputs and only one output.

  ex) Matmul, Conv2D, Relu



* __Sink Unit__

  Sink unit has only have inputs. They only receive tensors and make no outputs. Data visualizer or savers are usually this kind of unit. Sink unit can have multiple inputs.

  ex) Graph plotter, Data saver

  

__Copy Units__

Copy units play role of delivering output tensor from previous main unit to next main unit

Each input-output pair between different main units has copy units. 

If main units are being executed between different hardwares, copy unit can copy data between different devices.



__How to construct graph__

1. Add units to graph

   First of all, we need to tell which kind of units we require.

   We can add each unit using these methods (Declared in Engine.hpp)

   _inputTensorInfoVector_ and _outputTensorInfoVector_ are vector of _TensorInfo_ Which describes shape and characteristics of each input and output tensor.

   ```C++
   //! Adds sourceUnit to sourceUnitVector and assigns ID for the unit
   //! \param outputTensorInfoVector:  vector of TensorInfo of outputs
   //! \return : assigned id of the unit
   static size_t AddSourceUnit(std::vector<TensorInfo> outputTensorInfoVector);
   
   //! Adds intermediateUnit to intermediateUnitVector and assigns ID for the
   //! unit
   //! \param inputTensorInfoVector : vector of TensorInfo of inputs
   //! \param outputTensorInfoVector : vector of TensorInfo of outputs
   //! \return : assigned id of the unit
   static size_t AddHiddenUnit(std::vector<TensorInfo> inputTensorInfoVector,
                               std::vector<TensorInfo> outputTensorInfoVector);
   
   //! Adds sinkUnit to intermediateUnitVector and assigns ID for the unit
   //! \param inputTensorInfoVector : vector of TensorInfo of inputs
   //! \return : assigned id of the unit
   static size_t AddSinkUnit(std::vector<TensorInfo> inputTesorInfoVector);
   ```

   Each methods will generate units using given tensor shape. They return unique id of each unit. This id is used to connect units together.

   

2. Connect units

   Using ID returned from Add methods we now connect them together. 

   Depending on types of units we want to connect, 3 types of methods are prepared.

   _destInputIndex_ parameter indicates which index of next unit that output of origin will be connected to. _destInputIndex_ must be an integer between 0 and number of input tensors in next unit.

   ```c++
   //! Connects between sourceUnit and intermediateUnit by assigning copyUnit
   //! between them
   //! \param originID : sourceUnit ID to connect
   //! \param destID : intermediateUnit ID of destination
   //! \param destInputIndex : input index of this connection to destination
   static void ConnectSourceToIntermediate(size_t originID, size_t destID,
                                           size_t destInputIndex = 0);
   
   //! Connects between intermediateUnit and intermediateUnit by assigning
   //! copyUnit between them
   //! \param originID : unique ID of origin intermediateUnit
   //! \param destID : unique ID of destination intermediateUnit
   //! \param destInputIndex : input index of this connection to destination
   static void ConnectIntermediateToIntermediate(size_t originID,
                                                 size_t destID,
                                                 size_t destInputIndex = 0);
   
   //! Connects between intermediateUnit and sinkUnit by assigning
   //! copyUnit between them
   //! \param originID : unique ID of origin intermediateUnit
   //! \param destID : unique ID of destination sinkUnit
   //! \param destInputIndex : input index of this connection to destination
   static void ConnectIntermediateToSink(size_t originID, size_t destID,
                                         size_t destInputIndex = 0);
   ```

   

3. Execute

   These methods are used to execute graphs

   Start Execution spawns threads for executing main and copy units

   After calling StartExecution(), JoinThreads() must be called to join spawned threads

   ```c++
   //! Initializes thread pool
   //! \param mainThreadNum : number of threads to assign to main operation
   //! \param copyThreadNum : number of threads to assign to copy operation
   //! \param epochs : number of epochs to execute the graph
   //! concurrency)
   static void StartExecution(size_t mainThreadNum, size_t copyThreadNum,
                              size_t epochs);
   
   //! Joins threads in the thread queue and terminates the execution
   static void JoinThreads();
   ```





__Example__

Here is the example of sample code

![Model](Medias/model.png)

Here is the code snippet for generating and executing this model

```c++
const std::vector<TensorInfo> sourceOutputTensorInfoVector = {
    TensorInfo({ 1, 1, 1 }), TensorInfo({ 1, 2, 1 })
};

const std::vector<TensorInfo> inputTensorInfoVector1 = { TensorInfo(
    { 1, 1, 1 }) };
const std::vector<TensorInfo> outputTensorInfoVector1 = { TensorInfo(
    { 3, 3, 3 }) };

const std::vector<TensorInfo> inputTensorInfoVector2 = { TensorInfo(
    { 3, 3, 3 }) };
const std::vector<TensorInfo> outputTensorInfoVector2 = { TensorInfo(
    { 6, 6, 6 }) };

const std::vector<TensorInfo> inputTensorInfoVector3 = { TensorInfo(
    { 1, 2, 1 }) };
const std::vector<TensorInfo> outputTensorInfoVector3 = { TensorInfo(
    { 3, 3, 3 }) };

const std::vector<TensorInfo> inputTensorInfoVector4 = { TensorInfo(
    { 3, 3, 3 }) };
const std::vector<TensorInfo> outputTensorInfoVector4 = { TensorInfo(
    { 6, 6, 6 }) };

const std::vector<TensorInfo> sinkInputTensorInfoVector = {
    TensorInfo({ 6, 6, 6 }), TensorInfo({ 6, 6, 6 })
};

const auto sourceID = Engine::AddSourceUnit(sourceOutputTensorInfoVector);

const auto intermediate1ID =
    Engine::AddHiddenUnit(inputTensorInfoVector1, inputTensorInfoVector2);
const auto intermediate2ID =
    Engine::AddHiddenUnit(inputTensorInfoVector2, outputTensorInfoVector2);
const auto intermediate3ID =
    Engine::AddHiddenUnit(inputTensorInfoVector3, outputTensorInfoVector3);
const auto intermediate4ID =
    Engine::AddHiddenUnit(inputTensorInfoVector4, outputTensorInfoVector4);
const auto sinkID = Engine::AddSinkUnit(sinkInputTensorInfoVector);

Engine::ConnectSourceToIntermediate(sourceID, intermediate1ID);
Engine::ConnectSourceToIntermediate(sourceID, intermediate3ID);
Engine::ConnectIntermediateToIntermediate(intermediate1ID, intermediate2ID);
Engine::ConnectIntermediateToIntermediate(intermediate3ID, intermediate4ID);
Engine::ConnectIntermediateToSink(intermediate2ID, sinkID, 0);
Engine::ConnectIntermediateToSink(intermediate4ID, sinkID, 1);

Engine::StartExecution(3, 3, 100);
Engine::JoinThreads();
```

