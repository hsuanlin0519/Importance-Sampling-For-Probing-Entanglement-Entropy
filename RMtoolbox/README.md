# RMtoobox - A toolbox for purity calculation using randomized measurements

## Getting started
This toolbox provides a simple approach to measure quantum state purity, which is built based on the
randomized measurement protocol. With an object-oriented programming style, it should have good flexibility
to fit your need, the target is to calculate the purity of a specific quantum state and store the random unitaries
and X(u) values [1]
## Repository Hierarchy
The only code you will need to run is the RMtool.py file (main function).

    1. RMtool.py => main program
    2. RMclassdef.py => define class objects for quantum circuits and result managing
    3. RMfunctions.py => define function X(u) and randomized measurements
    4. RMdatavisual.py => define data visualization, figure output functions
    5. RMJsonParse.py => define functions to parse json string ouput files of RMtool.py  
    6. decomp.py => define functions of decomposing and reconstructing random unitaries (won't be used for now)
## Program Outputs
After running RMtool.py, program will automatically create a new directory ./resultsRM_(index),
there will be four output files included :
    
    1. unitary.json => stores the unitaries of randomized measurement
    2. xResult.json => stores the results of X(u) calculation
    3. expLog.txt => stores the purity and some setup parameters
    4. circuit.png => figure of quantum circuit
## Program setups
There are two parts you shall manually set up
1. In RMtool.py lines 20-23
```python
# Set experiment parameters, you can always manually change these to fit your demand
n_qbits = 8  # total size of your quantum system
n_cbits = 4  # classical (measurement) bits of your quantum system
n_shots = 1000 # shots of circuit measurements
n_circuits = 1000 # amount of circuits you want to build for purity calculation
```
This toolbox provides both bipartite and full-system measurements, for the system to perform full
system measurements, please make sure n_qbits == n_cbits. On the other hand for bipartite, n_qbits/2 == n_cbits 
2. In RMtool.py line 33
```python
Cir_q.run_product() # specify your quantum state for this experiment
```
There are several quantum states you can call such as run_product(), run_ghz() ... etc.
You can check on those quantum states in RMclassdef.py , on the other hand you can always use
your own quantum state, just make sure it is a _qiskit.circuit.quantumcircuit.QuantumCircuit_ object.

### After setting up these two parts of RMtool.py, you are ready to go !
## Output file Json string parsing
RMJsonParse provides two functions to parse json strings of xResults.json and unitary.json.
you can use `data_preprocess()` function to easily store data into np.array type.

For xResults, the format will be simple 1D array

**_[ X<sub>0</sub>, X<sub>1</sub>, X<sub>2</sub>, ..., X<sub>n</sub>]_**

For unitary, the format will be a 2D array with every parametrized local unitary transformation  forming one 1D array
containing each of θ and φ, N is the number of unitaries you use when measuring.

_**[ [ θ<sub>0</sub>, θ<sub>1</sub>, ..., θ<sub>N</sub>, φ<sub>0</sub>, φ<sub>1</sub>, ..., φ<sub>N</sub> ], [ θ<sub>0</sub>, θ<sub>1</sub>, ..., θ<sub>N</sub>, φ<sub>0</sub>, φ<sub>1</sub>, ..., φ<sub>N</sub> ] ...]**_

## Contributors & Institution
國立政治大學應用物理所 林敬軒

國立政治大學應用物理所 副教授 許琇娟

### Contact me for any bug reports and code explanations:
hsuanlin0519@gmail.com

## References

    [1] Tiff Brydges, Andreas Elben, Petar Jurcevic, Benoî t Vermersch, Christine Maier,Ben P. Lanyon, Peter Zoller, Rainer Blatt, and Christian F. Roos.
    Probing rényi entanglement entropy via randomized measurements. Science, 364(6437):260–263