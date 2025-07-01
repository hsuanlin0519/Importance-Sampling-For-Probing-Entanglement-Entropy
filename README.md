# Measuring entanglement entropy with importance sampling

This is a program probing entanglement entropy using importance sampling to enhance
the power of randomized measurement on purity calculation. It's a combination of machine learning,
randomized measurements and sampling algorithms, which provides a different approach to estimate the purity of 
a specific quantum state.[1]
## Getting started with the tool

In the Directory "example", you only need to run main.py.


## File Hierarchy
In directory "/example"

    1. main.py => main program 
    2. classdef.py => define class objects for neural net, datasets, quantum circuits, result managing
    3. func.py => define functions for randomized measurements, purity estimation ...etc
    4. datavisual.py => define data visualization, figure output functions
    5. decomp.py => define functions of decomposing and reconstructing random unitaries
    6. trot_state.py => define functions to construct quench dynamics singlet bell state

In directory "/example/data"

ML datasets are stored as two json files for each respective state showed as below

    1 . purityCellList => stores all the X(u) values
    2 . unitaryOP => stores all the unitary used in randomized measurements (matrix format)


## Some explanation for the program

1 . ML part

      label => X(u) function value X
      angle => parametrized angles of a local unitary transformation u
      answers => prediction of neural net (an approximation of X)

##There are still works to do... will fill out this part later

## Contributors & Institution
國立政治大學應用物理所 林敬軒

國立政治大學應用物理所 助理教授 許琇娟

![plot](./RMtoolbox/PhysLogo.png)
### Contact me for any bug reports and code explanations:
hsuanlin0519@gmail.com  or  110755003@nccu.edu.tw

## References

    [1] Aniket Rath, Rick van Bijnen, Andreas Elben, Peter Zoller, and Benoît Vermersch.
    Importance sampling of randomized measurements for probing entanglement. Phys.Rev. Lett., 127:200503, Nov 2021
    
    [2] https://pytorch.org/tutorials/
