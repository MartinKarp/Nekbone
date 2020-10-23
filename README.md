Nekbone
=======

Nekbone solves a standard Poisson equation using a conjugate gradient iteration with a simple or spectral element multigrid preconditioner
on a block or linear geometry. It exposes the principal computational kernel to reveal the essential elements of the algorithmic-
architectural coupling that is pertinent to Nek5000.


This implementation depends heavily on Intel's FPGA SDK for OpenCL. The compilation of the device code requires that your paths have been setup accordingly and for setting up you renvironment and genereal troubleshotting to get aoc to run I refer you to "Intel FPGA SDK for OpenCL Pro Edition: Programming Guide" and "Intel FPGA SDK for OpenCL Pro Edition: Getting Started Guide".

The device code can be found under src/device and in general we compile it with the following command

aoc -v -report -ffp-reassoc -ffp-contract=fast <PATH_TO_CL_FILE> 

or 

aoc -v -report -ffp-reassoc -ffp-contract=fast -no-interleaving=default <PATH_TO_CL_FILE> 

to compile without interleaving the main memory. The code right now is assuming 4 memory banks of DDR memory.

For emulation one can use

aoc -v -report -legacy-emulator -march=emulator <PATH_TO_CL_FILE>

As for execution, in the directory test/nek_fpga the host code i compiled with ./makenek. The program can then be executed with mpirun -np 1 ./nekbone <PATH_TO_BINARY>. If you want to emulate the code please use CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 mpirun -np 1 ./nekbone <PATH_TO_BINARY>

To change the problem size one can modify the polynomial degree and max number of elements in tests/nek_fpga/SIZE and what tests to run in tests/nek_fpga/data.rea. Please observe that rather than specifyin the number of elements in data.rea, but rather you specify ielN and compute for 2^ielN elements.

The main changes to the standard nekbone are located in src/cg.f and to some extent src/driver.f

We utilize the CLFORTRAN interface from CASS which can be located in its orignal form at https://github.com/cass-support/clfortran.

