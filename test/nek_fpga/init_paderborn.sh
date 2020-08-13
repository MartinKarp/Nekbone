#!/bin/bash
export QUARTUS_ROOTDIR=/home/martin/fpga/intelFPGA_pro/20.1/quartus
export INTELFPGAOCLSDKROOT=/home/martin/fpga/intelFPGA_pro/20.1/hld
export AOCL_BOARD_PACKAGE_ROOT="/home/martin/fpga/Paderborn/s10"
export PATH="$PATH:$QUARTUS_ROOTDIR/bin:$INTELFPGAOCLSDKROOT/bin"
export LD_LIBRARY_PATH="$AOCL_BOARD_PACKAGE_ROOT/linux64/lib:"\
"$INTELFPGAOCLSDKROOT/host/linux64/lib:$AOCL_BOARD_PACKAGE_ROOT/tests/extlibs/lib":$LD_LIBRARY_PATH
#export LM_LICENSE_FILE=31411@license-altera-1.pdc.kth.se
export LM_LICENSE_FILE=27009@lic03.ug.kth.se

echo "----------------------------------------------------------------------------------------------------------"
echo;
echo "Cheat Sheet (Compilation)"
echo "To compile for emulator: "
echo "                aoc -legacy-emulator -march=emulator <file.cl>"
echo;
echo "To compile for FPGA:"
echo "                aoc -board=p520_max_sg280l -report <file.cl>"
echo;
echo "----------------------------------------------------------------------------------------------------------"
echo;
echo "Cheat Sheet (Execution)"
echo "To run with emulator: CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 <binary>"
echo;
echo "----------------------------------------------------------------------------------------------------------"
echo;
echo "Valuable FLAGS for optimizations:"
echo "               -ffp-reassoc       ->  Allow the compiler to relax the order of arithmetic operations."
echo "               -ffp-contract=fast ->  Removes intermediary roundings/conversions where possible and change rounding (accuracy changes!)"
echo "               -fast-compile      ->  Fast compilation times but reduced fmax (use for prototyping)"
echo "               -high-effort       ->  For large designs."
echo "               -report            ->  Generated report (html file)"
echo;
echo "----------------------------------------------------------------------------------------------------------"
echo;
echo "Resources:"
echo "Intel FPGA SDK for OpenCL Pro Version: "
echo "               https://www.intel.com/content/dam/www/programmable/us/en/pdfs/literature/hb/opencl-sdk/aocl_getting_started.pdf"
echo;
echo "Intel FPGA SDK for OpenCL Best Practices: "
echo "               https://www.intel.com/content/dam/www/programmable/us/en/pdfs/literature/hb/opencl-sdk/aocl-best-practices-guide.pdf"
echo;
echo "Intel FPGA SDK for OpenCL Programming guide: "
echo "               https://www.intel.com/content/www/us/en/programmable/documentation/mwh1391807965224.html"
echo;
echo "For Noctua setup: "
echo "               https://wikis.uni-paderborn.de/pc2doc/Noctua_FPGA_Usage,_Integration_and_Development"
echo;
echo "----------------------------------------------------------------------------------------------------------"
echo;
echo "A Large number of examples can be found under: /opt/intel/intelFPGA_pro/20.1/hld/examples_aoc/"
echo;
echo "Note many of these are highly optimizes and can be hard to read. Always start simply and then refactor code as you go along."
echo;
echo "----------------------------------------------------------------------------------------------------------"
