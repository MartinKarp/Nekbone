#!/bin/bash
export QUARTUS_ROOTDIR=/home/martinkarp/intelFPGA/18.0/quartus
export INTELFPGAOCLSDKROOT=/home/martinkarp/intelFPGA/18.0/hld
export AOCL_BOARD_PACKAGE_ROOT="/home/martinkarp/intelFPGA/BSPs/de5net"
export PATH="$PATH:$QUARTUS_ROOTDIR/bin:$INTELFPGAOCLSDKROOT/bin"
export LD_LIBRARY_PATH="$AOCL_BOARD_PACKAGE_ROOT/linux64/lib:"\
"$INTELFPGAOCLSDKROOT/host/linux64/lib:$AOCL_BOARD_PACKAGE_ROOT/tests/extlibs/lib"
#export LM_LICENSE_FILE=31411@license-altera-1.pdc.kth.se
export LM_LICENSE_FILE=27009@lic03.ug.kth.se
