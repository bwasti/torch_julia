PYTORCH_DIR=$(python -c 'import os, torch; print(os.path.dirname(os.path.realpath(torch.__file__)))')
JULIA_DIR=$(julia -e 'print(joinpath(Sys.BINDIR, Base.DATAROOTDIR, "julia"))')
mkdir -p build && cd build
cmake .. -DPYTORCH_DIR=${PYTORCH_DIR} -DJULIA_DIR=${JULIA_DIR} -DCMAKE_BUILD_TYPE=Release
VERBOSE=1 make -j 24
