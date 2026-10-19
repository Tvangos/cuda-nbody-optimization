VERSION=$1
THREADS=$2
DEBUG_FLAG=$3
EXE="${VERSION}/nbody"

export OMP_NUM_THREADS=$THREADS

make -f Makefile_cuda clean DIR=$VERSION
make -f Makefile_cuda DIR=$VERSION

python3 ./generate_dataset.py

echo "Running $EXE..."

DEBUG=$DEBUG_FLAG $EXE

