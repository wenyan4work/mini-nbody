SRC=nbody-nosoft.cu
EXE=nbody-nosoft

nvcc -arch=sm_52 -ftz=true -use_fast_math -I../ -o $EXE $SRC -DSHMOO

echo $EXE

K=1024
for i in {1..8}
do
    ./$EXE $K
    K=$(($K*2))
done

