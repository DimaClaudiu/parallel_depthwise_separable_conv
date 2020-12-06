build: ImageProcessing.c
	mpicc -o imageProcessing ImageProcessing.c ../Utils/utils.c -lm

run: build
	mpirun -np 4 imageProcessing ../Inputs/baby-yoda.pnm ../Outputs/mpi_baby-yoda.pnm 2 2

clean:
	rm imageProcessing
