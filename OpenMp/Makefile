build: conv_openmp.c
	gcc -o conv_openmp conv_openmp.c ../Utils/utils.c -lm -fopenmp

run: build
	time ./conv_openmp 4 ../Inputs/baby-yoda.pnm ../Outputs/onpenmp_baby-yoda.pnm 3 5

clean:
	rm conv_openmp
