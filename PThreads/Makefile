build: conv_threads.c
	gcc -o conv_threads conv_threads.c ../Utils/utils.c -lm -lpthread

run: build
	time ./conv_threads 4 ../Inputs/baby-yoda.pnm ../Outputs/pthreads_baby-yoda.pnm 10 2

clean:
	rm conv_threads
