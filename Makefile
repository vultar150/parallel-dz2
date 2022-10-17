all: compile

compile: main.c
	mpicc -std=c99 main.c -lm

clean:
	rm -rf  output err a.out
