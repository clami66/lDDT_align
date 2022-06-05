all: InterfaceComparison

InterfaceComparison: InterfaceComparison.cpp InterfaceComparison.h

	g++ -static -O3 -ffast-math -lm -o InterfaceComparison InterfaceComparison.cpp 
clean:
	rm -f InterfaceComparison

test:
	sh ./test.sh
