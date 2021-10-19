.SILENT:

LIBRARIES = -lGL -lglut -lGLU -L /usr/lib/ -lpardiso600-GNU800-X86-64 -llapack -lgfortran -fopenmp -lpthread -lzmq -lm -Ofast -lrt
LIB_PATH = /usr/local/include/eigen3/
STD = -std=c++11

default: clean simDER

simDER: main.cpp world.cpp elasticRod.cpp elasticStretchingForce.cpp elasticBendingForce.cpp elasticTwistingForce.cpp externalGravityForce.cpp dampingForce.cpp inertialForce.cpp timeStepper.cpp setInput.cpp collision.cpp
	g++ -O3 -Wno-deprecated-declarations $(STD) -fopenmp -I $(LIB_PATH) main.cpp world.cpp elasticRod.cpp elasticStretchingForce.cpp elasticBendingForce.cpp elasticTwistingForce.cpp externalGravityForce.cpp inertialForce.cpp dampingForce.cpp timeStepper.cpp setInput.cpp collision.cpp $(LIBRARIES) -o simDER

clean:
	rm -rf simDER

run:
	OMP_NUM_THREADS=1 PARDISOLICMESSAGE=1 NUMBA_DISABLE_PERFORMANCE_WARNINGS=1 ./simDER option.txt

