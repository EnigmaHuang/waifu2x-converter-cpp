# waifu2x-converter-cpp makefile, Enigma Huang, 2016-05-02
# The original project does not have a makefile or cmake configure 
# file, I manually added this make file and it seems can work.

EXENAME = waifu2x-cpu
OPENCV_PATH = /home/enigma/opencv-3.1.0-gcc

#`pkg-config --cflags --libs opencv`
GCCFLAGS  = -O2 -std=c++11 -fopenmp -march=native -I../include -I$(OPENCV_PATH)/include -L$(OPENCV_PATH)/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_features2d 
ICCFLAGS  = -O2 -std=c++11 -openmp -xHost -I../include -I$(OPENCV_PATH)/include -L$(OPENCV_PATH)/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_features2d 

CC     = icpc
CFLAGS = $(ICCFLAGS)

CCSRCS  = $(wildcard *.cpp)
OBJS    = $(CCSRCS:.cpp=.o)

build : $(EXENAME)

all : build

$(EXENAME): $(OBJS) 
	$(CC) $(CFLAGS) -o $(EXENAME) $(OBJS)
	cp $(EXENAME) ../

%.o: %.cpp  
	$(CC) $(CFLAGS) -c $^

clean:
	$(RM) *.o *.gch $(EXENAME)
	clear
