SFTPATH=/Users/wyan/local

CXX= g++-6
LINK = $(CXX)

# for gcc
CXXFLAGS= -O3 -fopenmp -march=native -mavx2
LINKFLAGS=  -fopenmp

# set of libraries to correctly link to the target
USERLIB =  
USERINCLUDE = 

INCLUDE_DIRS = $(USERINCLUDE)
LIBRARY_DIRS =  
LIBRARIES = $(USERLIB)

# System-specific settings
SHELL = /bin/bash
SYSLIB =	
SIZE =	size

# Files

SRC =   main.cpp
INC =   
         
# Definitions

EXE =   N2Sum.X
OBJ =   $(SRC:.cpp=.o)

# Link rule
$(EXE):	$(OBJ)
	$(LINK) $(OBJ)  -o $(EXE) $(LINKFLAGS) $(SYSLIB) $(LIBRARY_DIRS) $(LIBRARIES)
	$(SIZE) $(EXE)

# Compilation rules
.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $*.cpp -o $*.o

# Individual dependencies
$(OBJ): $(INC)

all: $(EXE)

clean: 
	rm ./*.o
	rm ./$(EXE)
