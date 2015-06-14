# $@ - left side of ':'
# $^ - right side of ':'
# $< - first of dependencies

CC = clang++
VPATH = src/opencl src test libs/cpp
IDIR = libs/include
ODIR = obj
BINDIR = bin
LIBS = -lm -L libs/lib -l OpenCL
EXECNAME = cnn.exe

CFLAGS = -std=c++11 \
	-c \
	-g \
	-Wall \
	-Wextra \
	-stdlib=libstdc++ \
	-isystem "C:\programs\install\MinGW\include" \
	-isystem "C:\programs\install\MinGW\lib\gcc\mingw32\4.7.2\include\c++" \
	-isystem "C:\programs\install\MinGW\lib\gcc\mingw32\4.7.2\include\c++\mingw32" \
	-I$(IDIR)

LFLAGS = -std=c++11 \
	-l "stdc++" \
	-I$(IDIR)

# _OBJ = Main_cl.o Config.o Layers.o Context.o UtilsOpenCL.o Kernel.o gason.o
_OBJ = Main_cl.o Config.o Context.o UtilsOpenCL.o Kernel.o gason.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ)) # append ODIR to each entry

# _TEST_OBJ = TestRunner.o Config.o Layers.o Context.o UtilsOpenCL.o Kernel.o gason.o TestDataProvider.o
_TEST_OBJ = TestRunner.o Config.o Context.o UtilsOpenCL.o Kernel.o gason.o TestDataProvider.o
TEST_OBJ = $(patsubst %,$(ODIR)/%,$(_TEST_OBJ))



# default target
build: $(EXECNAME)

compile: $(OBJ)

run: $(EXECNAME)
	@echo -----------------------
	@$(BINDIR)/$<

test: $(TEST_OBJ)
	@echo Linking tests..
	g++ -o $(BINDIR)/test.exe $^ $(LFLAGS) $(LIBS)
	@echo -----------------------
	@$(BINDIR)/test.exe


clean:
	rm -f $(ODIR)/*.o
	rm -f $(BINDIR)/*



$(EXECNAME): $(OBJ)
	@echo Linking..
	g++ -o $(BINDIR)/$@ $^ $(LFLAGS) $(LIBS)

$(ODIR)/%.o: %.cpp
	$(CC) -c -o $@ $< $(CFLAGS)
