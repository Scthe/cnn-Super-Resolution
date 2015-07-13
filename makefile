# $@ - left side of ':'
# $^ - right side of ':'
# $< - first of dependencies

CC = clang++
VPATH = src/opencl src test test/specs libs/cpp
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

__OBJ = Config.o \
	LayerData.o \
	DataPipeline.o \
	ConfigBasedDataPipeline.o \
	Utils.o \
	Context.o \
	UtilsOpenCL.o \
	Kernel.o \
	gason.o

_OBJ = Main_cl.o $(__OBJ)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ)) # append ODIR to each entry

# _TEST_OBJ = TestRunner.o $(__OBJ) TestDataProvider.o LayerDeltasTest.o BackpropagationTest.o
_TEST_OBJ = TestRunner.o $(__OBJ) \
	ExtractLumaTest.o \
	MeanSquaredErrorTest.o \
	SubtractFromAllTest.o \
	SumTest.o \
	LayerDeltasTest.o \
	BackpropagationTest.o \
	LayerTest.o \
	LastLayerDeltaTest.o \
	WeightDecayTest.o \
	UpdateParametersTest.o \
	ConfigTest.o
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
