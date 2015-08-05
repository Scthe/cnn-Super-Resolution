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
	pch.o \
	Context.o \
	UtilsOpenCL.o \
	Kernel.o \
	gason.o

_OBJ = Main_cl.o $(__OBJ)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ)) # append ODIR to each entry

# _TEST_OBJ = TestRunner.o $(__OBJ) TestDataProvider.o LayerDeltasTest.o BackpropagationTest.o
_TEST_OBJ = TestRunner.o $(__OBJ) \
	TestCase.o \
	ExtractLumaTest.o \
	SwapLumaTest.o \
	SquaredErrorTest.o \
	SubtractFromAllTest.o \
	SumTest.o \
	LayerDeltasTest.o \
	BackpropagationTest.o \
	LayerTest.o \
	LastLayerDeltaTest.o \
	UpdateParametersTest.o \
	ConfigTest.o
TEST_OBJ = $(patsubst %,$(ODIR)/%,$(_TEST_OBJ))


# If the first argument is "run"...
ifeq (run,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif


# default target
build: $(EXECNAME)

compile: $(OBJ)

# if You pass arguments do it like this:
# 'make run -- ARGS_HERE'
run: $(EXECNAME)
	@echo -----------------------
	@$(BINDIR)/$< $(RUN_ARGS)

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
