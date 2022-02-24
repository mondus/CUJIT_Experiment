#include <nvrtc.h>
#include <cuda.h>
#include <iostream>

#define NUM_THREADS 128
#define NUM_BLOCKS 32
#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)

const char* saxpy = R"###(    

#define NUM_THREADS 128
#define NUM_BLOCKS 32
#define N NUM_THREADS*NUM_BLOCKS

__device__ float dX[N];
__device__ float dY[N];
                                   
extern "C" __global__ void saxpy(float a, float *out, size_t n)   
{                                                               
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = a * dX[tid] + dY[tid];
  } 
}
)###";

int main()
{
	// Create an instance of nvrtcProgram with the SAXPY code string.
	nvrtcProgram prog;
	NVRTC_SAFE_CALL(
		nvrtcCreateProgram(&prog,         // prog
			saxpy,         // buffer
			"saxpy.cu",    // name
			0,             // numHeaders
			NULL,          // headers
			NULL));        // includeNames
	
	// register the named expressions
	NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, "&dX"));
	//NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, "&dY"));
						   
	// Compile the program with fmad disabled.
	// Note: Can specify GPU target architecture explicitly with '-arch' flag.
	
	nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
		0,     // numOptions
		0); // options
	// Obtain compilation log from the program.
	size_t logSize;
	NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
	char* log = new char[logSize];
	NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
	std::cout << log << '\n';
	delete[] log;
	if (compileResult != NVRTC_SUCCESS) {
		exit(1);
	}
	// Obtain PTX from the program.
	size_t ptxSize;
	NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
	char* ptx = new char[ptxSize];
	NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
	printf("%s\n", ptx);
	// Load the generated PTX and get a handle to the SAXPY kernel.
	CUdevice cuDevice;
	CUcontext context;
	CUmodule module;
	CUfunction kernel;
	CUDA_SAFE_CALL(cuInit(0));
	CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
	CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));

	// Generate input for execution, and create output buffers.
	size_t n = NUM_THREADS * NUM_BLOCKS;
	size_t bufferSize = n * sizeof(float);
	float a = 5.1f;
	float* hX = new float[n];
	float* hX10 = new float[n];	// same as hX but x10
	float* hY = new float[n];
	float *hOut = new float[n];
	for (size_t i = 0; i < n; ++i) {
		hX[i] = static_cast<float>(i);
		hX10[i] = static_cast<float>(i*10);
		hY[i] = static_cast<float>(i * 2);
	}
	//alocated memory
	CUdeviceptr dX10, dOut;
	CUDA_SAFE_CALL(cuMemAlloc(&dOut, bufferSize));
	CUDA_SAFE_CALL(cuMemAlloc(&dX10, bufferSize));

	//options
	unsigned int SYMBOL_COUNT = 1;
	CUjit_option opts[3] = { CU_JIT_GLOBAL_SYMBOL_COUNT, CU_JIT_GLOBAL_SYMBOL_NAMES, CU_JIT_GLOBAL_SYMBOL_ADDRESSES };
	const char* SYMBOL_NAMES[1] = { "dX" };
	void* SYMBOL_ADDRESSES[1] = { (void*)dX10 };
	void* optvals[3] = { &SYMBOL_COUNT, SYMBOL_NAMES, SYMBOL_ADDRESSES };

	// create and link module
	CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 3, opts, optvals));
	CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "saxpy"));
	// device symbols
	CUdeviceptr dX, dY;
	// get mangled device name
	const char* dX_mangled_name;
	const char* dY_mangled_name;
	//NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, "&dX", &dX_mangled_name));
	NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, "&dY", &dY_mangled_name));
	// get device pointers
	//CUDA_SAFE_CALL(cuModuleGetGlobal(&dX, NULL, module, dX_mangled_name));
	CUDA_SAFE_CALL(cuModuleGetGlobal(&dY, NULL, module, dY_mangled_name));

	//CUDA_SAFE_CALL(cuMemcpyHtoD(dX, hX, bufferSize));
	CUDA_SAFE_CALL(cuMemcpyHtoD(dX10, hX10, bufferSize));
	CUDA_SAFE_CALL(cuMemcpyHtoD(dY, hY, bufferSize));
	// Destroy the program.
	NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
	// Execute SAXPY.
	void* args[] = { &a, &dOut, &n };
	CUDA_SAFE_CALL(
		cuLaunchKernel(kernel,
			NUM_BLOCKS, 1, 1,    // grid dim
			NUM_THREADS, 1, 1,   // block dim
			0, NULL,             // shared mem and stream
			args, 0));           // arguments
	CUDA_SAFE_CALL(cuCtxSynchronize());
	// Retrieve and print output.
	CUDA_SAFE_CALL(cuMemcpyDtoH(hOut, dOut, bufferSize));
	for (size_t i = 0; i < n; ++i) {
		std::cout << a << " * " << hX[i] << " + " << hY[i]
			<< " = " << hOut[i] << '\n';
	}
	// Release resources.
	CUDA_SAFE_CALL(cuMemFree(dOut));
	CUDA_SAFE_CALL(cuModuleUnload(module));
	CUDA_SAFE_CALL(cuCtxDestroy(context));
	delete[] hX;
	delete[] hY;
	delete[] hOut;
	return 0;
}