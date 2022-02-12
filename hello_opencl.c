#define CL_TARGET_OPENCL_VERSION 110

#include <CL/cl.h>
#include <stdio.h>

const char * get_error_string(cl_int err){
  switch(err){
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11: return "CL_BUILD_PROGRAM_FAILURE";
  case -12: return "CL_MAP_FAILURE";

  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  default: return "Unknown OpenCL error";
  }
}



#define NWITEMS 512
// A simple memset kernel
const char *source =
"kernel void memset(   global uint *dst )             \n"
"{                                                    \n"
"    dst[get_global_id(0)] = get_global_id(0);        \n"
"}                                                    \n";

/* "kernel void memset(  global uint *dst )       \n" */
/* "{                                             \n" */
/* "    dst[get_global_id(0)] = get_global_id(0); \n" */
/* "}                                             \n";*/

int
main(int argc,  char **agrv)
{
  // 1. Get a platform
  cl_platform_id platform;
  cl_int ret;
  ret = clGetPlatformIDs(1, &platform, NULL);
  if(ret!=0) {
    puts("clGetPlatformIDs");
    return ret;
  }

  // 2. Find a gpu device.
  cl_device_id device;
  ret = clGetDeviceIDs(
                 platform,
                 CL_DEVICE_TYPE_GPU,
                 1,
                 &device,
                 NULL);
  if(ret!=0) {
    puts("clGetDeviceIDs");
    return ret;
  }

  // 3. Create a context and command queue on that device.
  cl_context context = clCreateContext(
                                       NULL,
                                       1,
                                       &device,
                                       NULL,
                                       NULL,
                                       NULL);

  cl_command_queue queue = clCreateCommandQueue(
                                                context,
                                                device,
                                                0,
                                                NULL);

  // 4. Perform runtime source compilation, and obtain kernel entry point.
  cl_program program = clCreateProgramWithSource(
                                                 context,
                                                 1,
                                                 &source,
                                                 NULL,
                                                 NULL);
  ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if(ret!=0) {
    const char * err = get_error_string(ret);
    printf("clBuildProgram: %s\n", err);
    return ret;
  }

  cl_kernel kernel = clCreateKernel(program, "memset", NULL);

  // 5. Create a data buffer.
  cl_mem buffer = clCreateBuffer(
                                 context,
                                 CL_MEM_WRITE_ONLY,
                                 NWITEMS * sizeof(cl_uint),
                                 NULL,
                                 NULL);
  // 6. Launch the kernel. Let OpenCL pick the local work size.
  size_t global_work_size = NWITEMS;
  ret = clSetKernelArg(kernel, 0, sizeof(buffer), (void *) &buffer);
  if(ret!=0) {
    puts("clSetKernelArg");
    return ret;
  }

  ret = clEnqueueNDRangeKernel(queue,
                         kernel,
                         1,
                         NULL,
                         &global_work_size,
                         NULL,
                         0,
                         NULL,
                         NULL);
  if(ret!=0) return ret;

  ret = clFinish(queue);
  if(ret!=0) {
    puts("clFinish");
    return ret;
  }

  // 7. Look at the results via synchronous buffer map.
  cl_uint *ptr;
  ptr = (cl_uint *) clEnqueueMapBuffer(
                                       queue,
                                       buffer,
                                       CL_TRUE,
                                       CL_MAP_READ,
                                       0,
                                       NWITEMS * sizeof(cl_uint),
                                       0,
                                       NULL,
                                       NULL,
                                       NULL);
  for(int i = 0; i < NWITEMS; i++) {
    printf("%d %d\n", i, ptr[i]);
  }

  return 0;
}


