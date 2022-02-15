#define CL_TARGET_OPENCL_VERSION 110

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define NDEVS 1

// A parallel min() kernel that works well on CPU and GPU

int
main()
{
  cl_platform_id  platform;
  cl_device_type devs[NDEVS] = {  CL_DEVICE_TYPE_GPU };

  cl_uint *src_ptr;
  unsigned int num_src_items = 4096*4096;

  // load source file
  const char *kernel_source;
  {
    struct stat stat_buf;
    const char *source_path = "./parallel_min.clc";
    char * buf;
    size_t size;
    int fd;
    if(stat(source_path, &stat_buf) == -1) {
      printf("stat\n");
      return -1;
    };
    size = stat_buf.st_size;
    buf = (char *)malloc(size);
    fd = open(source_path, O_RDONLY);
    if(read(fd, buf, size) == -1) {
      printf("read\n");
      return -1;
    }
    kernel_source = buf;
  }

  // 1. quick & dirty MWC random init of source buffer.
  // Random seed (portable).
  time_t ltime;
  time(&ltime);

  src_ptr = (cl_uint *) malloc(num_src_items * sizeof(cl_uint));

  cl_uint a = (cl_uint) ltime, b = (cl_uint) ltime;
  cl_uint min = (cl_uint) - 1;
  // Do serial computation of min() for result verification.
  for(unsigned int i = 0; i < num_src_items; i++) {
    src_ptr[i] = (cl_uint) (b = (a * (b & 65535)) + (b >> 16));
    min = src_ptr[i] < min ? src_ptr[i] : min;
  }
  printf("min: %d\n", min);

  // Get a platform.
  clGetPlatformIDs(1, &platform, NULL);


  // 3. Iterate over devices.
  for(int dev = 0; dev < NDEVS; dev++) {
    cl_device_id    device;
    cl_context     context;
    cl_command_queue queue;
    cl_program     program;
    cl_kernel         minp;
    cl_kernel       reduce;

    cl_mem         src_buf;
    cl_mem         dst_buf;
    cl_mem         dbg_buf;

    cl_uint       *dst_ptr,
                  *dbg_ptr;

    printf("\n%s: ", devs[dev] == CL_DEVICE_TYPE_CPU ? "CPU" : "GPU");
    // Find the device.
    clGetDeviceIDs(platform, devs[dev], 1, &device, NULL);

    // 4. Compute work sizez.
    cl_uint compute_units;
    size_t global_work_size;
    size_t local_work_size;
    size_t num_groups;

    clGetDeviceInfo(device,
                    CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(cl_uint),
                    &compute_units,
                    NULL);
    printf("compute units: %d\n", compute_units);
    if(devs[dev] == CL_DEVICE_TYPE_CPU) {
      global_work_size = compute_units * 1; // 1 thread per core
      local_work_size = 1;
    }
    else {
      cl_uint ws = 64;
      global_work_size = compute_units * 7 * ws; // 7 wavefronts per SIMD
      while((num_src_items / 4) % global_work_size != 0)
        global_work_size += ws;
      local_work_size = ws;
    }
    printf("global_work_size : %lu\n", global_work_size);
    num_groups = global_work_size / local_work_size;
    // Create a context and command queue on that device.
    context = clCreateContext(NULL,
                              1,
                              &device,
                              NULL,
                              NULL,
                              NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);
    // Minimal error check.
    if(queue == NULL) {
      printf("Compute device setup failed\n");
      return -1;
    }

    // Perform runtime source compilation, and obtain kernel entry point.
    program = clCreateProgramWithSource(context,
                                        1,
                                        &kernel_source,
                                        NULL,
                                        NULL);
    // Tell compiler to dump intermediate .il and .isa GPU files.
    cl_int ret = clBuildProgram(program,
                                1,
                                &device,
                                "-save-temps",
                                NULL,
                                NULL);
    // 5. Print compiler error messages
    if(ret != CL_SUCCESS) {
      printf("clBuildProgram failed: %d\n", ret);
      char buf[0x10000];
      clGetProgramBuildInfo(program,
                            device,
                            CL_PROGRAM_BUILD_LOG,
                            0x10000,
                            buf,
                            NULL);
      printf("\n%s\n", buf);
      return -1;
    }

    minp   = clCreateKernel(program, "minp", &ret);
    if(ret != CL_SUCCESS) {
      printf("minp kernel: %d\n", ret);
      return -1;
    }
    reduce = clCreateKernel(program, "reduce", &ret);
    if(ret != CL_SUCCESS) {
      printf("reduce kernel: %d\n", ret);
      return -1;
    }
    // Create input, output and debug buffer.
    src_buf = clCreateBuffer(context,
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             num_src_items * sizeof(cl_uint),
                             src_ptr,
                             &ret);
    if(ret != CL_SUCCESS) {
      printf("create src buffer: %d\n", ret);
      return -1;
    }
    dst_buf = clCreateBuffer(context,
                             CL_MEM_READ_WRITE,
                             num_groups * sizeof(cl_uint),
                             NULL,
                             &ret);
    if(ret != CL_SUCCESS) {
      printf("create dst buffer: %d\n", ret);
      return -1;
    }
    dbg_buf = clCreateBuffer(context,
                             CL_MEM_WRITE_ONLY,
                             global_work_size * sizeof(cl_uint),
                             NULL,
                             &ret);
    if(ret != CL_SUCCESS) {
      printf("create dbg buffer: %d\n", ret);
      return -1;
    }
    clSetKernelArg(minp, 0, sizeof(void *),        (void *) &src_buf);
    clSetKernelArg(minp, 1, sizeof(void *),        (void *) &dst_buf);
    clSetKernelArg(minp, 2, 1 * sizeof(cl_uint),   (void *) NULL);
    clSetKernelArg(minp, 3, sizeof(void *),        (void *) &dbg_buf);
    clSetKernelArg(minp, 4, sizeof(num_src_items), (void *) &num_src_items);
    clSetKernelArg(minp, 5, sizeof(dev),           (void *) &dev);

    clSetKernelArg(reduce, 0, sizeof(void *), (void *) &src_buf);
    clSetKernelArg(reduce, 1, sizeof(void *), (void *) &dst_buf);

    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    /* CPerfCounter t; */
    /* t.Reset(); */
    /* t.Start(); */

    // 6. Main timing loop.
    #define NLOOPS 500

    cl_event ev;
    int nloops = NLOOPS;

    while(nloops--) {
      cl_int ret = clEnqueueNDRangeKernel(queue,
                             minp,
                             1,
                             NULL,
                             &global_work_size,
                             &local_work_size,
                             0,
                             NULL,
                             &ev);
      if (ret != CL_SUCCESS) {
        printf("minp %d\n", ret);
        return -1;
      }
      ret = clEnqueueNDRangeKernel(queue,
                             reduce,
                             1,
                             NULL,
                             &num_groups,
                             NULL,
                             1,
                             &ev,
                             NULL);
      if (ret != CL_SUCCESS) {
        printf("reduce %d\n", ret);
        return -1;
      }
    }
    ret =  clFinish(queue);
    if (ret != CL_SUCCESS) {
      printf("finish %d\n", ret);
      return -1;
    }
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = ((1.0e9 * (double)(end.tv_sec - start.tv_sec)) + (double)(end.tv_nsec - start.tv_nsec)) / 1e9;

    printf("B/W %.2f GB/sec, ", ((float) num_src_items * sizeof(cl_uint) * NLOOPS) / elapsed / 1e9);

    // 7. Look at the results via synchronous buffer map.
    dst_ptr = (cl_uint *) clEnqueueMapBuffer(queue,
                                             dst_buf,
                                             CL_TRUE,
                                             CL_MAP_READ,
                                             0,
                                             num_groups * sizeof(cl_uint),
                                             0,
                                             NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
      printf("dst ptr %d\n", ret);
      return -1;
    }
    dbg_ptr = (cl_uint *) clEnqueueMapBuffer(queue,
                                             dbg_buf,
                                             CL_TRUE,
                                             CL_MAP_READ,
                                             0,
                                             global_work_size * sizeof(cl_uint),
                                             0,
                                             NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
      printf("dbg ptr %d\n", ret);
      return -1;
    }
    // 8. Print some debug info.
    printf("%d groups, %d threads, count %d, stride %d\n", dbg_ptr[0], dbg_ptr[1], dbg_ptr[2], dbg_ptr[3]);

    printf("computed value: %d\n", dst_ptr[0]);
    if(dst_ptr[0] == min)
      printf("result correct\n");
    else
      printf("result INcorrect\n");
  }

  printf("\n");
  return 0;
}
