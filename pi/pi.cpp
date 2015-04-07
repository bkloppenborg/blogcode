


// Several standalone methods of computing Pi via. Monte Carlo methods
// using OpenCL.

// Variables defined via. compiler directives
//  KERNEL_SOURCE_DIR

#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <CL/cl.hpp>
#include "cl_helpers.hpp"
#include "HighResClock.hpp"

#ifndef ulong
typedef unsigned long ulong;
#endif

using namespace std;
using namespace cl;
using namespace chrono;
using namespace timer;

const int WORK_SIZE = 1000;
const int samples = 2E7;

/// Prints a header for the results
void print_result_header()
{
    cout << "|     Method     |     Device     | Pi estimate | GPU (usec) | CPU (usec) | Total (usec) |" << endl;
    cout << "|----------------|----------------|-------------|------------|------------|--------------|" << endl;
}

/// Prints an individual result
void print_result(const string & method, const string & deviceName,
    float pi_estimate,
    float kernel_time_usec, float cpu_time_usec, float total_time)
{
    cout << left << "| " << setw(15) << method;
    cout << left << "| " << setw(15) << deviceName;
    cout << "| " << setw(12) << pi_estimate;
    cout << "| " << setw(11) << kernel_time_usec;
    cout << "| " << setw(11) << cpu_time_usec;
    cout << "| " << setw(13) << total_time;
    cout << "| " << endl;
}

/// Compute Pi on the CPU using traditional Monte Carlo Methods
void pi_cpu(void)
{
	int count = 0;
	float x, y;
    vector<float> randomNums(2*samples);

    // init random numbers
    for(int i = 0; i < 2*samples; i++)	{
    	randomNums[i] = float(rand()) / RAND_MAX;
    }

	// start the timer
	auto start = HighResClock::now();

	// count how many samples reside in the circle
    for(int i = 0; i < samples; i++)
	{
		x = randomNums[2*i];
		y = randomNums[2*i + 1];

		if(x*x + y*y < 1.0)
			count++;
	}

	// end the timer and calculate the execution time
	auto stop = HighResClock::now();
	auto time = duration_cast<microseconds>(stop - start).count();

    // calculate the estimate and print the result
    float pi_estimate = 4 * float(count) / samples;
    print_result("Single Core CPU", "CPU", pi_estimate, 0, time, time);
}

/// A direct port of the above CPU algorithim to the GPU with no optimizations
/// The kernel, pi_initial.cl is located in this directory.
void pi_initial(cl::Context context, cl::Device device,
    cl::CommandQueue queue)
{
	vector<float> h_randNums(2*samples);
	vector<float> h_results(samples);
    vector<cl::Device> devices;
    devices.push_back(device);

	srand(time(NULL));

	for(int i = 0; i < 2*samples; i++)
	{
		h_randNums[i] = float(rand()) / RAND_MAX;
	}

    string programSource = readFile(KERNEL_SOURCE_DIR "/pi_initial.cl");
    cl::Program::Sources sources;
    sources.push_back(std::make_pair(programSource.c_str(), programSource.size()));
    cl::Program program(context, sources);
	try {
	   program.build(devices);
	}
	catch ( cl::Error & e) {
		cout << e.what() << endl;
		cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;

	}

    //Wait for program to build
    while(program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) != CL_BUILD_SUCCESS) {
        //sleep(1);
    }

    vector<Kernel> kernels;
    program.createKernels(&kernels);

	// Setup buffers
	::size_t size = samples * sizeof(float);
	Buffer d_randNums(context, CL_MEM_COPY_HOST_PTR, 2*size, h_randNums.data());
	Buffer d_results(context, CL_MEM_WRITE_ONLY, size);

	// timer
	cl::Event launch;
	auto start = HighResClock::now();

    // Setup and launch kernel
	kernels[0].setArg(0, d_randNums);
	kernels[0].setArg(1, d_results);
	queue.enqueueNDRangeKernel(kernels[0], 0, NDRange(samples),
        cl::NullRange, nullptr, &launch);

	launch.wait();
	ulong g_start = launch.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	ulong g_stop  = launch.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	float gpu_time = (g_stop - g_start) * 1E-3f;

    // Retrieve results (sum on the CPU)
	queue.enqueueReadBuffer(d_results, CL_TRUE, 0, size, h_results.data());

    auto sum_start = HighResClock::now();
	int sum = 0;
	for(auto sample: h_results)
		sum += sample;

	auto stop = HighResClock::now();
	auto cpu_time = duration_cast<microseconds>(stop - sum_start).count();
	auto total_time = duration_cast<microseconds>(stop - start).count();

	float estimatedValue = 4.0 * sum / samples;

    string deviceName = device.getInfo<CL_DEVICE_NAME>();
    print_result("OCL naive", deviceName, estimatedValue, gpu_time, cpu_time, total_time);
}

/// Calculate Pi via. Monte Carlo methods. Unlike `pi_initial` above, this
/// implementation reduces the load on the CPU by making each thread responsible
/// for multiple samples.
void pi_gpu_reduction(cl::Context context, cl::Device device,
    cl::CommandQueue queue)
{
	vector<float> h_randNums(2*samples);
	vector<float> h_results(samples/WORK_SIZE);
    vector<cl::Device> devices;
    devices.push_back(device);

	srand(time(NULL));

	for(int i = 0; i < 2*samples; i++)
	{
		h_randNums[i] = float(rand()) / RAND_MAX;
	}

    string programSource = readFile(KERNEL_SOURCE_DIR "/pi_gpu_reduction.cl");
    cl::Program::Sources sources;
    sources.push_back(std::make_pair(programSource.c_str(), programSource.size()));
    cl::Program program(context, sources);
	try {
	   program.build(devices);
	}
	catch ( cl::Error & e) {
		cout << e.what() << endl;
		cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;

	}

    //Wait for program to build
    while(program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) != CL_BUILD_SUCCESS) {
        //sleep(1);
    }

    vector<Kernel> kernels;
    program.createKernels(&kernels);

	// Setup buffers
	Buffer d_randNums(context, CL_MEM_COPY_HOST_PTR,
		h_randNums.size() * sizeof(float), h_randNums.data());
	Buffer d_results(context, CL_MEM_WRITE_ONLY, h_results.size() * sizeof(float));

	// timer
	cl::Event launch;
	auto start = HighResClock::now();

    // Setup and launch kernel
	kernels[0].setArg(0, d_randNums);
	kernels[0].setArg(1, d_results);
	queue.enqueueNDRangeKernel(kernels[0], 0, NDRange(h_results.size()),
        cl::NullRange, nullptr, &launch);

	launch.wait();
	ulong g_start = launch.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	ulong g_stop  = launch.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	float gpu_time = (g_stop - g_start) * 1E-3f;

    // Retrieve results (sum on the CPU)
	queue.enqueueReadBuffer(d_results, CL_TRUE, 0,
		h_results.size() * sizeof(float), h_results.data());


	auto sum_start = HighResClock::now();
	int sum = 0;
	for(auto sample: h_results)
		sum += sample;

	auto stop = HighResClock::now();
	auto cpu_time = duration_cast<microseconds>(stop - sum_start).count();
	auto total_time = duration_cast<microseconds>(stop - start).count();

	float estimatedValue = 4.0 * sum / samples;

    string deviceName = device.getInfo<CL_DEVICE_NAME>();
    print_result("OCL reduction", deviceName, estimatedValue, gpu_time, cpu_time, total_time);
}

/// Compute Pi on the GPU via. Monte Carlo methods. This method improves upon
/// the memory access patterns  of `pi_gpu_reduction` by using float2 to access
/// the sample points.
void pi_coalesced_memory(cl::Context context, cl::Device device,
    cl::CommandQueue queue)
{
	vector<float> h_randNums(2*samples);
	vector<float> h_results(samples/WORK_SIZE);
    vector<cl::Device> devices;
    devices.push_back(device);

	srand(time(NULL));

	for(int i = 0; i < 2*samples; i++)
	{
		h_randNums[i] = float(rand()) / RAND_MAX;
	}

    string programSource = readFile(KERNEL_SOURCE_DIR "/pi_coalesced_memory.cl");
    cl::Program::Sources sources;
    sources.push_back(std::make_pair(programSource.c_str(), programSource.size()));
    cl::Program program(context, sources);
	try {
	   program.build(devices);
	}
	catch ( cl::Error & e) {
		cout << e.what() << endl;
		cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;

	}

    //Wait for program to build
    while(program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) != CL_BUILD_SUCCESS) {
        //sleep(1);
    }

    vector<Kernel> kernels;
    program.createKernels(&kernels);

	// Setup buffers
	Buffer d_randNums(context, CL_MEM_COPY_HOST_PTR,
		h_randNums.size() * sizeof(float), h_randNums.data());
	Buffer d_results(context, CL_MEM_WRITE_ONLY, h_results.size() * sizeof(float));

	// timer
	cl::Event launch;
	auto start = HighResClock::now();

    // Setup and launch kernel
	kernels[0].setArg(0, d_randNums);
	kernels[0].setArg(1, d_results);
	queue.enqueueNDRangeKernel(kernels[0], 0, NDRange(h_results.size()),
        cl::NullRange, nullptr, &launch);

	launch.wait();
	ulong g_start = launch.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	ulong g_stop  = launch.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	float gpu_time = (g_stop - g_start) * 1E-3f;

    // Retrieve results (sum on the CPU)
	queue.enqueueReadBuffer(d_results, CL_TRUE, 0,
		h_results.size() * sizeof(float), h_results.data());

	auto sum_start = HighResClock::now();
	int sum = 0;
	for(auto sample: h_results)
		sum += sample;

	auto stop = HighResClock::now();
	auto cpu_time = duration_cast<microseconds>(stop - sum_start).count();
	auto total_time = duration_cast<microseconds>(stop - start).count();

	float estimatedValue = 4.0 * sum / samples;

    string deviceName = device.getInfo<CL_DEVICE_NAME>();
    print_result("OCL Coalesced", deviceName, estimatedValue, gpu_time, cpu_time, total_time);
}

/// Compute Pi on the GPU via. Monte Carlo methods. This method improves on
/// `pi_coalesced_memory` by using zero copy buffers. The performance increase
/// should only be apparent on integrated graphics devices.
void pi_zero_copy(cl::Context context, cl::Device device,
    cl::CommandQueue queue)
{
	vector<float> h_randNums(2*samples);
	int result_size = samples / WORK_SIZE;
    vector<cl::Device> devices;
    devices.push_back(device);

	srand(time(NULL));

	for(int i = 0; i < 2*samples; i++)
	{
		h_randNums[i] = float(rand()) / RAND_MAX;
	}

    string programSource = readFile(KERNEL_SOURCE_DIR "/pi_coalesced_memory.cl");
    cl::Program::Sources sources;
    sources.push_back(std::make_pair(programSource.c_str(), programSource.size()));
    cl::Program program(context, sources);
	try {
	   program.build(devices);
	}
	catch ( cl::Error & e) {
		cout << e.what() << endl;
		cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;

	}

    //Wait for program to build
    while(program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) != CL_BUILD_SUCCESS) {
        //sleep(1);
    }

    vector<Kernel> kernels;
    program.createKernels(&kernels);

	// Setup zero-copy buffers
	Buffer d_randNums(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
		h_randNums.size() * sizeof(float), h_randNums.data());
	Buffer d_results(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY, result_size * sizeof(float));

	// timer
	cl::Event launch;
	auto start = HighResClock::now();

    // Setup and launch kernel
	kernels[0].setArg(0, d_randNums);
	kernels[0].setArg(1, d_results);
	queue.enqueueNDRangeKernel(kernels[0], 0, NDRange(result_size),
        cl::NullRange, nullptr, &launch);

	launch.wait();
	ulong g_start = launch.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	ulong g_stop  = launch.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	float gpu_time = (g_stop - g_start) * 1E-3f;

    // Retrieve results (sum on the CPU)
    float * h_results = (float*) queue.enqueueMapBuffer(d_results, CL_TRUE, CL_MAP_READ, 0, result_size * sizeof(float));

	auto sum_start = HighResClock::now();
	int sum = 0;
	for(int i = 0; i < result_size; i++)
		sum += h_results[i];

    // release the buffer back to OpenCL’s control
    queue.enqueueUnmapMemObject(d_results, h_results);

	auto stop = HighResClock::now();
	auto cpu_time = duration_cast<microseconds>(stop - sum_start).count();
	auto total_time = duration_cast<microseconds>(stop - start).count();

	float estimatedValue = 4.0 * sum / samples;

    string deviceName = device.getInfo<CL_DEVICE_NAME>();
    print_result("OCL ZeroCopy", deviceName, estimatedValue, gpu_time, cpu_time, total_time);
}


void run_benchmarks(cl::Device & device)
{
	// create a command queue
	cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;

	vector<cl::Device> devices;
	devices.push_back(device);
	cl::Context context(devices);
	cl::CommandQueue queue(context, device, properties, NULL);
	pi_initial(context, device, queue);
	pi_gpu_reduction(context, device, queue);
	pi_coalesced_memory(context, device, queue);
	pi_zero_copy(context, device, queue);

	return;
}

int main()
{
	print_result_header();

	// get a list of platforms
	vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);

	// print out results for all devices
	for (auto platform : all_platforms)
	{
		vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

		for (auto device : devices)
		{
			run_benchmarks(device);
		}
	}

	pi_cpu();
}
