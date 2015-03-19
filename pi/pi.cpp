/*************************************************
** ArrayFire Training Day 1						**
** Monte Carlo Pi Estimation					**
**												**
** This program will estimate pi by calculating **
** the ratio of	points that fall inside of a	**
** unit circle with the points that did not		**
**************************************************/

#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <CL/cl.hpp>
#include "cl_helpers.hpp"
#include <chrono>

using namespace std;
using namespace std::chrono;
using namespace cl;

const cl_device_type DEVICE_TYPE = CL_DEVICE_TYPE_GPU;
const int WORK_SIZE = 1000;
const int samples = 2E7;

void print_result_header()
{
    cout << "|     Method     |     Device     | Pi estimate | GPU (usec) | CPU (usec) | Total (usec) |" << endl;
    cout << "|----------------|----------------|-------------|------------|------------|--------------|" << endl;
}

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
	auto start = high_resolution_clock::now();

	// count how many samples reside in the circle
    for(int i = 0; i < samples; i++) 
	{
		x = randomNums[2*i]; 
		y = randomNums[2*i + 1];
    
		if(x*x + y*y < 1.0)
			count++;
	}

	// end the timer and calculate the execution time
	auto stop = high_resolution_clock::now();
	auto time = duration_cast<microseconds>(stop - start).count();

    // calculate the estimate and print the result
    float pi_estimate = 4 * float(count) / samples;
    print_result("Single Core CPU", "CPU", pi_estimate, 0, time, time);
}


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

    string programSource = readFile("pi_initial.cl");
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
	auto start = high_resolution_clock::now();

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

    auto sum_start = high_resolution_clock::now();
	int sum = 0;
	for(auto sample: h_results)
		sum += sample;

	auto stop = high_resolution_clock::now();
	auto cpu_time = duration_cast<microseconds>(stop - sum_start).count();
	auto total_time = duration_cast<microseconds>(stop - start).count();

	float estimatedValue = 4.0 * sum / samples;

    string deviceName = device.getInfo<CL_DEVICE_NAME>();
    print_result("OCL naive", deviceName, estimatedValue, gpu_time, cpu_time, total_time);

}

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

    string programSource = readFile("pi_gpu_reduction.cl");
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
	auto start = high_resolution_clock::now();

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


	auto sum_start = high_resolution_clock::now();
	int sum = 0;
	for(auto sample: h_results)
		sum += sample;

	auto stop = high_resolution_clock::now();
	auto cpu_time = duration_cast<microseconds>(stop - sum_start).count();
	auto total_time = duration_cast<microseconds>(stop - start).count();

	float estimatedValue = 4.0 * sum / samples;

    string deviceName = device.getInfo<CL_DEVICE_NAME>();
    print_result("OCL reduction", deviceName, estimatedValue, gpu_time, cpu_time, total_time);
}


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

    string programSource = readFile("pi_coalesced_memory.cl");
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
	auto start = high_resolution_clock::now();

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

	auto sum_start = high_resolution_clock::now();
	int sum = 0;
	for(auto sample: h_results)
		sum += sample;

	auto stop = high_resolution_clock::now();
	auto cpu_time = duration_cast<microseconds>(stop - sum_start).count();
	auto total_time = duration_cast<microseconds>(stop - start).count();

	float estimatedValue = 4.0 * sum / samples;

    string deviceName = device.getInfo<CL_DEVICE_NAME>();
    print_result("OCL Coalesced", deviceName, estimatedValue, gpu_time, cpu_time, total_time);
}

int main()
{
    // get an OpenCL context and setup the device
    cl::Context context(DEVICE_TYPE);
    vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	
    // create a command queue
	cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;

    print_result_header();

	pi_cpu();
    
    for(cl::Device device: devices)
    {
	    cl::CommandQueue queue(context, device, properties, NULL);
	    pi_initial(context, device, queue);
	    pi_gpu_reduction(context, device, queue);	
	    pi_coalesced_memory(context, device, queue);
    }

	return 0;
}
