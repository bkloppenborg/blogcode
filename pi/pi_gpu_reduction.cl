// pi_gpu_reduction.cl

#define WORK_SIZE 1000

__kernel
void pi(__global float * randSamples, __global float * results)
{
	int id = get_global_id(0);
	int offset = 2 * WORK_SIZE * id;

	float count = 0;
	for(int i = 0; i < WORK_SIZE; i++)
	{
		float x = randSamples[offset + 2*i    ];
		float y = randSamples[offset + 2*i + 1];

		if(x*x + y*y < 1.0f)
			count += 1.0;
	}

	results[id] = count;	
}
