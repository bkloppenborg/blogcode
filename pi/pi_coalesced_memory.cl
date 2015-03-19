// pi_coalesced_memory.cl

#define WORK_SIZE 1000

__kernel
void pi(__global float2 * randSamples, __global float * results)
{
	int gid = get_global_id(0);
	int lid = get_local_id(0);
    int local_size = get_local_size(0);
	int offset = WORK_SIZE * get_group_id(0) * local_size;

	float count = 0;
	for(int i = 0; i < WORK_SIZE; i++)
	{
		float2 sample = randSamples[offset + local_size * i + lid]; 

		if(sample.x * sample.x + sample.y * sample.y < 1.0f)
			count += 1.0;
	}

	results[gid] = count;	
}
