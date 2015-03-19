// pi_initial.cl

__kernel
void pi(__global float * randSamples, __global float * results)
{
	int id = get_global_id(0);

	float x = randSamples[2*id    ];
	float y = randSamples[2*id + 1];
	
	if(x * x + y * y < 1.0f)
		results[id] = 1;
	else
		results[id] = 0;
}
