// pi_float4.cl

#define WORK_SIZE 1000

__kernel
void pi(__global float4 * g_x, __global float4 * g_y,
    __global float * results)
{
	int gid = get_global_id(0);
	int lid = get_local_id(0);
    int local_size = get_local_size(0);
	int offset = WORK_SIZE * get_group_id(0) * local_size;

	float count = 0;
	for(int i = 0; i < WORK_SIZE; i++)
	{
        float4 x = g_x[offset + local_size * i + lid];
        float4 y = g_y[offset + local_size * i + lid];

        float4 distance = x*x + y*y;

        if(distance.x < 1.0)
            count += 1.0;
        if(distance.y < 1.0)
            count += 1.0;
        if(distance.z < 1.0)
            count += 1.0;
        if(distance.w < 1.0)
            count += 1.0;
	}

	results[gid] = count;
}
