// This file is part of EVR-cpp, a command line software for
// calculating chromosome structure based on interactive data
// using C++ language.
// 
// This file is the kernel file for OpenCL computing.

__kernel void evr(__global const float* positions,
	__global const float *matrix,
	__global float *trans,
	const float min_dis,
	const float max_dis,
	const int bin_num)
{
	int gid = get_global_id(0);

	float x = positions[gid * 3];
	float y = positions[gid * 3 + 1];
	float z = positions[gid * 3 + 2];

	float3 divec = (float3)(0.0, 0.0, 0.0);
	float3 ervec = (float3)(0.0, 0.0, 0.0);

	float mod = 0.0;
	float err = 0.0;
	float dis = 0.0;

	for (int i = 0; i < bin_num; i++)
	{
		if (i == gid)
			continue;

		divec.x = positions[i * 3] - x;
		divec.y = positions[i * 3 + 1] - y;
		divec.z = positions[i * 3 + 2] - z;

		mod = sqrt(divec.x * divec.x + divec.y * divec.y + divec.z * divec.z);

		dis = matrix[i * bin_num + gid];
		if (dis == 0.0 || mod == 0.0)
			continue;

		divec = divec / mod;
		if (i == (gid + 1) % bin_num || i == (gid - 1) % bin_num)
		{
			err = 0;
			if (mod > max_dis)
				err = mod - max_dis;
			if (mod < min_dis)
				err = mod - min_dis;
		}
		else
		{
			err = mod - dis;
		}
		divec = divec * err;
		ervec = ervec + divec;
	}
	ervec = ervec / bin_num;
	trans[gid * 3] += ervec.x;
	trans[gid * 3 + 1] += ervec.y;
	trans[gid * 3 + 2] += ervec.z;
}