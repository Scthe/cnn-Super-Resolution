__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
															 CLK_ADDRESS_CLAMP_TO_EDGE |
															 CLK_FILTER_NEAREST;

__kernel
void main(__read_only image2d_t image,
					__global uchar* target,
					int w, int h){

	// const uint w = get_global_size(0);
	// const uint h = get_global_size(1);
	const int2 pos = {get_global_id(0), get_global_id(1)};
	float2 normCoor = convert_float2(pos) / (float2)( w, h );

	// TODO remember about bpp !!!
	if(pos.x >= 0 && pos.x < w && pos.y >= 0 && pos.y < h){
		int idx = pos.y * w + pos.x;

		uint4 pixel_col = read_imageui(image, sampler, pos);
		target[idx] = (uchar)pixel_col.x;
	}

}
