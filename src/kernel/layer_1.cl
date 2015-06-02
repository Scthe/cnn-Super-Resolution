float sigmoid(float x){
	return 1 / (1 + exp(-x));
}

// http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html

/**
 * [main description]
 * @param source        single channel source image of size w*h
 * @param target        result buffer of size (w-f1+1)*(h-f1+1)*n1, zeroed
 * @param W             weights of size f1*f1 per each filter (total: f1*f1*n1)
 * @param B             biases of size f1*f1 per each filter (total: f1*f1*n1)
 * @param f1            spatial filter size
 * @param n1            filter count TODO use macro
 * @param w             source width
 * @param h             source height
 */
__kernel
void main(__read_only __global float* source,
					__global float* target,
					__global float* W,
					__global float* B,
					uint f1,
					int w, int h){
	const int half_f1 = f1 / 2; // 1
	// left top corner
	const int2 padding = {half_f1, half_f1}; // 1,1
	const int2 source_size = {w, h}; // 5,5
	// bottom right corner
	const int2 br = source_size - padding; // 4,4
	int2 pos = {get_global_id(0), get_global_id(1)};


	if(pos.x >= padding.x && pos.x <= br.x &&
		 pos.y >= padding.y && pos.y <= br.y){

		// 2d position on result cube(w-f1+1, h-f1+1, n1),
		// that is smaller then source image(w*h)
		int2 pos_res = pos - padding;
		int2 size_res = {w - f1 + 1, h - f1 + 1}; // w,h of the result
		int base_idx = ((pos_res.y * size_res.x) + pos_res.x) * N1_FILTER_COUNT;

		// result cache
		float vals_by_filter[N1_FILTER_COUNT];
		for (size_t filter_id = 0; filter_id < N1_FILTER_COUNT; filter_id++) {
			vals_by_filter[filter_id] = 0.0f;
		}

		for (size_t dy = 0; dy < f1; dy++) { // TODO double check loop order for cache locality
			for (size_t dx = 0; dx < f1; dx++) {
				// for every pixel in patch:
				int2 delta = {dx, dy};
				int2 pixel_pos = pos + delta - padding;
				int pixel_idx = pixel_pos.y * w + pixel_pos.x;
				float pixel_value = source[pixel_idx];
				int base_W_idx = ((dy * size_res.x) + dx) * N1_FILTER_COUNT;
				for (size_t filter_id = 0; filter_id < N1_FILTER_COUNT; filter_id++) {
					float W_value = W[base_W_idx + filter_id];
					float B_value = B[filter_id];
					vals_by_filter[filter_id] += W_value * pixel_value + B_value;
				}
			}
		}
		for (size_t filter_id = 0; filter_id < N1_FILTER_COUNT; filter_id++) {
			target[base_idx + filter_id] = sigmoid(vals_by_filter[filter_id]);
			// target[base_idx + filter_id] = vals_by_filter[filter_id];
			// target[base_idx + filter_id] = sum;
		}

	//
	// TESTS:
	//
		/*
		for (size_t filter_id = 0; filter_id < N1_FILTER_COUNT; filter_id++) {
			target[base_idx + filter_id] = filter_id;
			// target[base_idx + filter_id] = half_f1;
			// target[base_idx + filter_id] = 1;
		}

		// target[base_idx +0] = pos_res.s0;
		// target[base_idx +1] = pos_res.s1;
		// target[base_idx +2] = 0;
		*/

		// W test:
		/*
		int x = 1, y = 1; // 0.51
		int base_W_idx = ((y * size_res.x) + x) * N1_FILTER_COUNT;
		target[base_idx +0] = W[base_W_idx+0];
		target[base_idx +1] = W[base_W_idx+1];
		target[base_idx +2] = W[base_W_idx+2];
		*/
	}
}
