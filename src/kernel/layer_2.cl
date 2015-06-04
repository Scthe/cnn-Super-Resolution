float sigmoid(float x){
	return 1 / (1 + exp(-x));
}

// MACRO: N2_FILTER_COUNT filter count for second layer

/**
 * [main description]
 * @param source        output of first layer, of size (img_w-f1+1)*(img_h-f1+1)*n1
 * @param target        result buffer, of size (source_w-f2+1)*(source_h-f2+1)*n2,
 *                      where source_% means dimension for source parameter matrix
 * @param W             weights (for layer 2), of size f2*f2*n1 per each filter (total: f2*f2*n1*n2)
 * @param B             biases (for layer 2), of size n2
 * @param f1            spatial size for first layer
 * @param n1            filter count for first layer
 * @param f2            spatial size for second layer
 */
__kernel
void main(__read_only __global float* source,
					__global float* target,
					__global float* W,
					__global float* B,
					uint f1, uint n1, uint f2){
	const int half_f2 = f2 / 2;
	const int2 padding = {half_f2, half_f2}; // left top corner
	const int2 size_source = {f1, f1};
	const int2 br = size_source - padding; // bottom right corner
	const int2 pos = {get_global_id(0), get_global_id(1)};

	if(pos.x >= padding.x && pos.x <= br.x &&
		 pos.y >= padding.y && pos.y <= br.y){

			// 2D position on result cube(f1-f2+1, f1-f2+1, n2),
			// (result cube's first 2 dimensions(f1-f2+1, f1-f2+1) are smaller then source's(f1*f1))
			int2 pos_res = pos - padding; // 0,0
			int2 size_res = {f1 - f2 + 1, f1 - f2 + 1}; // w,h of the result // 1,1
			int base_idx = ((pos_res.y * size_res.x) + pos_res.x) * N2_FILTER_COUNT; // 0

			// result cache
			float vals_by_filter[N2_FILTER_COUNT];
			for (size_t filter_id = 0; filter_id < N2_FILTER_COUNT; filter_id++) {
				vals_by_filter[filter_id] = 0.0f;
			}

			for (size_t dy = 0; dy < f2; dy++) {
				for (size_t dx = 0; dx < f2; dx++) {
					int2 delta = {dx, dy};
					int2 point_pos = pos + delta - padding;
					int point_idx = (point_pos.y * f1 + point_pos.x) * f1;

					for (size_t i_n1 = 0; i_n1 < n1; i_n1++) {
						// for every feature map in source:
						float point_value = source[point_idx + i_n1];
						int base_W_idx = ((dy * f1) + dx) * f1;
						base_W_idx = (base_W_idx+i_n1) * N2_FILTER_COUNT;

						for (size_t filter_id = 0; filter_id < N2_FILTER_COUNT; filter_id++) {
							float W_value = W[base_W_idx + filter_id];
							vals_by_filter[filter_id] += W_value * point_value;
							// vals_by_filter[filter_id] += point_value; // part of tests
							// vals_by_filter[filter_id] += W_value; // part of tests
						}
						// target[point_idx+i_n1] = base_W_idx;  // part of W-value tests
						// target[point_idx+i_n1] = W[base_W_idx + 1];  // part of W-value tests
					}
				}
			}

			// write cached results to target buffer
			for (size_t filter_id = 0; filter_id < N2_FILTER_COUNT; filter_id++) {
				float B_value = B[filter_id];
				float result = vals_by_filter[filter_id] + B_value;
				// target[base_idx + filter_id] = result;
				target[base_idx + filter_id] = sigmoid(result);
			}

	}
}

///
/// TESTS:
///
// for (size_t filter_id = 0; filter_id < N2_FILTER_COUNT; filter_id++) {
	// target[base_idx+filter_id] = filter_id;
// }
// target[0] = pos_res.x;
// target[1] = pos_res.y;
// target[3] = size_res.x;
// target[4] = size_res.y;
// target[5] = base_idx;
