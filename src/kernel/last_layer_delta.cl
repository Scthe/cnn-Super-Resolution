/* clang-format off */
/**
 * [main description]
 * @param  float*        ground_truth_image [description]
 * @param  float*        algo_result        [description]
 * @param  float*        target             [description]
 * @param  float         weight_decay       regularization term to bring the weights down
 * @param  uint          ground_truth_w     [description]
 * @param  uint          algo_result_w
 * @param  uint          algo_result_h
 * @return {[type]}             [description]
 */
/* clang-format on */
__kernel void last_layer_delta(__read_only __global float* ground_truth_image,
                               __read_only __global float* algo_result,
                               __global float* target,       //
                               __const uint ground_truth_w,  //
                               __const uint ground_truth_h,  //
                               __const uint algo_result_w,   //
                               __const uint algo_result_h) {
  const int2 pos = {get_global_id(0), get_global_id(1)};  // x=col=i, y=row=j
  const uint sample_id = get_global_id(2);
  const int2 out_size = {algo_result_w, algo_result_h};
  const int idx = (pos.y * algo_result_w) + pos.x;
  const size_t padding = (ground_truth_w - algo_result_w) / 2;

#define IMAGE_OFFSET_GT sample_id* ground_truth_w* ground_truth_h
#define IMAGE_OFFSET_ALGO sample_id* algo_result_w* algo_result_h

  // size of ground_truth != algo res (padding)
  // The offset is not const, since it depends on the row we are in
  // algo for ground_truth_idx:
  // (row + padding_on_top_of_image) * width + padding_left + col
  const size_t ground_truth_idx =
      (pos.y + padding) * ground_truth_w + padding + pos.x;

  if (pos.x >= 0 && pos.x < out_size.x &&  //
      pos.y >= 0 && pos.y < out_size.y) {
    // usuall square error derivative calculations
    float t = ground_truth_image[IMAGE_OFFSET_GT + ground_truth_idx];
    float y = algo_result[IMAGE_OFFSET_ALGO + idx];
    float d = y - t;

    // relu
    float relu_deriv = y > 0.0f ? 1.0f : 0.0f;

    // write result
    target[IMAGE_OFFSET_ALGO + idx] = d * relu_deriv;
  }
}
