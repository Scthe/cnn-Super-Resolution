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
__kernel void main(__read_only __global float* ground_truth_image,
                   __read_only __global float* algo_result,
                   __global float* target,       //
                   __const float weight_decay,   //
                   __const uint ground_truth_w,  //
                   __const uint algo_result_w,   //
                   __const uint algo_result_h) {
  const int2 pos = {get_global_id(0), get_global_id(1)};  // x=col=i, y=row=j
  const int2 out_size = {algo_result_w, algo_result_h};
  const int idx = (pos.y * algo_result_w) + pos.x;
  const size_t padding = (ground_truth_w - algo_result_w) / 2;

  // size of ground_truth != algo res (padding)
  // The offset is not const, since it depends on the row we are in
  // algo for ground_truth_idx:
  // (row + padding_on_top_of_image) * width + padding_left + col
  const size_t ground_truth_idx =
      (pos.y + padding) * ground_truth_w + padding + pos.x;

  if (pos.x >= 0 && pos.x < out_size.x &&  //
      pos.y >= 0 && pos.y < out_size.y) {
    // usuall mean square error derivative calculations
    float t = ground_truth_image[ground_truth_idx];
    float y = algo_result[idx];
    float d = y - t;
    // sigmoid
    float x = log(y / (1 - y));  // reverse sigmoid(log==ln)
    float sigm_deriv = x * (1 - x);
    // write result
    target[idx] = d * sigm_deriv + weight_decay;
  }
}
