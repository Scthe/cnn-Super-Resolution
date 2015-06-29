/**
 * Part of mean square error calculations. Here we take 2 same sized,
 * single color channel buffers with image data and get the difference
 * between respective pixels.
 */
__kernel void main(__read_only __global float* ground_truth_image,
                   __read_only __global float* algo_result,
                   __global float* target,
                   __const uint ground_truth_w,
                   __const uint algo_result_w,
                   __const uint algo_result_size) {
  const int global_index = get_global_id(0);

  if (global_index < algo_result_size) {
    // size of ground_truth != algo res (padding)
    // The offset is not const, since it depends on the row we are in
    size_t padding = (ground_truth_w - algo_result_w) / 2,
           row = global_index / algo_result_w,
           col = global_index % algo_result_w,
           g_t_idx = (row + padding) * ground_truth_w + padding + col;
    // or g_t_idx = global_index + padding * (row + ground_truth_w - padding);

    float a = ground_truth_image[g_t_idx];
    float b = algo_result[global_index];
    float d = a - b;
    target[global_index] = d * d;
  }
}
