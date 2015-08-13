/* clang-format off */
/**
 * @see http://suhorukov.blogspot.com/2011/12/opencl-11-atomic-operations-on-floating.html
 * @see  http://simpleopencl.blogspot.com/2013/05/atomic-operations-and-floats-in-opencl.html
 *
 * @param float*   source  [description]
 * @param float    operand [description]
 */
inline void atomic_add_global(volatile __global float* source, const float operand) {
  /* clang-format on */
  union {
    unsigned int intVal;
    float floatVal;
  } newVal;

  union {
    unsigned int intVal;
    float floatVal;
  } prevVal;

  // NOTE: atomic_cmpxchg(volatile __global unsigned int *p,
  // 	                    unsigned int cmp, unsigned int val)
  do {
    prevVal.floatVal = *source;
    newVal.floatVal = prevVal.floatVal + operand;
  } while (atomic_cmpxchg((volatile __global unsigned int*)source,
                          prevVal.intVal,  //
                          newVal.intVal) != prevVal.intVal);
}

/* clang-format off */
/**
 *
 * Calculate grad_w and grad_b. Very expensive due too barriers & locks.
 *
 *
 * In following notation (l), (l-1) describes relative layer and [...] lower indices.
 *
 * Algo for dJ/dw[abnk] on layer (l), where:
 *   a = 0..spatial_size(l)
 *   b = 0..spatial_size(l)
 *   n = 0..filter_count(l)
 *   k = 0..filter_count(l+1)
 *
 * dJ/dw[abnk] = 0
 * for i = 0..output_w(l):               # for each node where this weight is used
 * for j = 0..output_h(l):
 *   for n = 0..filter_count(l):
 *     for a = 0..spatial_size(l):       # offset to weight
 *     for b = 0..spatial_size(l):       # (it's kernel size, what You expect ?)
 *       for k = 0..filter_count(l-1):   # for all inputs
 *         dJ/dw[abnk] += deltas[i,j,n]  # (1) error for this point
 *           * layer_input[i+b,j+a,k]    # (2) input at this point
 */
/* clang-format on */
__kernel void backpropagate(__read_only __global float* deltas,       //
                            __read_only __global float* layer_input,  //
                            __global float* target_grad_w,            //
                            __global float* target_grad_b,            //
                            uint n_current_filter_cnt,                //
                            uint n_prev_filter_cnt,                   //
                            uint f_spatial_size,                      //
                            uint layer_out_w, uint layer_out_h) {
  const int id = get_global_id(0);
  const uint input_w = layer_out_w + f_spatial_size - 1;
  // weight dimensions
  const size_t d2 = n_prev_filter_cnt * n_current_filter_cnt,
               d3 = d2 * f_spatial_size;
  const size_t weights_size = d3 * f_spatial_size;

  // reverse id to get weight parameters: a(as dx), b(as dy), n, k
  int w_tmp = id;
  const int dy = w_tmp / d3;
  w_tmp -= dy * d3;
  const int dx = w_tmp / d2;
  w_tmp -= dx * d2;
  const int k = w_tmp / n_current_filter_cnt;
  const int n =
      w_tmp - k * n_current_filter_cnt;  // = id % n_current_filter_cnt

  if (id < weights_size) {
    float grad_w = 0.0f, grad_b = 0.0f;
    for (size_t row = 0; row < layer_out_h; row++) {
      for (size_t col = 0; col < layer_out_w; col++) {
        // (1) delta[i,j,n](l)
        int idx = ((row * layer_out_w) + col) * n_current_filter_cnt;
        float delta = deltas[idx + n];
        grad_b += delta;

        // (2) layer_input[i+b,j+a,k]
        // NOTE: we normally should be subtracting [dx,dy], but it does
        // depend on indexing
        int2 prev_layer_pos = {col + dx, row + dy};
        int prev_layer_idx = ((prev_layer_pos.y * input_w) + prev_layer_pos.x) *
                             n_prev_filter_cnt;

        float input = layer_input[prev_layer_idx + k];
        grad_w += input * delta;
      }
    }

    // write
    // NOTE: atomic_add_global is custom function, see beginning of the file
    target_grad_w[id] += grad_w;
    if (k == 0 && dx == 0 && dy == 0)
      atomic_add_global(target_grad_b + n, grad_b);
  }
}
