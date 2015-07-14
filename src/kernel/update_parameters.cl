__kernel void main(__read_only __global float* weights,                 //
                   __read_only __global float* bias,                    //
                   __read_only __global float* grad_weights,            //
                   __read_only __global float* grad_bias,               //
                   __read_only __global float* previous_delta_weights,  //
                   __read_only __global float* previous_delta_bias,     //
                   __const float momentum,                              //
                   __const float learning_rate,                         //
                   __const uint weights_size,                           //
                   __const uint bias_size) {
  const size_t idx = get_global_id(0);

  // update weights
  if (idx < weights_size) {
    float delta_w = momentum * previous_delta_weights[idx] +
                    learning_rate * grad_weights[idx];
    weights[idx] -= delta_w;
    previous_delta_weights[idx] = delta_w;
  }

  // update bias
  if (idx < bias_size) {
    float delta_b = momentum * previous_delta_bias[idx] +  //
                    learning_rate * grad_bias[idx];
    bias[idx] -= delta_b;
    previous_delta_bias[idx] = delta_b;
  }
}
