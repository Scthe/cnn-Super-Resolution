__kernel void main(__read_only __global float* weights,                 //
                   __read_only __global float* bias,                    //
                   __read_only __global float* grad_weights,            //
                   __read_only __global float* grad_bias,               //
                   __read_only __global float* previous_delta_weights,  //
                   __read_only __global float* previous_delta_bias,     //
                   __const float momentum,                              //
                   __const float weight_decay_parameter,                //
                   __const float learning_rate,                         //
                   __const uint batch_size,                             //
                   __const uint weights_size,                           //
                   __const uint bias_size) {
  const size_t idx = get_global_id(0);

  // update weights
  if (idx < weights_size) {
    float weight_value = weights[idx];
    float delta_w = momentum * previous_delta_weights[idx] +
                    learning_rate * grad_weights[idx] +
                    weight_decay_parameter * weight_value;
    weights[idx] = weight_value - delta_w / batch_size;
    previous_delta_weights[idx] = delta_w;
  }

  // update bias
  if (idx < bias_size) {
    float delta_b = momentum * previous_delta_bias[idx] +  //
                    learning_rate * grad_bias[idx];
    bias[idx] -= delta_b / batch_size;
    previous_delta_bias[idx] = delta_b;
  }
}
