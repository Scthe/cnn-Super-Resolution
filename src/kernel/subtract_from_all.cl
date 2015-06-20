__kernel
void main(__global float* data,
          __const float value,
          __const uint len){
  const int global_index = get_global_id(0);
  if (global_index < len) {
    data[global_index] = data[global_index] - value;
  }
}
