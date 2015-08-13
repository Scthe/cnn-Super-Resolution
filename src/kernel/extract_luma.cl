__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |  //
                               CLK_ADDRESS_CLAMP_TO_EDGE |    //
                               CLK_FILTER_NEAREST;            //

__constant float4 rgb2y = {0.299f, 0.587f, 0.114f, 0.0f};

__kernel void extract_luma(__read_only image2d_t image,  //
                           __global float* target,       //
                           int w, int h) {
  const int2 pos = {get_global_id(0), get_global_id(1)};

  if (pos.x >= 0 && pos.x < w &&  //
      pos.y >= 0 && pos.y < h) {
    int idx = pos.y * w + pos.x;
    uint4 pixel_col = read_imageui(image, sampler, pos);
    float4 pixel_col_f = convert_float4(pixel_col);
#ifdef NORMALIZE
    target[idx] = dot(pixel_col_f, rgb2y) / 255.0f;
#else
    target[idx] = dot(pixel_col_f, rgb2y);
#endif  // NORMALIZE
  }
}
