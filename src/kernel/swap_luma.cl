__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |  //
                               CLK_ADDRESS_CLAMP_TO_EDGE |    //
                               CLK_FILTER_NEAREST;

// http://www.equasys.de/colorconversion.html
/* clang-format off */
__constant float4 rgb2y =  {   0.299f,    0.587f,    0.114f,  0.0f};
__constant float4 rgb2Cb = { -0.1687f,  -0.3312f,      0.5f,  0.0f};
__constant float4 rgb2Cr = {     0.5f,  -0.4186f,  -0.0813f,  0.0f};
// __constant float4 rgb2Cb = {-37.797f, -74.203f,   112.0f,  0.0f};
// __constant float4 rgb2Cr = { 112.0f,  -98.786f, -18.214f,  0.0f};

__constant float4 YCbCr2r = { 1.0f,     0.0f,    1.4f,  0.0f};
__constant float4 YCbCr2g = { 1.0f,  -0.343f, -0.711f,  0.0f};
__constant float4 YCbCr2b = { 1.0f,   1.765f,    0.0f,  0.0f};
/* clang-format on */

__kernel void swap_luma(__read_only image2d_t original_image,  //
                        __read_only __global float* new_luma,  //
                        __global uchar* target,                //
                        const uint ground_truth_w,
                        const uint ground_truth_h,  //
                        const uint luma_w, const uint luma_h) {
  const size_t padding = (ground_truth_w - luma_w) / 2;
  const int2 pos = {get_global_id(0), get_global_id(1)},
             pos_luma = {pos.x - padding, pos.y - padding};
  const size_t idx = pos.y * ground_truth_w + pos.x,
               idx_luma = pos_luma.y * luma_w + pos_luma.x;

  if (pos.x < 0 || pos.x >= ground_truth_w ||  //
      pos.y < 0 || pos.y >= ground_truth_h)
    return;

  const uint4 pixel_col = read_imageui(original_image, sampler, pos);
  const float4 pixel_col_f = convert_float4(pixel_col);
  uint3 new_color;
  if (pos_luma.x < 0 || pos_luma.x >= luma_w ||  //
      pos_luma.y < 0 || pos_luma.y >= luma_h) {
    // sample original image
    new_color.x = pixel_col.x;  // 0..255
    new_color.y = pixel_col.y;  // 0..255
    new_color.z = pixel_col.z;  // 0..255
  } else {
    // combine new luma with chroma from original image
    // to do this we first have to remove old luma
    // NOTE: during conversion we skip +128 / -128 steps as they cancel
    // themselves out
    // TODO after writing tests use matrix version of this code
    float raw_luma = new_luma[idx_luma];       // 0..1
    float4 YCbCr = {raw_luma * 255.0f,         // 0..255
                    dot(pixel_col_f, rgb2Cb),  // 0..255
                    dot(pixel_col_f, rgb2Cr),  // 0..255
                    0.0f};
    float3 rgb = {dot(YCbCr, YCbCr2r),  //
                  dot(YCbCr, YCbCr2g),  //
                  dot(YCbCr, YCbCr2b)};
    rgb = clamp(rgb, 0.0f, 255.0f);
    // TODO mix luma values in edges of new luma area, to make the transition
    // less jarring
    new_color.x = convert_uint(rgb.x);
    new_color.y = convert_uint(rgb.y);
    new_color.z = convert_uint(rgb.z);
  }

  // write
  target[idx * 3 + 0] = convert_uchar(new_color.x);
  target[idx * 3 + 1] = convert_uchar(new_color.y);
  target[idx * 3 + 2] = convert_uchar(new_color.z);
}
