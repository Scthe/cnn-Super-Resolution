from PIL import Image

rgb2y   = [  0.299,    0.587,    0.114,  0.0]
rgb2Cb  = [-0.1687,  -0.3312,      0.5,  0.0]
rgb2Cr  = [    0.5,  -0.4186,  -0.0813,  0.0]
YCbCr2r = [1.0,     0.0,    1.4,  0.0]
YCbCr2g = [1.0,  -0.343, -0.711,  0.0]
YCbCr2b = [1.0,   1.765,    0.0,  0.0]

def dot(a,b):
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def extract_luma(img):
  luma_channel = []
  width, height = img.size
  pixels = img.load() # this is not a list, nor is it list()'able
  for y in range(height):
    for x in range(width):
      cpixel = pixels[x, y]
      luma_val = dot(cpixel, rgb2y)
      luma_channel.append(luma_val/255)
  l = int(len(luma_channel)**0.5)
  for i in range(l):
    xs = luma_channel[i*l : (i+1)*l]
    print(', '.join(["{:>6.3}".format(x) for x in xs]))

def swap_luma(img, padding, out_path):
  "To verify this works run it on f.e. 256*256 picture, but it may take some time (like 3 min or so)"
  print("Deprecated, see SwapLumaTest.cpp")
  raise Exception('SwapLuma does not produce acceptable result file')

  img_w, img_h = img.size

  # generate luma to swap into
  total_padding = padding * 2
  luma_w,luma_h = img_w - total_padding, img_h - total_padding
  new_luma_size = luma_w * luma_w
  new_luma = [(i/new_luma_size) for i in range(new_luma_size)]
  # print(new_luma)

  pixels = img.load() # this is not a list, nor is it list()'able
  for y in range(img_w):
    for x in range(img_h):
      pos_luma = x - padding, y - padding
      idx_luma = pos_luma[1] * luma_w + pos_luma[0]
      # idx = y * img_w + x
      cpixel = pixels[x, y] # 0..255

      if pos_luma[0] >= 0 and pos_luma[0] < luma_w and \
         pos_luma[1] >= 0 and pos_luma[1] < luma_h:
        raw_luma = new_luma[idx_luma]
        YCbCr = (raw_luma * 255, # 0..255
                 dot(rgb2Cb, cpixel),
                 dot(rgb2Cr, cpixel))
        clamp = lambda x: int(min(255, max(0, x)))
        new_color = (clamp(dot(YCbCr2r, YCbCr)), \
                     clamp(dot(YCbCr2g, YCbCr)), \
                     clamp(dot(YCbCr2b, YCbCr)))
      else:
        new_color = cpixel
      # print(new_color)
      pixels[x, y] = new_color
      img.save(out_path, "JPEG")



if __name__ == '__main__':
  extract_luma_img = Image.open("../data/color_grid.png")
  extract_luma(extract_luma_img)

  # swap_luma_img = Image.open(  "../data/color_grid2.jpg")
  # swap_luma_img = Image.open(  "../data/color_grid3.png")
  # swap_luma(swap_luma_img, 10, "../data/color_grid2_luma_swapped.png")
