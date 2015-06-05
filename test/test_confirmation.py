import math
import json
from pprint import pprint
#
# This file contains python version of code in opencl kernels.
# It's purpose is to double check the test values.
# If both python and opencl versions return same values
# both scripts should work correctly.
#


def sigmoid(x):
  return 1.0/(1.0 + math.e**(-x))

def format_arr(arr, comma_places=3):
  return ', '.join([("%."+str(comma_places)+"f") % x for x in arr])

class Layer1:
  '''
  filter 1:
  []__[]
  __[]__
  []__[]

  filter 2:
  __[]__
  []__[]
  __[]__

  filter 3:
  ______
  __[]__
  ______
  '''

  def __call__(self, cfg):
    print('### LAYER 1')

    raw_in = cfg['input']
    mean = sum(raw_in) / len(raw_in)
    _in = [x - mean for x in raw_in]

    f1 = cfg['f1']
    n1 = cfg['n1']
    in_w = cfg['input_w']
    weights = cfg['weights']
    result = [ [[]] * f1,[[]] * f1,[[]] * n1]

    for i in range(f1*f1):
      x = i % f1 + 1
      y = i // f1 + 1
      # print(str(x)+":"+str(y))
      pos = x, y
      padding = f1//2, f1//2

      filter_vals = [0] * n1
      # apply weights
      for dy in range(f1):
        for dx in range(f1):
          delta = dx,dy
          pixel_pos = pos[0] + delta[0] - padding[0],\
                      pos[1] + delta[1] - padding[1]
          pixel_idx = pixel_pos[1] * in_w + pixel_pos[0]
          pixel_value = _in[pixel_idx]
          # print("\t"+str(pixel_pos)+"->"+str(pixel_value))
          base_W_idx = ((dy * f1) + dx) * f1
          for filter_id in range(n1):
            W_value = weights[base_W_idx + filter_id]
            filter_vals[filter_id] += W_value * pixel_value

      # apply bias
      biases = cfg['bias']
      for filter_id in range(n1):
        B_value = biases[filter_id]
        filter_vals[filter_id] += B_value

      filter_vals = list(map(sigmoid, filter_vals))
      print("{}:{} = {}".format(y, x, format_arr(filter_vals)))
      result[y-1][x-1] = filter_vals
    return result


class Layer2:

  def __call__(self, layer_1_cfg, layer_2_cfg):
    print('\n### LAYER 2')

    f1 = layer_1_cfg['f1']
    f2 = layer_2_cfg['f2']
    n1 = layer_1_cfg['n1']
    n2 = layer_2_cfg['n2']
    _in = layer_2_cfg['input']
    weights = layer_2_cfg['weights']
    bias = layer_2_cfg['bias']
    img_in = layer_1_cfg['input_w'], layer_1_cfg['input_h']
    out_dim = img_in[0] - f1 - f2 + 2,\
              img_in[1] - f1 - f2 + 2
    # print(out_dim)

    for i in range(out_dim[0]*out_dim[1]):
        x = i % f1  # +1 ?
        y = i // f1 # +1 ?
        # x = i % f1   +1
        # y = i // f1  +1
        pos = x, y
        padding = f2//2, f2//2
        vals_by_filter = [0.0] * n2

        # apply weights
        for dy in range(f2):
          for dx in range(f2):
              delta = dx,dy
              point_pos = pos[0] + delta[0] - padding[0],\
                          pos[1] + delta[1] - padding[1]
              point_idx = (point_pos[1] * f1 + point_pos[0]) * f1
              for i_n1 in range(n1):
                  point_value = _in[point_idx + i_n1]
                  base_W_idx = ((((dy * f1) + dx) * f1)+i_n1) * n2
                  # print(">  {}--{}:{} = {}".format(point_idx + i_n1, y, x, point_value))
                  for filter_id in range(n2):
                      vals_by_filter[filter_id] += weights[base_W_idx+filter_id] * point_value

        # add bias
        for filter_id in range(n2):
            vals_by_filter[filter_id] += bias[filter_id]
        print("{}:{} = {}".format(y, x, format_arr(vals_by_filter)))

if __name__ == '__main__':
    cfg = None
    with open('test/data/test_cases.json') as json_file:
        cfg = json.load(json_file)
    # pprint(cfg)

    l1 = Layer1()
    l2 = Layer2()

    layer2_in = l1(cfg['layer_1'])
    l2(cfg['layer_1'], cfg['layer_2']['data_set_1'])
    l2(cfg['layer_1'], cfg['layer_2']['data_set_2'])
