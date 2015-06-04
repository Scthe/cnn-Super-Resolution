import math

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
  __in = [0.0,     255.0,   207.073, 217.543, 111.446,
          43.246,  178.755, 105.315, 225.93,  200.578,
          109.577, 76.245,  149.685, 29.07,   180.345,
          170.892, 190.035, 217.543, 190.035, 76.278,
          205.58,  149.852, 218.917, 151.138, 179.001]

  _w = [1, 0, 0,
        0, 1, 0,
        1, 0, 0,
        0, 1, 0,
        1, 0, 1,
        0, 1, 0,
        1, 0, 0,
        0, 1, 0,
        1, 0, 0];
  _b = [0.1, 0.2, 0.3]

  def preprocess(self, data):
    a1 = [x / 255.0 for x in data]
    # return a1
    mean = sum(a1) / len(a1)
    return [x - mean for x in a1]


  def __call__(self):
    print('### LAYER 1')
    _in = self.preprocess(self.__in)
    result = [ [[]] * 3,[[]] * 3,[[]] * 3]

    for i in range(9):
      x = i % 3 + 1
      y = i // 3 + 1
      # print(str(x)+":"+str(y))
      pos = x, y
      padding = 1,1

      filter_vals = [0,0,0]
      # apply weights
      for dy in range(3):
        for dx in range(3):
          delta = dx,dy
          pixel_pos = pos[0] + delta[0] - padding[0],\
                      pos[1] + delta[1] - padding[1]
          pixel_idx = pixel_pos[1] * 5 + pixel_pos[0]
          pixel_value = _in[pixel_idx]
          # print("\t"+str(pixel_pos)+"->"+str(pixel_value))
          base_W_idx = ((dy * 3) + dx) * 3
          for filter_id in range(3):
            W_value = self._w[base_W_idx + filter_id]
            filter_vals[filter_id] += W_value * pixel_value

      for filter_id in range(3):
        B_value = self._b[filter_id]
        filter_vals[filter_id] += B_value

      filter_vals = list(map(sigmoid, filter_vals))
      print("{}:{} = {}".format(y, x, format_arr(filter_vals)))
      result[y-1][x-1] = filter_vals
    return result




class Layer2:
  _w1 = [[(1.000,1.001,1.002), # row 0
          (1.010,1.011,1.012),
          (1.020,1.021,1.022)],
         [(1.100,1.101,1.102), # row 1
          (1.110,1.111,1.112),
          (1.120,1.121,1.122)],
         [(1.200,1.201,1.202), # row 2
          (1.210,1.211,1.212),
          (1.220,1.221,1.222)]]
  _w2 = [[(2.000,2.001,2.002), # row 0
          (2.010,2.011,2.012),
          (2.020,2.021,2.022)],
         [(2.100,2.101,2.102), # row 1
          (2.110,2.111,2.112),
          (2.120,2.121,2.122)],
         [(2.200,2.201,2.202), # row 2
          (2.210,2.211,2.212),
          (2.220,2.221,2.222)]]
  _b = [0.1, 0.2]

  def __call__(self, _in):
    print('\n### LAYER 2')
    sum_ = [0.0,0.0]
    for dy in range(3):
      for dx in range(3):
        # print(str(dy+1)+":"+str(dx+1)+" = "+format_arr(_in[dy][dx]))
        # W_mul = sum(self._w1[dy][dx]), sum(self._w2[dy][dx])
        W_mul = sum([ a*b for a,b in zip(self._w1[dy][dx], _in[dy][dx])]), \
                sum([ a*b for a,b in zip(self._w2[dy][dx], _in[dy][dx])])
        # W_mul = sum(_in[dy][dx]), sum(_in[dy][dx])
        res = W_mul[0], W_mul[1]
        sum_[0] += res[0]
        sum_[1] += res[1]
    sum_[0] += self._b[0]
    sum_[1] += self._b[1]
    print ("f1: {:.6} f2: {:.6}".format(sum_[0], sum_[1]))




if __name__ == '__main__':
  l1 = Layer1()
  l2 = Layer2()

  layer2_in = l1()
  l2(layer2_in)

  # a = [0.0,     255.0,   207.073, 217.543, 111.446,43.246,  178.755, 105.315, 225.93,  200.578,109.577, 76.245,  149.685, 29.07,   180.345,170.892, 190.035, 217.543, 190.035, 76.278, 205.58,  149.852, 218.917, 151.138, 179.001]
  # a1 = [x / 255.0 for x in a]
  # print( ','.join(["{:.5}f".format(x) for x in a1]))





'''
############################
LAYER 1:
############################


1:1 = 1124.896
2:1 = 1444.6159999999998
3:1 = 1426.985
1:2 = 1241.2930000000001
2:2 = 1362.613
3:2 = 1374.7790000000002
1:3 = 1488.326
2:3 = 1372.5199999999998
3:3 = 1392.012

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

1f, 0f, 0f,   // 0,0
0f, 1f, 0f,   // 1,0
1f, 0f, 0f,   // 2,0
0f, 1f, 0f,   // 0,1
1f, 0f, 1f,   // 1,1
0f, 1f, 0f,   // 2,1
1f, 0f, 0f,   // 0,2
0f, 1f, 0f,   // 1,2
1f, 0f, 0f};  // 2,2

645.0f, 479.806, 178.755
683.2f, 761.443, 105.315
874.5f, 552.5060000000001, 225.93
613.2f, 628.052, 76.245
934.4f, 428.173, 149.685
628.8f, 745.995, 29.07
873.8f, 614.532, 190.035
623.8f, 748.672, 217.543
918.0f, 474.029, 190.035
'''
