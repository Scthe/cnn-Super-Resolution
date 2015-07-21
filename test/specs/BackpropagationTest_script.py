#helper script to generate expected delta values
#for BackpropagationTest

inputs = [[
  -0.083,   0.075,   -0.058,   -0.068,  -0.013,
   0.169,   0.181,    0.136,   -0.165,   0.159,
  -0.112,   0.003,   -0.123,   -0.102,   0.242,
   0.406,  -0.442,   -0.627,    0.376,   0.680,
   0.121,  -0.103,    0.106,   -0.036,   0.052],
 [-0.064,  -0.055,   -0.138,   -0.144,   0.176,
   0.049,  -0.051,   -0.062,   -0.176,  -0.060,
   0.228,  -0.138,   -0.027,   -0.061,  -0.069,
   0.419,   0.685,   -0.489,    0.563,  -0.371,
  -0.075,   0.031,    0.033,   -0.052,  -0.035]]

deltas = [0.122, 0.083, 0.064,  # row 1, col 1
          0.057, 0.075, 0.055,  # row 1, col 2
          0.025, 0.058, 0.138,  # row 1, col 3

          0.170, 0.068, 0.144,  # row 2, col 1
          0.121, 0.013, 0.176,  # row 2, col 2
          0.065, 0.169, 0.049,  # row 2, col 3

          0.003, 0.181, 0.051,  # row 3, col 1
          0.021, 0.136, 0.062,  # row 3, col 2
          0.066, 0.165, 0.176]  # row 3, col 3

f=3
out = 3,3
inn = 5,5
n_curr = 3
n_prev = 2

w = [1.5] * 54 # we will add algo result to this

def kernel(x,y):
    for n in range(n_curr):
        delta_idx = ((y * out[0]) + x) * n_curr + n
        delta = deltas[delta_idx]
        for a in range(f):
            for b in range(f):
                for k in range(n_prev):
                    p = x+b, y+a
                    val = inputs[k][p[1] * inn[0] + p[0]]
                    idx = ((a * f) + b) *n_curr*n_prev + k * n_curr + n
                    w[idx] += val * delta
                    # w[idx] += val
                    # w[idx] += delta

for y in range(out[1]):
    for x in range(out[0]):
        kernel(x,y)
# print('\n'.join(["[{}]\t{:>6.3}".format(i,x) for i,x in enumerate(w)]))
# print('\n'.join(["[{}]\t{:>6.3}".format(i,x) for i,x in enumerate(w) if i%3==0]))
# print('\n'.join(["[{}]\t{:>6}".format(i,x) for i,x in enumerate(w)]))
for i in range(9):
    xs = w[i*6:(i+1)*6]
    print(', '.join(["{:>7.5}".format(x) for i,x in enumerate(xs)]))


print('\n\nbias:')
bias_res = [
    sum([x for i,x in enumerate(deltas) if i%3==0]),
    sum([x for i,x in enumerate(deltas) if i%3==1]),
    sum([x for i,x in enumerate(deltas) if i%3==2])]
print(', '.join(["{:>6.3}".format(x) for i,x in enumerate(bias_res)]))


'''
ONLY INPUT:

PY:
[0]      0.188
[3]     -0.258
[6]     -0.121
[9]     -0.852
[12]     0.008
[15]    -0.561
[18]    -0.409
[21]     0.614
[24]    -0.763
[27]     0.244
[30]     0.576
[33]    -0.752
[36]    -0.771
[39]     0.667
[42]    -0.948
[45]     0.545
[48]     0.568
[51]    -0.508

GPU:
b[0]    0.188
b[3]   -0.258
b[6]   -0.121
b[9]   -0.852
b[14]   0.008
b[17]  -0.561

b[18]  -0.409
b[23]   0.614
b[24]  -0.763
b[29]   0.244
b[32]   0.576
b[35]  -0.752

b[38]  -0.771
b[41]   0.667
b[44]  -0.948
b[47]   0.545
b[50]   0.568
b[53]  -0.508
'''
