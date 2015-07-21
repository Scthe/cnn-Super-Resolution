import json
import argparse
from os.path import join
from pprint import pprint
from PIL import Image, ImageDraw, ImageColor

# cfg_file = 'config_f.json'
# scale = None
per_weight_cell_padding = 2

def layer_data(cfg, layer_id):
  'returns (f,k,n)'
  read = lambda prop: int(cfg[prop])
  if layer_id == 1:
    return read('f1'),1,read('n1')
  elif layer_id == 2:
    return read('f2'),read('n1'),read('n2')
  elif layer_id == 3:
    return read('f3'),read('n2'),1
  else:
    raise Exception("Only 1,2,3 are valid layers")

def idx(layer, dy,dx,n,k):
  f, layer_k, layer_n = layer
  # print('layer: ',layer, ' a,b:',dy,dx, '   n,k: ',n,k)
  return dy * layer_n * layer_k * f + \
         dx * layer_n * layer_k + \
         k  * layer_n + \
         n

def filter_weights(weights, layer, curr_n,curr_k):
  f = layer[0]
  filter_wx = [0]*(f*f)
  for dy in range(f):
    for dx in range(f):
      w_idx = idx(layer, dy,dx,curr_n,curr_k)
      # print(dy*f+dx,'/',len(filter_wx))
      # print(w_idx,'/',len(weights))
      filter_wx[dy*f+dx] = weights[w_idx]
  min_w, max_w = min(filter_wx), max(filter_wx)
  # norm_w = max_w + min_w
  # print('min_w: {}, max_w: {}'.format(min_w, max_w))

  a,b=-999,999
  for dy in range(f):
    for dx in range(f):
      w = filter_wx[dy*f+dx]
      # w = (w-min_w) / (max_w + min_w)
      w = (w-min_w) / (max_w - min_w) if max_w != min_w else 0.5
      # w = min(1,max(0,w))
      yield dy,dx,w
      a=max(a,w)
      b=min(b,w)
  # print('{:8}\t: {:8} \t-> {:8} : {}'.format(min_w, max_w, b, a))

def visualize(cfg, params, scale, layer_id, out_path):
  print('--- layer ', layer_id, ' ---')
  weights = params['layer' + str(layer_id)]['weights']
  min_w, max_w = min(weights), max(weights)
  print('min_w: {}, max_w: {}'.format(min_w, max_w))

  f, l_k, l_n = layer = layer_data(cfg, layer_id)
  cell_size = f * scale + 2 * per_weight_cell_padding
  print(layer)
  if f == 1:
    print('f==1, drawing weights would not show anything')
    return

  rows = int((l_n*l_k)**0.5)
  cells_in_row = int((l_n*l_k+rows-1) / rows)
  print('columns: ', cells_in_row, 'rows: ', rows)
  # size = cell_size * l_n, cell_size * l_k
  size = cell_size * cells_in_row, cell_size * rows

  img = Image.new('RGB', size, color='#000000')
  filter_img = Image.new('RGB', (f*scale,f*scale))
  filter_draw = ImageDraw.Draw(filter_img)

  for n in range(l_n):
    for k in range(l_k):
      idx = n * l_k + k
      row, col = idx // cells_in_row, idx % cells_in_row
      # print(idx, '\t-> ',row,', ',col)
      pos = int(cell_size * col + per_weight_cell_padding), \
            int(cell_size * row + per_weight_cell_padding)
      for (dy,dx,val) in filter_weights(weights, layer, n,k):
        v = int(val*255)
        col = "rgb({0},{0},{0})".format(v)
        pos_ab = dx*scale, dy*scale
        pos_ab_ = pos_ab[0] + scale - 1, \
                  pos_ab[1] + scale - 1
        filter_draw.rectangle((pos_ab, pos_ab_), fill=col)
      img.paste(filter_img, pos)

  img.save(out_path, "PNG")

if __name__ == '__main__':
  help_text = 'Draw weights. Usage: ' + \
              '"weights_visualize.py -o data -s 10 data\config_f.json"'

  parser = argparse.ArgumentParser(description=help_text)
  parser.add_argument('config', help='config file to analize' )
  parser.add_argument('--parameters-file', '-p', required=False, help='parameters file holding all weights and biases')
  parser.add_argument('--out-dir', '-o', required=False, default='', help='where to store result images')
  parser.add_argument('--scale', '-s', required=False, default=10, type=int, help='scale factor - cause sometimes 10x10 image is too small')
  args = parser.parse_args()

  print(args.config)
  print(args.parameters_file)
  print(args.out_dir)
  print(args.scale)

  with open(args.config) as data_file:
    cfg = json.load(data_file)
  # pprint(cfg)

  if args.parameters_file:
    par_file = args.parameters_file
  elif 'parameters_file' in cfg:
    par_file = cfg['parameters_file']
  else:
    raise Exception('Either write parameter file path to config or provide as parametr')
  print('Parameter file: \'',par_file,'\'')
  with open(par_file) as data_file:
    params = json.load(data_file)
  # pprint(params)

  visualize(cfg, params, args.scale, 1, join(args.out_dir, 'weights1.png'))
  visualize(cfg, params, args.scale, 2, join(args.out_dir, 'weights2.png'))
  visualize(cfg, params, args.scale, 3, join(args.out_dir, 'weights3.png'))
