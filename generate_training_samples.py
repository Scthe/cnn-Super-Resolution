import os
from os.path import isfile, join
from random import randint

import argparse
from PIL import Image

img_id = 0
created_files =[]

def list_files(dir_path):
  return [f for f in os.listdir(dir_path) if isfile(join(dir_path,f)) ]

def process_img(in_dir,out_dir,file_name, out_size,small_scale):
  global img_id, created_files
  in_path = join(in_dir, file_name)
  large_path = join(out_dir, 'sample_{0:}_{1:}.jpg'.format(img_id, "large"))
  small_path = join(out_dir, 'sample_{0:}_{1:}.jpg'.format(img_id, "small"))
  # print( in_path)
  # print(large_path)
  # print(small_path)
  img_id += 1

  im = Image.open(in_path)
  if im.width < out_size or im.height < out_size:
    raise Exception('Image \'{0:}\' is smaller then requested out-size'.format(file_name))

  crop_upper_left = randint(0,im.width-out_size), randint(0,im.height-out_size)
  large = im.crop((crop_upper_left[0],\
                   crop_upper_left[1], \
                   crop_upper_left[0] + out_size,\
                   crop_upper_left[1] + out_size))
  # size = out_size, out_size
  # im.resize(size, Image.ANTIALIAS)
  large.save(large_path, "JPEG")

  small_size = int(out_size/small_scale)
  small1 = large.resize((small_size,small_size), Image.ANTIALIAS)
  small2 = small1.resize((out_size,out_size), Image.ANTIALIAS)
  small2.save(small_path, "JPEG")

  created_files.append((large_path, small_path))



if __name__ == '__main__':
  help_text = 'Mass resize images. Example usage: ' + \
              '" python generate_training_samples.py -i data\\org -o data\\train_samples -s 550 -d 5"'

  parser = argparse.ArgumentParser(description=help_text)
  parser.add_argument('--in-dir',   '-i',required=True, help='input directory' )
  parser.add_argument('--out-dir',  '-o',required=True, help='output directory' )
  parser.add_argument('--out-size', '-s',required=True, help='size of output images', type=int)
  parser.add_argument('--degrade-factor', '-d', help='scale factor when producing smaller image', type=float, default=2)
  args = parser.parse_args()

  in_files = list_files(args.in_dir)
  # print('Found following files in \''+args.in_dir+'\': ')
  # print(in_files)

  os.makedirs(args.out_dir, exist_ok=True)
  for f in in_files:
    try:
      process_img(args.in_dir,args.out_dir,f, args.out_size, args.degrade_factor)
    except IOError:
      print("cannot create train samples for '{0:}'".format(f))
    except Exception as e:
      print(str(e))

  if not created_files:
    print('No files were created')
  else:
    print('created {0:} files'.format(len(created_files)))
    # print('\n'.join([item.replace("\\","\\\\") for sublist in created_files for item in sublist]))
