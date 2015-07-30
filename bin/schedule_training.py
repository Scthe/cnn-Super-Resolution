import os
import time
import shutil
import argparse
import subprocess
from time import gmtime, strftime

'''
typical .bat file:

make build
if %errorlevel%==0 (
  bin\cnn.exe train -c data\config.json --epochs 100 -i data\train_samples -o data\parameters.json
)
'''

epochs_per_iteration = 500
pars_file = 'data\\parameters.json'

seconds_per_epoch = 0.8 # derived experimentally
#seconds_per_epoch = 0.28

cmd = 'bin\\cnn.exe train -c data\config.json --epochs {} -i data\\train_samples'.format(epochs_per_iteration)


def get_dst_file_path():
  #strftime("%Y-%m-%d %H:%M:%S")
  tt = strftime("%Y-%m-%d--%H-%M-%S")
  log_folder = lambda s: os.path.join('logs', s)
  return log_folder('log_{}.txt'.format(tt)), \
         log_folder('parameters_{}.json'.format(tt)), \
         tt


seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}

def convert_to_seconds(s):
  return int(s[:-1]) * seconds_per_unit[s[-1]]

if __name__ == '__main__':
  help_text = 'Start training with either duration or #epochs'
  parser = argparse.ArgumentParser(description=help_text)
  action = parser.add_mutually_exclusive_group(required=True)
  action.add_argument('--duration', '-d', help='Duration, provided as: X[s|m|h|d|w] (s=seconds, m=minutes, h=hours, d=days, w=week)')
  action.add_argument('--epochs',   '-e', type=int, help='Number of epochs')
  parser.add_argument('--dry', action='store_true', required=False, help='Do not output any files')

  args = parser.parse_args()
  if args.duration:
    time_in_s = convert_to_seconds(args.duration)
    total_epochs = int(time_in_s / seconds_per_epoch)
  else:
    total_epochs = args.epochs
  total_epochs = max(total_epochs, 1)

  cmd_ = cmd.split(' ')
  if args.dry:
    cmd_.append('dry')
  else:
    cmd_.append('-o')
    cmd_.append(pars_file)
  print('Command to execute:')
  print('\'' + (' '.join(cmd_)) + '\'')

  start = time.time()
  iters = total_epochs // epochs_per_iteration
  total_epochs = iters * epochs_per_iteration # last iter have same #epochs as others
  print('Will do {0:} iterations, {1:} epochs per iteration = {2:} total'.format( \
            iters, epochs_per_iteration, iters * epochs_per_iteration))
  est_time = total_epochs * seconds_per_epoch
  print('Estimated required time: {}s = {} min'.format(est_time, est_time//60))

  for i in range(iters):
    log_path, tmp_params_path, stamp = get_dst_file_path()
    total_epochs_left = (iters - i) * epochs_per_iteration
    print('\n---- {0:} - {1:} (time left: {2:d}min)----'.format(i+1, stamp, int(total_epochs_left*seconds_per_epoch)//60))

    # execute training
    with open(log_path, "w") as tmp_log:
      ret_code = subprocess.call(cmd_, stdout=tmp_log, stderr=subprocess.STDOUT)
      print('return code: '+str(ret_code))
      if ret_code is not 0:
        print('---- FAIL ----')
        exit()

    # backup results
    if not args.dry:
      print('saving sub results to: \'' + tmp_params_path + '\'')
      shutil.copy2(pars_file, tmp_params_path)

  end = time.time()
  dt = end - start
  print("Execution time: {:.3}s = {:.2}min ({:.5} s/epoch)".format(dt, dt/60, dt/total_epochs))
