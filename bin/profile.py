import re
import time
import subprocess

epochs = 100
seconds_per_epoch = 0.3
cmd = 'bin\\cnn.exe train dry -c data\config.json --epochs {0:} -i data\\train_samples'.format(epochs)

kernel_profile_regex = "Kernel '.*/(.*?]).*?([\-e.\d]+)ns.*?([\-e.\d]+)s"
def get_kernel_profiling_info(out):
  out = out.decode('UTF-8')
  rr = re.findall(kernel_profile_regex, out)
  l = [(x[0], int(x[1]), float(x[2])) for x in rr]
  l = sorted(l, key=lambda x: x[2])
  ts = 0.0
  for _,_,t in l:
      ts += t
  return l, ts


if __name__ == '__main__':
  import sys

  kernel_mode = 'kernel' in sys.argv

  cmd_ = cmd.split(' ')
  if kernel_mode:
    cmd_.append('profile')
  print('Command to execute:')
  print('\'' + (' '.join(cmd_)) + '\'')

  est_time = epochs * seconds_per_epoch
  print('Will do {0:} epochs'.format( epochs))
  print('Estimated required time: {}s = {} min'.format(est_time, est_time//60))

  start = time.time()
  proc = subprocess.Popen(cmd_, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  outs, errs = proc.communicate()
  if proc.returncode is not 0:
        print('---- FAIL ----')
        exit()
  end = time.time()
  dt = end - start

  print("Execution time: {:.3f}s = {:.2f}min ({:.5f} s/epoch)".format(dt, dt/60, dt/epochs))

  if kernel_mode:
    kps, kernel_time = get_kernel_profiling_info(outs)
    for name,ns,s in kps:
      name = name.replace('-D ', '').replace('\'', '').replace('[--]','')
      print("{0:7.4f}s ({1:5.2f}%)- {2:.65}".format(s, s*100/kernel_time, name))
    print( "Time spend in kernel: {:f}s".format(kernel_time))
    print("Percent of time spend in kernel: {:.4f}%".format(kernel_time*100/dt))
