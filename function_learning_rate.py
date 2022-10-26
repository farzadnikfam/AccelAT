"""Learning Rate Choosing"""

import math
import function_global


# list of learning rate types:
# 0 - constant
# 1 - piecewiese
# 2 - linear decay
# 3 - exponential decay
# 4 - 1 cycle
# 5 - accelat

"""
in = initial
fi = final
pa = partial
cy = cycle
tot = total
ft = first
sd = second
ep = epochs
pw = power
sum = summary
"""

def learning_rate (lr_type=0, in_lr=0.01, pa_lr=0.001, fi_lr=0.0001, tot_ep=1000, ft_ep=None, sd_ep=None, pw=1, cy_lr=False):
   "Choose the learning rate to use"
   if ft_ep == None:
      ft_ep=tot_ep/3
   if sd_ep == None:
      sd_ep=tot_ep/3*2

   def constant(epoch):
      "Constant learning rate"
      lr_value = in_lr
      return lr_value

   def piecewise(epoch):
      "Piecewise learning rate"
      lr_value = in_lr
      if epoch > ft_ep:
         lr_value = pa_lr
      if epoch > sd_ep:
         lr_value = fi_lr
      return lr_value

   def linear_decay(epoch):
      "Linear decay learning rate"
      if cy_lr == False:
         step = min(epoch, ft_ep)
         lr_value = (in_lr - fi_lr) * (1 - step / ft_ep) ** (pw) + fi_lr
         return lr_value
      else:
         step = ft_ep * math.ceil((epoch+1) / ft_ep)
         lr_value = (in_lr - fi_lr) * (1 - (epoch+1) / step) ** (pw) + fi_lr
         return lr_value

   def exponential_decay(epoch):
      "Exponential decay learning rate"
      lr_value = in_lr * pw ** (epoch / ft_ep)  #the ** operator is the exponential
      return lr_value

   def one_cycle(epoch):
      "1 cycle learning rate"
      lr_value = (in_lr - in_lr * 10) * (1 - epoch / ft_ep) + in_lr * 10
      if epoch > ft_ep:
         lr_value = (in_lr * 10 - in_lr) * (1 - (epoch - ft_ep) / (sd_ep - ft_ep)) + in_lr
      if epoch > sd_ep:
         lr_value = (in_lr - in_lr * 0.01) * (1 - (epoch - sd_ep) / (tot_ep - sd_ep)) + in_lr * 0.01
      return lr_value

   def accelat(epoch):
      "AccelAT learning rate"
      return function_global.lr_value

   learning_rate_list=[constant, piecewise, linear_decay, exponential_decay, one_cycle, accelat] #list of the learning rate functions
   return learning_rate_list[lr_type]
