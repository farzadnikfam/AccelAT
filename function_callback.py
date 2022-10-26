"""Callback"""

import tensorflow as tf
import numpy as np
import function_global
from function_log import log_directory
from function_telegram_bot import telegram_bot_text
from function_learning_rate import learning_rate
from tensorflow import keras


def callback(call_type=None, logpath=log_directory(logdir=0), call_all=False, reach_acc=0.90, lr_type=0, in_lr=0.01, pa_lr=0.001, fi_lr=0.0001,
             tot_ep=1000, ft_ep=None, sd_ep=None, pw=1, cy_lr=False):
   "Choose callback type"
   if call_all == True:
      return callback('dp', logpath=logpath, lr_type=lr_type, in_lr=in_lr, pa_lr=pa_lr, fi_lr=fi_lr, tot_ep=tot_ep,
                      ft_ep=ft_ep, sd_ep=sd_ep, pw=pw, cy_lr=cy_lr), \
             callback('lr', logpath=logpath, lr_type=lr_type, in_lr=in_lr, pa_lr=pa_lr, fi_lr=fi_lr, tot_ep=tot_ep,
                      ft_ep=ft_ep, sd_ep=sd_ep, pw=pw, cy_lr=cy_lr), \
             callback('ra', logpath=logpath, reach_acc = reach_acc), \
             callback('tf', logpath=logpath)

   elif call_type == 'dp' and call_all == False:
      # prints results on telegram bot and saves the learning rate
      class data_print(tf.keras.callbacks.Callback):
         def on_epoch_end(self, epoch, logs={}):
            lr_value = keras.backend.eval(self.model.optimizer.lr) #imports current learning rate value
            if len(logs) == 4:
                telegram_bot_text("*{}* -- lr: {:.3}\n==> loss: {:.3} -- acc: {:2.2%}\n==> val loss: {:.3} -- "
                              "val acc: {:2.2%}".format(epoch, lr_value, logs.get('loss'), logs.get('accuracy'),
                               logs.get('val_loss'), logs.get('val_accuracy')))
            elif len(logs) == 2:
                telegram_bot_text("*{}* -- lr: {:.3}\n==> loss: {:.3} -- acc: {:2.2%}"
                              .format(epoch, lr_value, logs.get('loss'), logs.get('accuracy')))
            else:
                pass
            tf.summary.scalar('learning_rate', data=lr_value, step=epoch) #saves the learning rate value
      return data_print()

   elif call_type == 'lr' and call_all == False:
      # selects the learning rate type
      class accelat_function(tf.keras.callbacks.Callback): #AccelAT algorithm
         def on_epoch_end(self, epoch, logs={}):
            function_global.acc_list.append(logs.get('val_accuracy')) #evaluation on validation accuracy gradient
            function_global.lr_value = keras.backend.eval(self.model.optimizer.lr) #import current learning rate value
            acc_list_len = np.size(function_global.acc_list) #saves accuracy gradient list size
            acc_mean = np.mean(function_global.acc_list[acc_list_len - ft_ep:acc_list_len]) #accuracy gradient mean value
            a_list = function_global.acc_mean_list #accuracy gradient mean list
            quiet = function_global.quiet_count #counter for quiet time after a learning reduction
            quiet = quiet + 1
            if (np.size(a_list) > ft_ep) and (quiet > sd_ep) and ((acc_mean - a_list[np.size(a_list) - 1]) < pw):
              function_global.lr_value = function_global.lr_value * pa_lr #learning rate reduction
              quiet = 1 #restarting quiet time
            if function_global.lr_value < fi_lr:
              function_global.lr_value = fi_lr #minimum learning rate
            function_global.acc_mean_list.append(acc_mean)
            function_global.quiet_count = quiet
      lr = learning_rate(lr_type=lr_type, in_lr=in_lr, pa_lr=pa_lr, fi_lr=fi_lr, tot_ep=tot_ep, ft_ep=ft_ep,
                             sd_ep=sd_ep, pw=pw, cy_lr=cy_lr) #call the learning rate function
      if 'lr' == 5:
        return accelat_function(), tf.keras.callbacks.LearningRateScheduler(lr)  #sends values to the optimizer
      else:
        return tf.keras.callbacks.LearningRateScheduler(lr)  #sends value to the optimizer

   elif call_type == 'ra' and call_all == False:
      # stops training if a certain accuracy is reached
      class reach_accuracy(tf.keras.callbacks.Callback):
         def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > reach_acc):
               print("\nReached the {:2.2%} accuracy so training stopped!".format(reach_acc))
               telegram_bot_text("Reached the {:2.2%} accuracy so training stopped!".format(reach_acc))
               self.model.stop_training = True
      return reach_accuracy()

   elif call_type == 'tf' and call_all == False:
      # saves values in log file
      return tf.keras.callbacks.TensorBoard(log_dir=logpath)

   else:
      print("select a callback:\ndp\nlr\nra\ntf")
