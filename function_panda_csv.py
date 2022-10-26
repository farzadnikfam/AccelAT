"""Panda Dataframe and CSV Creation"""

import tensorflow as tf
import glob
import os
import pandas as pd
from function_telegram_bot import telegram_bot_file
from function_telegram_bot import telegram_bot_image
from tensorflow.python.framework import tensor_util


def panda_csv(logpath, file_name="coccobello", print_dataframe=True, send_dataframe=True, print_dataimage=False, send_dataimage=True):
    "create a panda dataframe and a CSV file"
    event_paths = glob.glob(os.path.join(logpath + '/train', "event*"))  #reads train data logs
    event_paths_val = glob.glob(os.path.join(logpath + '/validation', "event*")) #reads validation data logs
    panda_log = pd.DataFrame(columns=['learning_rate', 'accuracy', 'loss', 'val_accuracy', 'val_loss']) #creates panda dataframe
    for i in range(len(list(event_paths))): #reads all train logs and records their data
        for event in tf.compat.v1.train.summary_iterator(event_paths[i]):
            for value in event.summary.value:
                if value.tag == 'epoch_accuracy':
                    panda_log.loc[event.step,'accuracy'] = value.simple_value
                if value.tag == 'epoch_loss':
                    panda_log.loc[event.step,'loss'] = value.simple_value
                if value.tag == 'learning_rate':
                    panda_log.loc[event.step,'learning_rate'] = tensor_util.MakeNdarray(value.tensor)
    for i in range(len(list(event_paths_val))): #reads all validation logs and records their data
        for event in tf.compat.v1.train.summary_iterator(event_paths_val[i]):
            for value in event.summary.value:
                if value.tag == 'epoch_accuracy':
                    panda_log.loc[event.step,'val_accuracy'] = value.simple_value
                if value.tag == 'epoch_loss':
                    panda_log.loc[event.step,'val_loss'] = value.simple_value

    file_path = logpath + "/%s" %(file_name) #creates a path to record results
    panda_log_float = panda_log.astype(float)
    panda_log.to_csv(file_path + ".csv", index=None)

    # printing all data in figures
    panda_log_acc = panda_log_float.plot(y='accuracy', figsize=(30, 30), grid=True, fontsize=50, linewidth=5, legend=False)
    panda_log_loss = panda_log_float.plot(y='loss', figsize=(30, 30), grid=True, fontsize=50, linewidth=5, legend=False)
    panda_log_lr = panda_log_float.plot(y='learning_rate', figsize=(30, 30), grid=True, fontsize=50, linewidth=5, legend=False)
    panda_log_valacc = panda_log_float.interpolate(method='linear').plot(y='val_accuracy', figsize=(30, 30), grid=True, fontsize=50, linewidth=5, legend=False) #interpolation to pass the NaN problem if the validation frequency is not 1
    panda_log_valloss = panda_log_float.interpolate(method='linear').plot(y='val_loss', figsize=(30, 30), grid=True, fontsize=50, linewidth=5, legend=False) #interpolation to pass the NaN problem if the validation frequency is not 1

    # saving all figures
    panda_log_acc.get_figure().savefig(file_path + "_acc.png")
    panda_log_loss.get_figure().savefig(file_path + "_loss.png")
    panda_log_lr.get_figure().savefig(file_path + "_lr.png")
    panda_log_valacc.get_figure().savefig(file_path + "_valacc.png")
    panda_log_valloss.get_figure().savefig(file_path + "_valloss.png")

    # shows all figures
    if print_dataimage == True:
        panda_log_acc.figure.show()
        panda_log_loss.figure.show()
        panda_log_lr.figure.show()
        panda_log_valacc.figure.show()
        panda_log_valloss.figure.show()

    # sends all figures to telegram bot
    if send_dataimage == True:
        telegram_bot_image(file_path + "_acc.png", "accuracy")
        telegram_bot_image(file_path + "_loss.png", "loss")
        telegram_bot_image(file_path + "_lr.png", "learning rate")
        telegram_bot_image(file_path + "_valacc.png", "validation accuracy")
        telegram_bot_image(file_path + "_valloss.png", "validation loss")

    if print_dataframe == True: #prints the dataframe
        print(panda_log)
    if send_dataframe == True: #sends dataframe to telegram
        telegram_bot_file("%s" % (file_path+ ".csv"))
    return panda_log
