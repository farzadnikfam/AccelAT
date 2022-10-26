### AccelAT ###

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas
import tensorflow as tf
import function_global
from function_attack import attack
from function_callback import callback
from function_dataset import dataset
from function_log import log_directory
from function_log import log_zip
from function_panda_csv import panda_csv
from function_telegram_bot import telegram_bot_file
from function_telegram_bot import telegram_bot_text
from tensorflow import keras


function_global.init()
file_name=os.path.splitext("%s" %(os.path.basename(__file__)))[0] #records the file name for future use
telegram_bot_text("\n====================\n\n\n\n\n\n\n\n\n\n====================") #puts spaces on telegram bot
telegram_bot_text("====================\n====================\n=======START========\n====================\n====================") #starts on telegram bot
telegram_bot_text("*%s*" %(os.path.basename(__file__))) #prints current file name on telegram bot
print(__file__) #prints current file name and its path
print(file_name) #prints current file name without extension
print('tf version: %s'%(tf.__version__)) #prints tensorflow version

logdir = 1 #-1, 0 or 1 #to choose where to save log files, see function_log for more details
tot_ep = 50 #arbitrary choice, better between 50 and 500 #total epochs to be performed
val_freq = 1 #arbitrary choice, better between 1 and 10 #validation frequency, for example every 5 epochs
lr_type = 5 #0, 1, 2, 3, 4 or 5 #to choose the learning rate type, see function_learning_rate for more details
verbose = 2 #0, 1 or 2 #to choose how to see the training progress, it does not affect training, see keras documentation for more details, better 0 or 2
epsilons = [0.01] #arbitrary choice, better under 1 #attack perturbation value
data_set = 'cifar100' #cifar10 or cifar100 #dataset to be used for the training
data_range = '0-1' #0-1 or 0-255 #range of values for each pixel, better 0-1
datagroup = -1 #arbitrary choice or -1 to select the whole dataset #the number of images to be used for the training, better a small number for tests
batch_size = 128 #arbitrary choice, it depends on your GPU #batch size to be analized, better a power of 2
model_type = 'mobilenet' #resnet or mobilenet #model to be used for the training
attack_flag = 1 #0 or 1 #it is just a flag to decide whether to execute the attack or not, choose 1 to attack
attack_type = 'linfpgd' #linfpgd, deepfool or fgsm #to choose the attack type, see function_attack for more details
plot_images_attack = False #True or False #useful for finding the maximum epsilon indistinguishable from the human eye
base_model_trainable = True #True or False #for training the base model or not, better True

# list of values used for various learning rate type, see function_learning_rate for more details
# value_list = [constant, piecewiese, linear decay, exponential decay, 1 cycle, AccelAT]
in_lr_list = [       1e-4,       1e-4,        1e-4,        1e-8,       1e-4,       1e-3]; in_lr = in_lr_list[lr_type]
pa_lr_list = [          0,       1e-5,           0,           0,          0,        0.8]; pa_lr = pa_lr_list[lr_type]
fi_lr_list = [          0,       1e-6,        1e-5,           0,          0,       1e-6]; fi_lr = fi_lr_list[lr_type]
ft_ep_list = [          0,   tot_ep/3,    tot_ep/5,   tot_ep/10, tot_ep/5*2,          5]; ft_ep = ft_ep_list[lr_type]
sd_ep_list = [          0, tot_ep/3*2,           0,           0, tot_ep/5*4,          5]; sd_ep = sd_ep_list[lr_type]
pw_list =    [          0,          0,           1,          10,          0,      0.005]; pw = pw_list[lr_type]
cy_lr_list = [      False,      False,        True,       False,      False,      False]; cy_lr = cy_lr_list[lr_type]

function_global.lr_value = in_lr_list[lr_type] #value used for AccelAT
function_global.quiet_count = sd_ep_list[lr_type] #value used for AccelAT

if logdir == 0: #to use temporary log file
    log_directory(logdir=-1) #deletes past temporary files

logpath = log_directory(logdir, logname=file_name) #creates a fixed logpath for future files

# uncomment the following callback to use all the callbacks, see function_callback for more details
#callbacks=[callback(call_all=True, logpath=logpath, lr_type=lr_type, in_lr=in_lr, pa_lr=pa_lr, fi_lr=fi_lr, tot_ep=tot_ep, ft_ep=ft_ep, sd_ep=sd_ep, pw=pw, cy_lr=cy_lr)]
# or choose which callbacks to use, see function_callback for more details
callbacks=[callback('dp', logpath=logpath, lr_type=lr_type, in_lr=in_lr, pa_lr=pa_lr, fi_lr=fi_lr, tot_ep=tot_ep,
                      ft_ep=ft_ep, sd_ep=sd_ep, pw=pw, cy_lr=cy_lr), \
             callback('lr', logpath=logpath, lr_type=lr_type, in_lr=in_lr, pa_lr=pa_lr, fi_lr=fi_lr, tot_ep=tot_ep,
                      ft_ep=ft_ep, sd_ep=sd_ep, pw=pw, cy_lr=cy_lr), \
             callback('tf', logpath=logpath)]

file_writer = tf.summary.create_file_writer(logpath + "/train")  #to write and save more metrics
file_writer.set_as_default()

(x_train,y_train),(x_test,y_test),n_classes=dataset(data_set,data_range,datagroup) #dataset extraction
print('x_train.shape: %s'%(list(x_train.shape))); telegram_bot_text('x train shape: %s'%(list(x_train.shape)))
print('x_test.shape: %s'%(list(x_test.shape))); telegram_bot_text('x test shape: %s'%(list(x_test.shape)))

if model_type == 'resnet': #Kearas ResNet model
    base_model = keras.applications.resnet50.ResNet50(
        weights='imagenet',  #uploads ImageNet weights
        input_shape=x_train.shape[1:4],
        include_top=False)  #deletes softmax top layers

elif model_type == 'mobilenet': #Kearas MobileNet model
    base_model = tf.keras.applications.MobileNet(
        weights='imagenet',  #uploads ImageNet weights
        input_shape=x_train.shape[1:4],
        include_top=False)  #deletes softmax top layers

inputs = keras.Input(shape=x_train.shape[1:4]) #dataset dimensions (32, 32, 3) ==> 32*32 pixel RGB
x = base_model(inputs, training=False)  #sets batch norm layers in inference mode
x = keras.layers.GlobalAveragePooling2D()(x) #adds pooling layer
outputs = keras.layers.Dense(n_classes, activation="softmax")(x) #adds output layers
model = keras.Model(inputs, outputs) #creates the model

base_model.trainable = base_model_trainable
model.summary() #a summary of the model


if attack_flag == 1:
    x_train_adv = np.copy(x_train) #creates a new copy of the train dataset to be attacked
    x_test_adv = np.copy(x_test) #creates a new copy of the test dataset to be attacked
    iteration = batch_size * 10
    # adversarial images generation
    for i in range(0, x_train.shape[0], iteration):
        print(i)
        telegram_bot_text('x train iteration: %s'%(i))
        x_train_adv[i : i+iteration] = attack(model, x_train[i : i+iteration], y_train[i : i+iteration], epsilons, attack_type=attack_type, plot_images=plot_images_attack)[0].raw.numpy()
    for i in range(0, x_test.shape[0], iteration):
        print(i)
        telegram_bot_text('x test iteration: %s'%(i))
        x_test_adv[i : i+iteration] = attack(model, x_test[i : i+iteration], y_test[i : i+iteration], epsilons, attack_type=attack_type, plot_images=plot_images_attack)[0].raw.numpy()
else:
    pass

optimizer = keras.optimizers.SGD(learning_rate=in_lr_list[5]) #SGD optimizer
steps_per_epoch = x_train.shape[0] // batch_size
validation_steps = x_test.shape[0] // batch_size
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics="accuracy")
# fits the model and starts training
if attack_flag == 1:
    history = model.fit(x_train_adv, y_train, batch_size=batch_size, epochs=tot_ep, verbose=verbose, callbacks=callbacks,
                        validation_data=(x_test_adv,y_test), validation_batch_size=batch_size, validation_freq=val_freq)
else:
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=tot_ep, verbose=verbose, callbacks=callbacks)

# clean results
train_loss, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=verbose)
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=verbose)
print("Train accuracy: %f" %(train_acc)); telegram_bot_text("Train accuracy: %f" %(train_acc))
print("Train loss: %f" %(train_loss)); telegram_bot_text("Train loss: %f" %(train_loss))
print("Test accuracy: %f" %(test_acc)); telegram_bot_text("Test accuracy: %f" %(test_acc))
print("Test loss: %f" %(test_loss)); telegram_bot_text("Test loss: %f" %(test_loss))

# adversarial results
if attack_flag == 1:
    train_loss_adv, train_acc_adv = model.evaluate(x_train_adv, y_train, batch_size=batch_size, verbose=verbose)
    test_loss_adv, test_acc_adv = model.evaluate(x_test_adv, y_test, batch_size=batch_size, verbose=verbose)
    print("Train adversarial accuracy: %f" %(train_acc_adv)); telegram_bot_text("Train adversarial accuracy: %f" %(train_acc_adv))
    print("Train adversarial loss: %f" %(train_loss_adv)); telegram_bot_text("Train adversarial loss: %f" %(train_loss_adv))
    print("Test adversarial accuracy: %f" %(test_acc_adv)); telegram_bot_text("Test adversarial accuracy: %f" %(test_acc_adv))
    print("Test adversarial loss: %f" %(test_loss_adv)); telegram_bot_text("Test adversarial loss: %f" %(test_loss_adv))
else:
    pass

# closing operations and figures, zip and results generations
file_writer.close()
panda_log = panda_csv(logpath, file_name, print_dataframe=True, send_dataframe=True, print_dataimage=False, send_dataimage=True)
log_zip(logpath,file_name)
telegram_bot_file("%s" %(logpath + "/%s_train" %(file_name) + ".zip"))
try: telegram_bot_file("%s" %(logpath + "/%s_validation" %(file_name) + ".zip"))
except Exception: pass
telegram_bot_file("%s" %(os.path.basename(__file__)))

# txt log file generations
file_log = open(logpath + "/%s" %(file_name) + ".txt", "a")
file_log.write("%s\n\n" %(file_name))
file_log.write("%s\n\n" %(logpath))
file_log.write("logdir: %s\n" %(logdir))
file_log.write("tot_ep: %s\n" %(tot_ep))
file_log.write("val_freq: %s\n" %(val_freq))
file_log.write("lr_type: %s\n" %(lr_type))
file_log.write("verbose: %s\n" %(verbose))
file_log.write("epsilons: %s\n" %(epsilons))
file_log.write("data_set: %s\n" %(data_set))
file_log.write("data_range: %s\n" %(data_range))
file_log.write("datagroup: %s\n" %(datagroup))
file_log.write("batch_size: %s\n" %(batch_size))
file_log.write("model_type: %s\n" %(model_type))
file_log.write("attack_flag: %s\n" %(attack_flag))
file_log.write("attack_type: %s\n" %(attack_type))
file_log.write("plot_images_attack: %s\n" %(plot_images_attack))
file_log.write("base_model_trainable: %s\n\n\n" %(base_model_trainable))
file_log.write("type: train - test - train adv - test adv\n\n")
file_log.write("accuracy: {:2.2%} - {:2.2%} - {:2.2%} - {:2.2%}\n".format(train_acc,test_acc,train_acc_adv,test_acc_adv))
file_log.write("loss: {:.3} - {:.3} - {:.3} - {:.3}\n".format(train_loss,test_loss,train_loss_adv,test_loss_adv))
file_log.close()
telegram_bot_file("%s" %(logpath + "/%s" %(file_name) + ".txt"))

# terminates telegram bot session
telegram_bot_text("*%s*" %(os.path.basename(__file__)))
telegram_bot_text("\n========END=========")



# to plot results, useless if you use the telegram bot!
def plot_losses(history):
  pandas.DataFrame(history.history).plot(figsize=(10,8))
  plt.grid(True)
  plt.gca().set_ylim(0,1)
  plt.show()
#plot_losses(history) #uncomment to plot results
