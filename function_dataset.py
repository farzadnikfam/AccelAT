"""Dataset Selection"""

from tensorflow import keras


def dataset(dataname, datarange='0-1', datagroup=-1):
  "selection of the dataset to use"
  #dataname: cifar10, cifar100
  #datarange: 0-1, 0-255
  #datagroup: integer value or -1 to select the entire dataset <= dataset shape[0]

  if dataname=='cifar10': #cifar10 dataset, 10 classes
    (x_train_0, y_train), (x_test_0, y_test) = keras.datasets.cifar10.load_data()
    n_classes = 10
    x_train_1 = x_train_0.astype('float32')
    x_test_1 = x_test_0.astype('float32')
    x_train_1 /= 255
    x_test_1 /= 255
    if datagroup==-1: #to select the whole dataset
      datagroup=y_train.shape[0]
    if datarange=='0-1': #normalized values betwwen 0 and 1
      return (x_train_1[0:datagroup],y_train[0:datagroup]),(x_test_1[0:datagroup],y_test[0:datagroup]),n_classes #range: 0-1 #dtype: float32
    elif datarange=='0-255': #values between 0 and 255
      return (x_train_0[0:datagroup],y_train[0:datagroup]),(x_test_0[0:datagroup],y_test[0:datagroup]),n_classes #range: 0-255  #dtype: uint8
    else:
      print("select a range:\n0-1\n0-255")

  elif dataname=='cifar100': #cifar100 dataset, 100 classes
    (x_train_0, y_train), (x_test_0, y_test) = keras.datasets.cifar100.load_data()
    n_classes = 100
    x_train_1 = x_train_0.astype('float32')
    x_test_1 = x_test_0.astype('float32')
    x_train_1 /= 255
    x_test_1 /= 255
    if datagroup==-1: #to select the whole dataset
      datagroup=y_train.shape[0]
    if datarange=='0-1': #normalized values betwwen 0 and 1
      return (x_train_1[0:datagroup],y_train[0:datagroup]),(x_test_1[0:datagroup],y_test[0:datagroup]),n_classes #range: 0-1 #dtype: float32
    elif datarange=='0-255': #values between 0 and 255
      return (x_train_0[0:datagroup],y_train[0:datagroup]),(x_test_0[0:datagroup],y_test[0:datagroup]),n_classes #range: 0-255  #dtype: uint8
    else:
      print("select a range:\n0-1\n0-255")

  else:
    print("select a dataset:\ncifar10\ncifar100")
