"""Attack Selection"""

import tensorflow as tf
import numpy as np
import foolbox as fb
import eagerpy as ep
import matplotlib.pyplot as plt


def attack(model, x_train, y_train, epsilons, attack_type='linfpgd', datagroup=-1, plot_images=False):
  "Attack selection"

  if datagroup == -1:
    datagroup = y_train.shape[0]  #uses the entire dataset
  bounds = (0, 1)
  # extraction from foolbox
  fmodel = fb.TensorFlowModel(model, bounds=bounds)
  x_train = ep.astensors(tf.convert_to_tensor(x_train[0:datagroup], dtype=tf.float32))[0]
  y_train = ep.astensors(tf.convert_to_tensor(y_train[0:datagroup,0], dtype=tf.int32))[0]
  if plot_images == True: #to use for debugging
    clean_acc = fb.accuracy(fmodel, x_train, y_train)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

  # selects the attack
  if attack_type == 'linfpgd':
    attacktype = fb.attacks.LinfPGD()
  elif attack_type == 'deepfool':
    attacktype = fb.attacks.LinfDeepFoolAttack()
  elif attack_type == 'fgsm':
    attacktype = fb.attacks.LinfFastGradientAttack()
  else:
    print("no attack choosen, default ==> linfpgd")
    print("select an attack:\nlinfpgd\ndeepfool\nfgsm")
    attacktype = fb.attacks.LinfPGD() #LinfPGD default attack type

  # applies the attack
  if plot_images == True:
    attack_raw, attack_clip, attack_success = attacktype(fmodel, x_train, y_train, epsilons=epsilons)
    robust_accuracy = 1 - attack_success.float32().mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
      print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
  else:
    _, attack_clip, _ = attacktype(fmodel, x_train, y_train, epsilons=epsilons)

  # for plotting adversarial images
  if plot_images == True:
    plt.plot(epsilons, robust_accuracy.numpy()) #plot 1
    fb.plot.images(x_train.raw[0:5]) #plot 2
    fb.plot.images(attack_clip[0].raw[0:5]) #plot 3
    fb.plot.images(attack_clip[len(epsilons)-1].raw[0:5]) #plot 4
    ima_paragon = np.ndarray(shape=(len(epsilons), x_train.shape[1], x_train.shape[2], x_train.shape[3]), dtype=float)
    for i in range(len(epsilons)):
        ima_paragon[i] = attack_clip[i].raw[0].numpy()
    fb.plot.images(ima_paragon) #plot 5
    classifications = model.predict(ima_paragon)
    print(np.argmax(classifications,axis=1))
    print("Real label: %d" % (y_train[0].numpy()))
    plt.show()
  return attack_clip
