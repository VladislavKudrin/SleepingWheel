import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import matplotlib.pyplot as plt



class SleepyModel:
    def __init__(self, labels, resize=(80,80)):
      self.labels = labels
      self.data = []
      self.resize = resize
        
    def load_data(self, train_folder, image_count=50):
      for label in self.labels:
        print(label)
        if train_folder.endswith('/'):
            folder = train_folder + label
        else:
            folder = train_folder + '/' + label
              
        errors = 0
        count = 0
          
        for filedir in os.listdir(folder):
          try:
            img = cv2.imread(os.path.join(folder, filedir))
            img = cv2.resize(img, self.resize)
            self.data.append([img, self.labels.index(label)])
          except e:
            errors += 1
            print('Error: ' + e + '. Error count: '+ str(errors))
            
          count += 1
          if count%100 == 0:
            print(str(count) + 'loaded')
          if count%image_count == 0:
            break
                    
    def define_train_test_data(self):
        if self.data:
          X = []
          y = []

          for features, label in self.data:
            X.append(features)
            y.append(label)

          X = np.array(X).reshape(-1, self.resize[0], self.resize[1], 3) #-1: data Number; 80,80: size; 3: RGB
          self.y = np.array(y)
          self.X = X/255.0 #scale data

          self.X_train, self.X_test, self.y_train, self.y_test = \
                  train_test_split(self.X, self.y, stratify=y)
        else:
          print('You have no data to define.. Run "load_data" first. ')
        
    def configure_model(self, filters=32, 
                        kernel_size=(3,3), 
                        pool_size=(2,2), 
                        dense_nodes=256, 
                        decrease_dense=True,
                        loss='binary_crossentropy',
                        optimizer='adam',
                        activation='sigmoid',
                        drop=0.3,
                        metrics=keras.metrics.AUC(curve = 'PR')):
      self.model = Sequential()
      #First set of convolutional Layers
      self.model.add(Conv2D(
                      filters = filters,
                      kernel_size=kernel_size,
                      activation = 'relu',
                      input_shape = (self.resize[0], self.resize[1], 3)
                      ))
      self.model.add(Conv2D(
                      filters = filters,
                      kernel_size = kernel_size,
                      activation = 'relu'
                      ))
      self.model.add(Conv2D(
                      filters = filters,
                      kernel_size = kernel_size,
                      activation = 'relu'
                      ))
      
      #Pooling
      self.model.add(MaxPooling2D(pool_size = pool_size))
      
      #Second set of convolutional Layers
      self.model.add(Conv2D(
                      filters = filters,
                      kernel_size = kernel_size,
                      activation = 'relu'
                      ))
      self.model.add(Conv2D(
                      filters = filters,
                      kernel_size = kernel_size,
                      activation = 'relu'
                      ))
      
      #Pooling
      self.model.add(MaxPooling2D(pool_size = pool_size))
      
      #Flatten
      self.model.add(Flatten())
      
      #First set of Dense and Dropout Layers
      self.model.add(Dense(dense_nodes, activation='relu'))
      self.model.add(Dropout(drop))
      
      #First set of Dense and Dropout Layers
      dense_nodes = dense_nodes/2 if decrease_dense else dense_nodes
      self.model.add(Dense(dense_nodes, activation='relu'))
      self.model.add(Dropout(drop))
      
      #First set of Dense and Dropout Layers
      dense_nodes = dense_nodes/2 if decrease_dense else dense_nodes
      self.model.add(Dense(dense_nodes, activation='relu'))
      self.model.add(Dropout(drop))
      
      #Output Layes
      self.model.add(Dense(1, activation=activation))
      
      #Compile Model
      self.model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
        
    def fit(self, batch_size=32, epochs=800):
      self.batch_size = batch_size
      self.epochs = epochs
      if self.model is not None and self.X_train is not None:
        self.history = self.model.fit(self.X_train, self.y_train, 
                          batch_size = batch_size, 
                          validation_data = (self.X_test, self.y_test),
                          epochs = epochs)
        self.history = self.history.history
        self.trained = True
      else:
        print('''You have not defined traine data OR model.
                Please, use "configure_model" and "define_train_test_data"
                to do so.''')
    
    def evaluate_model(self):
      if self.trained:
        self.results = self.model.evaluate(self.X_test, self.y_test, verbose=1)
      else:
        print('You have not trained your model. Please, use "fit" to do so.')
    
    def save_model(self, path, name):
      self.evaluate_model()
      loss, auc = self.results
      loss = str(round(loss, 2)).split('.')[1]
      auc = str(round(auc, 2)).split('.')[1]
      name = str(loss) + '_' + str(auc) + '_' + name
      full_path = (path + '/' + 
                      str(self.batch_size) + 
                      '_' + str(self.epochs) + 
                      '/')
      self.model.save(full_path + name + '.h5')
      np.save(full_path + name + '_history.npy', self.history)
      print(name + ' is saved!')

    def load_model(self, path, name):
      try:
        self.model = keras.models.load_model(path + name + '.h5')
        self.history = np.load(path + name + 
                               '_history.npy',
                               allow_pickle='TRUE').item()
        self.trained = True
        print('Loaded')
      except Exception as e:
        print('Not loaded', e)

    def plot_model(self, metrics=None, parameter='acc'):
      if not metrics: metrics = ['auc', 'val_auc_', 'loss', 'val_loss']

      for key in self.history.keys():
        if parameter == 'acc':
          if key.startswith(metrics[0]): first_key = key
          if key.startswith(metrics[1]): second_key = key
        elif parameter == 'loss':
          if key.startswith(metrics[2]): first_key = key
          if key.startswith(metrics[3]): second_key = key
        else:
          return print('''You can only get loss or accuracy 
          from model. Put "loss" or "acc" in function''')

      plt.plot(self.history[first_key])
      plt.plot(self.history[second_key])
      plt.title('model accuracy')
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.show()



#Load data
labels = ['Open', 'Closed'] #Open,Closed for eyes. mouth_open, mouth_closed for mouth
model = SleepyModel(labels)
model.load_data(train_folder, 800)
#Define Train and Test Data
model.define_train_test_data()