import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam 
import tensorflow.keras.callbacks as call
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from datetime import datetime
import librosa , librosa.display

import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, save_model

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
json_path = 'order_Eol_paper.json'
json_path_faulty='order_Eol_faulty_paper.json'
json_path_faulty_1='order_Eol_faulty_paper_1.json'
filepath = 'E:/shaefler_thesis/thesis_fraunhofer/code/eol_dataset/paper_eol_test/order_no_seg/model_saved/my_model'
BATCH_SIZE=10

def load_data(json_path):

    with open(json_path) as fp:
        data = json.load(fp)

    x = np.array(data["order"])

    z = np.array(data["mapping"])

    return x, z

def load_data_test(json_path_faulty):

    with open(json_path_faulty) as fp:
        data = json.load(fp)
    
    x = np.array(data["order"])
    
    z = np.array(data["mapping"])  
    
    return x,z


def load_data_test_1(json_path_faulty):

    with open(json_path_faulty) as fp:
        data = json.load(fp)
    
    x = np.array(data["order"])
    
    z = np.array(data["mapping"])  
    
    return x,z



def build_model(input_shape):

    input_data = layers.Input(input_shape)
    
   
    x = layers.Dense(256, activation='relu')(input_data)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    

    BN  = layers.Dense(16, activation='relu',name='latent_space')(x)
    
    
    x = layers.Dense(32, activation='relu')(BN)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    

    output = layers.Dense(input_shape)(x)
    model = Model(input_data,output)
    encoder = Model(input_data,BN)

    
    
    return model,encoder



def train():
   
    tf.random.set_seed(7) 

    x_train,y_train = load_data(json_path)
    x_test,y_test = load_data_test(json_path_faulty)
    x_test_1,y_test_1 = load_data_test_1(json_path_faulty_1)
    x_test = np.concatenate((x_test,x_test_1),axis=0)
    y_test = np.concatenate((y_test,y_test_1),axis=0)

    
 
    x_train_reshaped = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
    
    x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])

    model,encoder = build_model(input_shape=(x_train_reshaped.shape[1]))

    
    model.summary()
    encoder.summary()
    optimiser = Adam(learning_rate=0.0001)

    model.compile(optimizer=optimiser,
                  loss='mse',
                  metrics=['mean_squared_error']) #binary_crossentropy
                  
   

    ###Normalization
    scalar = MinMaxScaler()
    x_train_transformed=scalar.fit_transform(x_train_reshaped)
    x_test_transformed=scalar.transform(x_test)
   



    early_stop = call.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=5,
    verbose=1, 
    mode='min',
    restore_best_weights=True
    )
    tf.keras.models.save_model(
    model,
    filepath,
    overwrite=True,
    include_optimizer=True,
    save_format="tf"
)

    kfold_score = []
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    for train_index, validation_index in kf.split(x_train_transformed):
        x_train, x_val = x_train_transformed[train_index], x_train_transformed[validation_index]    
        history = model.fit(x_train, x_train, validation_data=(x_val, x_val),shuffle=True, batch_size=BATCH_SIZE, epochs=50,callbacks=[early_stop])
        model.save(filepath)
        
        kfold_score.append(history.history['loss'])
        kfold_score.append(history.history['val_loss'])
        break
        print("loss",history.history['loss'])
        print("val_loss",history.history['val_loss'])
  


    
   
    
    reconstructions = model.predict(x_test_transformed)
    reconstructions_train=model.predict(x_train_transformed)

    
    

    mse_test = np.mean(np.abs(x_test_transformed - reconstructions), axis=1)
    mse_train = np.mean(np.abs(x_train_transformed - reconstructions_train), axis=1)


    df = pd.DataFrame({'mse_train': [],
                       'y_train': []})

    df2 = pd.DataFrame({'mse_test': [],
                       'y_test':[]})
    df['mse_train'] = mse_train.tolist()
    df['y_train'] = y_train.tolist()

    df2['mse_test'] = mse_test.tolist()
    df2['y_test'] = y_test.tolist()
    concat = pd.concat([df,df2], ignore_index=False, axis=1)
    concat.to_excel("mse_fix_error_dist.xlsx") 

    fig, ax = plt.subplots(figsize=(6,6))
    
    ax.hist(mse_test, bins=20, density=True, label="Test Set", alpha=0.75, color="green")
    ax.hist(mse_train, bins=20, alpha=0.75, density=True, label="Training Set", color="red")

    
    plt.title("MSE Distribution")
    plt.legend(loc='upper right')
    plt.xlabel('reconstruction error')
    plt.ylabel('occurrences')
    plt.savefig('mse_val_train_test.jpeg',format='jpeg')
    plt.show()
    plt.close()
    
    

    
    return history

def plot_history(history):


    fig, axs = plt.subplots(figsize=(10, 8))

    # create error sublpot
    axs.plot(history.history["loss"], label='Training Loss = %0.2f' % history.history["loss"][-1])
    axs.plot(history.history["val_loss"], label='Validation Loss = %0.2f' % history.history["val_loss"][-1])
    axs.set_ylabel("Loss")
    axs.set_xlabel("Epoch")
    axs.legend(loc="upper right")
    axs.set_title("Training/Validation Loss")
    
    plt.savefig('model_mean_squared_error.jpeg',format='jpeg')
    plt.show()


def reconstruct():
    
    pipeline = Pipeline([('normalizer', Normalizer()),
                     ('scaler', MinMaxScaler())])
    
    x_test_transformed = pipeline.transform(x_test)



    
if __name__ == "__main__":
 
  plot_history(train())

  


