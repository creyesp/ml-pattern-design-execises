#%%
import sys
sys.path.insert(0, "/home/creyesp/Projects/repos/personal/ml-pattern-design-execises/tensorflow_tabular/")
#%%
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, concatenate
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd

import ds
import config


METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]
#%%
file_path = '/home/creyesp/tmp/tabular/data/commit_tabular_000000000000.csv.gz'
file_path = '/home/creyesp/tmp/tabular/data/'
ds_train, ds_test, data_description = ds.get_dataset(file_path)

#%%
numerical_inputs = {}
numerical_features = []

# Numeric features.
for column in config.NUMERICAL_COLUMNS:
    print(column)
    mean = data_description['numerical'][column]['mean']
    variance = data_description['numerical'][column]['variance']
    
    numeric_col = tf.keras.Input(shape=(1,), name=f'input_{column}')
    normalization_layer = preprocessing.Normalization(name=f'layer_{column}', mean=mean, variance=variance)
    print(f'layer name: {normalization_layer.name}')
    encoded_numeric_col = normalization_layer(numeric_col)
    numerical_inputs[column] = numeric_col
    numerical_features.append(encoded_numeric_col)


#%%
categorical_inputs = {}
categorical_features = []

cat_col = ['country_id', 'day_of_week_order', 'order_hour', 'business_type']
for column in cat_col:
    print(column)
    input_col = tf.keras.Input(shape=(1,), name=f'input_{column}', dtype='int32')
    # category_encoding_layer = get_category_encoding_layer(cat, ds_train, 'int32')
    vocabulary = data_description['categorical'][column]['vocabulary']
    category_encoding_layer = preprocessing.IntegerLookup(name=f'layer_{column}', vocabulary=vocabulary, output_mode='binary', mask_token=-1, oov_token=-2)
    print(f'layer name: {category_encoding_layer.name}')
    encoded_cat_col = category_encoding_layer(input_col)
    categorical_inputs[column] = input_col
    categorical_features.append(encoded_cat_col)

#%%
embedding_inputs = {}
embedding_features = []

for column in ['restaurant_id']:
    input_embedding = tf.keras.Input(shape=(1,), name=f'input_{column}', dtype='int32')
    vocabulary = data_description['categorical'][column]['vocabulary']
    restaurant_id_type = tf.feature_column.categorical_column_with_vocabulary_list(column, vocabulary_list=vocabulary)
    restaurant_id_embedding = tf.feature_column.embedding_column(restaurant_id_type, dimension=512)
    embedding_inputs[column] = input_embedding
    embedding_features.append(restaurant_id_embedding)
embedding_layer = tf.keras.layers.DenseFeatures(embedding_features)(embedding_inputs)

    

#%%
initial_bias = data_description['bias']
inputs = {}
inputs.update(embedding_inputs)
inputs.update(categorical_inputs)
inputs.update(numerical_inputs)

feature_layer = tf.keras.layers.concatenate(categorical_features + numerical_features + [embedding_layer])
dense_layer = tf.keras.layers.Dense(512, activation='relu')(feature_layer)
# dense_layer = tf.keras.layers.Dropout(rate=0.3)(dense_layer)
dense_layer = tf.keras.layers.Dense(512, activation='relu')(dense_layer)
# dense_layer = tf.keras.layers.Dropout(rate=0.3)(dense_layer)
dense_layer = tf.keras.layers.Dense(512, activation='relu')(dense_layer)
# dense_layer = tf.keras.layers.Dropout(rate=0.3)(dense_layer)
dense_layer = tf.keras.layers.Dense(512, activation='relu')(dense_layer)
output_bias = tf.keras.initializers.Constant([initial_bias])
output = tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)(feature_layer)

model = tf.keras.models.Model(inputs=inputs, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS)
model

#%% 
tf.keras.utils.plot_model(model, show_shapes=False)

#%%
history = model.fit(ds_train,
          validation_data=ds_test,
          epochs=3,
          class_weight=data_description['class_weight'],
          )
#%%
print(history.history)
#%%
#pred_y_train = model.predict(ds_train)
pred_y_test = model.predict(ds_test)
#%%
pred_y_test.shape
#%%
y_test = [k for batch in ds_test.map(lambda x, y: y).as_numpy_iterator() for k in batch]
#%%
cm = confusion_matrix(y_test,pred_y_test.squeeze()>0.5)
print(cm)
ConfusionMatrixDisplay(cm).plot()

# %%
