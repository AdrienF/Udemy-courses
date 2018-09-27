import keras.models as models
import numpy as np
from keras.preprocessing import image
from datetime import datetime

model_name = 'c32c32c64fc64'
with open('models/'+model_name+'model.json','r') as f:
    model = models.model_from_json(f.read())
model.load_weights('models/'+model_name+'final_weights.h5')

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

t0 = datetime.today()
result = model.predict(test_image)
td = datetime.today() - t0

print(result)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print('prediciton : ', prediction)
print('(took {})'.format(td))