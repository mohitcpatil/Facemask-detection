## <center>Face Mask Detection (CNN - MLP - VGG99 - Mobinet)<center>

### **Overview**:
Since Covid-19 is now a Pandemic which a according to a lot of agencies, is still to be under control. In the U.S and many other countries in the world, masks are now mandatory in public including grocery stores, parks, schools etc. We see a need of applications which can detect masks on people entering buildings and also whether they are being socially distanced. This applications could be built in directly to the security camera feeds which is present in almost all buildings.
Places including grocery stores, parks, schools etc. We see a need of applications which can detect masks on people entering buildings and also whether they are being socially distanced. This applications could be built in directly to the security camera feeds which is present in almost all buildings.



```python
##Importing required libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from scipy.spatial import distance
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import seaborn as snsor
from tensorflow.keras.preprocessing.image import load_img
import random
import os
```


```python
import warnings
warnings.filterwarnings("ignore")
```

### **The Data** :

- Dataset of 10k images  divided between masks and no masks. 
- Images contain people across races and ages.
- Training data: 5000 with Mask and 5000 without Mask
- Validation data: 400 with Mask and 400 without Mask
- Testing data: 483 with Mask and 509 without Mask

![Image of Yaktocat](/Face Mask Dataset/figures/1st.png)

# 1. Haar Cascade for detecting social distancing violations



## Detecting faces using Haar Cascade
- Cascade function is trained from a lot of positive and negative images. 
- It is then used to detect objects in other images.
Can be operated in real time
- Feature extraction and feature evaluation (Rectangular features used)
- With new image representation, their calculation is very fast
- Classifier training and feature selection using AdaBoost
A degenerate decision tree of classifiers is formed


<img src="Face Mask Dataset/figures/haar_cascade_algorithm_flow_chart.png">

## Haar Features
- Haar features - All human faces share some similar properties. These regularities may be matched using Haar features.
- A few properties common to human faces:
The eye region is darker than the upper-cheeks
The nose bridge region is brighter than the eyes.
Composition of properties forming matchable facial features:
- Location and size: eyes, mouth, bridge of nose
- Value: oriented gradients of pixel intensities 


![alt text](https://github.com/mohitcpatilFace Mask Dataset/figures/1st.png?raw=true)

<img src="Face Mask Dataset/figures/haar_features.png">

- The four features matched by this algorithm are then sought in the image of a face
Rectangle features:
- Value = Sum (pixels in black area) - Sum (pixels in white area)
- Three types: two, three, four-rectangles, Viola & Jones used two-rectangle features
- For example: the difference in brightness between the white & black rectangles over a specific area
Each feature is related to a special location in the sub-window


## Feature extraction

<img src="Face Mask Dataset/figures/haar_feature_extraction.png">

## Complex example 

<img src="Face Mask Dataset/figures/haar_feature_complex.png">

## Creating Integral Image for faster calculation
- An image representation called the integral image evaluates rectangular features in constant time, which gives them a considerable speed advantage
- Because each features rectangular area is always adjacent to at least one other rectangle, it's easier to calculate difference between the features
- The integral images at location (x,y) is the sum of the pixels above and to the left of (x,y), inclusive.




## Integral image calculation demo

<img src="Face Mask Dataset/figures/haar_classifer_integral_image_demo.gif">


```python
all_files = []
for dirname, _, filenames in os.walk('Face Mask Dataset/images/'):
    for filename in filenames:
        all_files.append(os.path.join(dirname, filename))
```


```python
#loading haarcascade_frontalface_default.xml
face_model = cv2.CascadeClassifier('Face Mask Dataset/haarcascade_frontalface_default.xml')
```


```python
#trying it out on a sample image
img = cv2.imread('Face Mask Dataset/images/maksssksksss244.png')

img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4) #returns a list of (x,y,w,h) tuples

out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image

#plotting
for (x,y,w,h) in faces:
    cv2.rectangle(out_img,(x,y),(x+w,y+h),(0,0,255),1)
plt.figure(figsize=(12,12))
plt.imshow(out_img)
```




    <matplotlib.image.AxesImage at 0x7f800192ba90>




    
![png](Face-Mask-Detection-CNN-MLP-VGG99_files/Face-Mask-Detection-CNN-MLP-VGG99_21_1.png)
    


Iterating over the coordinates of faces and calculating the distance for each possible pair, if the distance for a particular pair is less than MIN_DISTANCE then the bounding boxes for those faces are colored red. MIN_DISTANCE must be manually initialized in such a way that it corresponds to the minimum allowable distance in real life.

### Model Validation


```python
MIN_DISTANCE = 130
```

Red box = Not socially distanced

Green box = Socially Distanced


```python
if len(faces)>=2:
    label = [0 for i in range(len(faces))]
    for i in range(len(faces)-1):
        for j in range(i+1, len(faces)):
            dist = distance.euclidean(faces[i][:2],faces[j][:2])
            if dist<MIN_DISTANCE:
                label[i] = 1
                label[j] = 1
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image
    for i in range(len(faces)):
        (x,y,w,h) = faces[i]
        if label[i]==1:
            cv2.rectangle(new_img,(x,y),(x+w,y+h),(255,0,0),1)
        else:
            cv2.rectangle(new_img,(x,y),(x+w,y+h),(0,255,0),1)
    plt.figure(figsize=(10,10))
    plt.imshow(new_img)
      
else:
    print("No. of faces detected is less than 2")
```


    
![png](Face-Mask-Detection-CNN-MLP-VGG99_files/Face-Mask-Detection-CNN-MLP-VGG99_26_0.png)
    


# 2. Face mask detection  

## 2.1 Multilayer Perceptron (CNN)

Multilayer perceptron is a broad terminology, considered as subset of Deep Neural Networks (DNN).This model introduce you with Multilayer Perceptron which is a class of Feedforward Artificial Neural Networks (ANN). It was the first and one of the simplest Artificial Neural Network. It considered as a simplest model as it only moves in one direction forward (no cycles or loops), From input nodes to hidden layer and an output layer. MLP uses backpropagation for training which anticipates the results to control the process of image transformation.


```python
import pandas as pd
import numpy as np 
import os  
from sklearn.model_selection import train_test_split 
```

### Preparing Training data



```python
labels = pd.read_csv("Face Mask Dataset/train_labels.csv")   # loading the labels
labels.head()           
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filename</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Image_1.jpg</td>
      <td>without_mask</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Image_2.jpg</td>
      <td>without_mask</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Image_3.jpg</td>
      <td>without_mask</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Image_4.jpg</td>
      <td>without_mask</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Image_5.jpg</td>
      <td>without_mask</td>
    </tr>
  </tbody>
</table>
</div>



### Image file path


```python
file_paths = [[fname, 'Face Mask Dataset/fm_train/' + fname] for fname in labels['filename']]
file_paths
```




    [['Image_1.jpg', 'Face Mask Dataset/fm_train/Image_1.jpg'],
     ['Image_2.jpg', 'Face Mask Dataset/fm_train/Image_2.jpg'],
     ['Image_3.jpg', 'Face Mask Dataset/fm_train/Image_3.jpg'],
     ['Image_4.jpg', 'Face Mask Dataset/fm_train/Image_4.jpg'],
     ['Image_5.jpg', 'Face Mask Dataset/fm_train/Image_5.jpg'],
     ['Image_6.jpg', 'Face Mask Dataset/fm_train/Image_6.jpg'],
     ['Image_7.jpg', 'Face Mask Dataset/fm_train/Image_7.jpg'],
     ['Image_8.jpg', 'Face Mask Dataset/fm_train/Image_8.jpg'],
     ['Image_9.jpg', 'Face Mask Dataset/fm_train/Image_9.jpg'],
     ['Image_10.jpg', 'Face Mask Dataset/fm_train/Image_10.jpg'],
     ['Image_11.jpg', 'Face Mask Dataset/fm_train/Image_11.jpg'],
     ['Image_12.jpg', 'Face Mask Dataset/fm_train/Image_12.jpg'],
     ['Image_13.jpg', 'Face Mask Dataset/fm_train/Image_13.jpg'],
     ['Image_14.jpg', 'Face Mask Dataset/fm_train/Image_14.jpg'],
     ['Image_15.jpg', 'Face Mask Dataset/fm_train/Image_15.jpg'],
     ['Image_16.jpg', 'Face Mask Dataset/fm_train/Image_16.jpg'],
     ['Image_17.jpg', 'Face Mask Dataset/fm_train/Image_17.jpg'],
     ['Image_18.jpg', 'Face Mask Dataset/fm_train/Image_18.jpg'],
     ['Image_19.jpg', 'Face Mask Dataset/fm_train/Image_19.jpg'],
     ['Image_20.jpg', 'Face Mask Dataset/fm_train/Image_20.jpg'],
     ['Image_21.jpg', 'Face Mask Dataset/fm_train/Image_21.jpg'],
     ['Image_22.jpg', 'Face Mask Dataset/fm_train/Image_22.jpg'],
     ['Image_23.jpg', 'Face Mask Dataset/fm_train/Image_23.jpg'],
     ['Image_24.jpg', 'Face Mask Dataset/fm_train/Image_24.jpg'],
     ['Image_25.jpg', 'Face Mask Dataset/fm_train/Image_25.jpg'],
     ['Image_26.jpg', 'Face Mask Dataset/fm_train/Image_26.jpg'],
     ['Image_27.jpg', 'Face Mask Dataset/fm_train/Image_27.jpg'],
     ['Image_28.jpg', 'Face Mask Dataset/fm_train/Image_28.jpg'],
     ['Image_29.jpg', 'Face Mask Dataset/fm_train/Image_29.jpg'],
     ['Image_30.jpg', 'Face Mask Dataset/fm_train/Image_30.jpg'],
     ['Image_31.jpg', 'Face Mask Dataset/fm_train/Image_31.jpg'],
     ['Image_32.jpg', 'Face Mask Dataset/fm_train/Image_32.jpg'],
     ['Image_33.jpg', 'Face Mask Dataset/fm_train/Image_33.jpg'],
     ['Image_34.jpg', 'Face Mask Dataset/fm_train/Image_34.jpg'],
     ['Image_35.jpg', 'Face Mask Dataset/fm_train/Image_35.jpg'],
     ['Image_36.jpg', 'Face Mask Dataset/fm_train/Image_36.jpg'],
     ['Image_37.jpg', 'Face Mask Dataset/fm_train/Image_37.jpg'],
     ['Image_38.jpg', 'Face Mask Dataset/fm_train/Image_38.jpg'],
     ['Image_39.jpg', 'Face Mask Dataset/fm_train/Image_39.jpg'],
     ['Image_40.jpg', 'Face Mask Dataset/fm_train/Image_40.jpg'],
     ['Image_41.jpg', 'Face Mask Dataset/fm_train/Image_41.jpg'],
     ['Image_42.jpg', 'Face Mask Dataset/fm_train/Image_42.jpg'],
     ['Image_43.jpg', 'Face Mask Dataset/fm_train/Image_43.jpg'],
     ['Image_44.jpg', 'Face Mask Dataset/fm_train/Image_44.jpg'],
     ['Image_45.jpg', 'Face Mask Dataset/fm_train/Image_45.jpg'],
     ['Image_46.jpg', 'Face Mask Dataset/fm_train/Image_46.jpg'],
     ['Image_47.jpg', 'Face Mask Dataset/fm_train/Image_47.jpg'],
     ['Image_48.jpg', 'Face Mask Dataset/fm_train/Image_48.jpg'],
     ['Image_49.jpg', 'Face Mask Dataset/fm_train/Image_49.jpg'],
     ['Image_50.jpg', 'Face Mask Dataset/fm_train/Image_50.jpg'],
     ['Image_51.jpg', 'Face Mask Dataset/fm_train/Image_51.jpg'],
     ['Image_52.jpg', 'Face Mask Dataset/fm_train/Image_52.jpg'],
     ['Image_53.jpg', 'Face Mask Dataset/fm_train/Image_53.jpg'],
     ['Image_54.jpg', 'Face Mask Dataset/fm_train/Image_54.jpg'],
     ['Image_55.jpg', 'Face Mask Dataset/fm_train/Image_55.jpg'],
     ['Image_56.jpg', 'Face Mask Dataset/fm_train/Image_56.jpg'],
     ['Image_57.jpg', 'Face Mask Dataset/fm_train/Image_57.jpg'],
     ['Image_58.jpg', 'Face Mask Dataset/fm_train/Image_58.jpg'],
     ['Image_59.jpg', 'Face Mask Dataset/fm_train/Image_59.jpg'],
     ['Image_60.jpg', 'Face Mask Dataset/fm_train/Image_60.jpg'],
     ['Image_61.jpg', 'Face Mask Dataset/fm_train/Image_61.jpg'],
     ['Image_62.jpg', 'Face Mask Dataset/fm_train/Image_62.jpg'],
     ['Image_63.jpg', 'Face Mask Dataset/fm_train/Image_63.jpg'],
     ['Image_64.jpg', 'Face Mask Dataset/fm_train/Image_64.jpg'],
     ['Image_65.jpg', 'Face Mask Dataset/fm_train/Image_65.jpg'],
     ['Image_66.jpg', 'Face Mask Dataset/fm_train/Image_66.jpg'],
     ['Image_67.jpg', 'Face Mask Dataset/fm_train/Image_67.jpg'],
     ['Image_68.jpg', 'Face Mask Dataset/fm_train/Image_68.jpg'],
     ['Image_69.jpg', 'Face Mask Dataset/fm_train/Image_69.jpg'],
     ['Image_70.jpg', 'Face Mask Dataset/fm_train/Image_70.jpg'],
     ['Image_71.jpg', 'Face Mask Dataset/fm_train/Image_71.jpg'],
     ['Image_72.jpg', 'Face Mask Dataset/fm_train/Image_72.jpg'],
     ['Image_73.jpg', 'Face Mask Dataset/fm_train/Image_73.jpg'],
     ['Image_74.jpg', 'Face Mask Dataset/fm_train/Image_74.jpg'],
     ['Image_75.jpg', 'Face Mask Dataset/fm_train/Image_75.jpg'],
     ['Image_76.jpg', 'Face Mask Dataset/fm_train/Image_76.jpg'],
     ['Image_77.jpg', 'Face Mask Dataset/fm_train/Image_77.jpg'],
     ['Image_78.jpg', 'Face Mask Dataset/fm_train/Image_78.jpg'],
     ['Image_79.jpg', 'Face Mask Dataset/fm_train/Image_79.jpg'],
     ['Image_80.jpg', 'Face Mask Dataset/fm_train/Image_80.jpg'],
     ['Image_81.jpg', 'Face Mask Dataset/fm_train/Image_81.jpg'],
     ['Image_82.jpg', 'Face Mask Dataset/fm_train/Image_82.jpg'],
     ['Image_83.jpg', 'Face Mask Dataset/fm_train/Image_83.jpg'],
     ['Image_84.jpg', 'Face Mask Dataset/fm_train/Image_84.jpg'],
     ['Image_85.jpg', 'Face Mask Dataset/fm_train/Image_85.jpg'],
     ['Image_86.jpg', 'Face Mask Dataset/fm_train/Image_86.jpg'],
     ['Image_87.jpg', 'Face Mask Dataset/fm_train/Image_87.jpg'],
     ['Image_88.jpg', 'Face Mask Dataset/fm_train/Image_88.jpg'],
     ['Image_89.jpg', 'Face Mask Dataset/fm_train/Image_89.jpg'],
     ['Image_90.jpg', 'Face Mask Dataset/fm_train/Image_90.jpg'],
     ['Image_91.jpg', 'Face Mask Dataset/fm_train/Image_91.jpg'],
     ['Image_92.jpg', 'Face Mask Dataset/fm_train/Image_92.jpg'],
     ['Image_93.jpg', 'Face Mask Dataset/fm_train/Image_93.jpg'],
     ['Image_94.jpg', 'Face Mask Dataset/fm_train/Image_94.jpg'],
     ['Image_95.jpg', 'Face Mask Dataset/fm_train/Image_95.jpg'],
     ['Image_96.jpg', 'Face Mask Dataset/fm_train/Image_96.jpg'],
     ['Image_97.jpg', 'Face Mask Dataset/fm_train/Image_97.jpg'],
     ['Image_98.jpg', 'Face Mask Dataset/fm_train/Image_98.jpg'],
     ['Image_99.jpg', 'Face Mask Dataset/fm_train/Image_99.jpg'],
     ['Image_100.jpg', 'Face Mask Dataset/fm_train/Image_100.jpg'],
     ['Image_101.jpg', 'Face Mask Dataset/fm_train/Image_101.jpg'],
     ['Image_102.jpg', 'Face Mask Dataset/fm_train/Image_102.jpg'],
     ['Image_103.jpg', 'Face Mask Dataset/fm_train/Image_103.jpg'],
     ['Image_104.jpg', 'Face Mask Dataset/fm_train/Image_104.jpg'],
     ['Image_105.jpg', 'Face Mask Dataset/fm_train/Image_105.jpg'],
     ['Image_106.jpg', 'Face Mask Dataset/fm_train/Image_106.jpg'],
     ['Image_107.jpg', 'Face Mask Dataset/fm_train/Image_107.jpg'],
     ['Image_108.jpg', 'Face Mask Dataset/fm_train/Image_108.jpg'],
     ['Image_109.jpg', 'Face Mask Dataset/fm_train/Image_109.jpg'],
     ['Image_110.jpg', 'Face Mask Dataset/fm_train/Image_110.jpg'],
     ['Image_111.jpg', 'Face Mask Dataset/fm_train/Image_111.jpg'],
     ['Image_112.jpg', 'Face Mask Dataset/fm_train/Image_112.jpg'],
     ['Image_113.jpg', 'Face Mask Dataset/fm_train/Image_113.jpg'],
     ['Image_114.jpg', 'Face Mask Dataset/fm_train/Image_114.jpg'],
     ['Image_115.jpg', 'Face Mask Dataset/fm_train/Image_115.jpg'],
     ['Image_116.jpg', 'Face Mask Dataset/fm_train/Image_116.jpg'],
     ['Image_117.jpg', 'Face Mask Dataset/fm_train/Image_117.jpg'],
     ['Image_118.jpg', 'Face Mask Dataset/fm_train/Image_118.jpg'],
     ['Image_119.jpg', 'Face Mask Dataset/fm_train/Image_119.jpg'],
     ['Image_120.jpg', 'Face Mask Dataset/fm_train/Image_120.jpg'],
     ['Image_121.jpg', 'Face Mask Dataset/fm_train/Image_121.jpg'],
     ['Image_122.jpg', 'Face Mask Dataset/fm_train/Image_122.jpg'],
     ['Image_123.jpg', 'Face Mask Dataset/fm_train/Image_123.jpg'],
     ['Image_124.jpg', 'Face Mask Dataset/fm_train/Image_124.jpg'],
     ['Image_125.jpg', 'Face Mask Dataset/fm_train/Image_125.jpg'],
     ['Image_126.jpg', 'Face Mask Dataset/fm_train/Image_126.jpg'],
     ['Image_127.jpg', 'Face Mask Dataset/fm_train/Image_127.jpg'],
     ['Image_128.jpg', 'Face Mask Dataset/fm_train/Image_128.jpg'],
     ['Image_129.jpg', 'Face Mask Dataset/fm_train/Image_129.jpg'],
     ['Image_130.jpg', 'Face Mask Dataset/fm_train/Image_130.jpg'],
     ['Image_131.jpg', 'Face Mask Dataset/fm_train/Image_131.jpg'],
     ['Image_132.jpg', 'Face Mask Dataset/fm_train/Image_132.jpg'],
     ['Image_133.jpg', 'Face Mask Dataset/fm_train/Image_133.jpg'],
     ['Image_134.jpg', 'Face Mask Dataset/fm_train/Image_134.jpg'],
     ['Image_135.jpg', 'Face Mask Dataset/fm_train/Image_135.jpg'],
     ['Image_136.jpg', 'Face Mask Dataset/fm_train/Image_136.jpg'],
     ['Image_137.jpg', 'Face Mask Dataset/fm_train/Image_137.jpg'],
     ['Image_138.jpg', 'Face Mask Dataset/fm_train/Image_138.jpg'],
     ['Image_139.jpg', 'Face Mask Dataset/fm_train/Image_139.jpg'],
     ['Image_140.jpg', 'Face Mask Dataset/fm_train/Image_140.jpg'],
     ['Image_141.jpg', 'Face Mask Dataset/fm_train/Image_141.jpg'],
     ['Image_142.jpg', 'Face Mask Dataset/fm_train/Image_142.jpg'],
     ['Image_143.jpg', 'Face Mask Dataset/fm_train/Image_143.jpg'],
     ['Image_144.jpg', 'Face Mask Dataset/fm_train/Image_144.jpg'],
     ['Image_145.jpg', 'Face Mask Dataset/fm_train/Image_145.jpg'],
     ['Image_146.jpg', 'Face Mask Dataset/fm_train/Image_146.jpg'],
     ['Image_147.jpg', 'Face Mask Dataset/fm_train/Image_147.jpg'],
     ['Image_148.jpg', 'Face Mask Dataset/fm_train/Image_148.jpg'],
     ['Image_149.jpg', 'Face Mask Dataset/fm_train/Image_149.jpg'],
     ['Image_150.jpg', 'Face Mask Dataset/fm_train/Image_150.jpg'],
     ['Image_151.jpg', 'Face Mask Dataset/fm_train/Image_151.jpg'],
     ['Image_152.jpg', 'Face Mask Dataset/fm_train/Image_152.jpg'],
     ['Image_153.jpg', 'Face Mask Dataset/fm_train/Image_153.jpg'],
     ['Image_154.jpg', 'Face Mask Dataset/fm_train/Image_154.jpg'],
     ['Image_155.jpg', 'Face Mask Dataset/fm_train/Image_155.jpg'],
     ['Image_156.jpg', 'Face Mask Dataset/fm_train/Image_156.jpg'],
     ['Image_157.jpg', 'Face Mask Dataset/fm_train/Image_157.jpg'],
     ['Image_158.jpg', 'Face Mask Dataset/fm_train/Image_158.jpg'],
     ['Image_159.jpg', 'Face Mask Dataset/fm_train/Image_159.jpg'],
     ['Image_160.jpg', 'Face Mask Dataset/fm_train/Image_160.jpg'],
     ['Image_161.jpg', 'Face Mask Dataset/fm_train/Image_161.jpg'],
     ['Image_162.jpg', 'Face Mask Dataset/fm_train/Image_162.jpg'],
     ['Image_163.jpg', 'Face Mask Dataset/fm_train/Image_163.jpg'],
     ['Image_164.jpg', 'Face Mask Dataset/fm_train/Image_164.jpg'],
     ['Image_165.jpg', 'Face Mask Dataset/fm_train/Image_165.jpg'],
     ['Image_166.jpg', 'Face Mask Dataset/fm_train/Image_166.jpg'],
     ['Image_167.jpg', 'Face Mask Dataset/fm_train/Image_167.jpg'],
     ['Image_168.jpg', 'Face Mask Dataset/fm_train/Image_168.jpg'],
     ['Image_169.jpg', 'Face Mask Dataset/fm_train/Image_169.jpg'],
     ['Image_170.jpg', 'Face Mask Dataset/fm_train/Image_170.jpg'],
     ['Image_171.jpg', 'Face Mask Dataset/fm_train/Image_171.jpg'],
     ['Image_172.jpg', 'Face Mask Dataset/fm_train/Image_172.jpg'],
     ['Image_173.jpg', 'Face Mask Dataset/fm_train/Image_173.jpg'],
     ['Image_174.jpg', 'Face Mask Dataset/fm_train/Image_174.jpg'],
     ['Image_175.jpg', 'Face Mask Dataset/fm_train/Image_175.jpg'],
     ['Image_176.jpg', 'Face Mask Dataset/fm_train/Image_176.jpg'],
     ['Image_177.jpg', 'Face Mask Dataset/fm_train/Image_177.jpg'],
     ['Image_178.jpg', 'Face Mask Dataset/fm_train/Image_178.jpg'],
     ['Image_179.jpg', 'Face Mask Dataset/fm_train/Image_179.jpg'],
     ['Image_180.jpg', 'Face Mask Dataset/fm_train/Image_180.jpg'],
     ['Image_181.jpg', 'Face Mask Dataset/fm_train/Image_181.jpg'],
     ['Image_182.jpg', 'Face Mask Dataset/fm_train/Image_182.jpg'],
     ['Image_183.jpg', 'Face Mask Dataset/fm_train/Image_183.jpg'],
     ['Image_184.jpg', 'Face Mask Dataset/fm_train/Image_184.jpg'],
     ['Image_185.jpg', 'Face Mask Dataset/fm_train/Image_185.jpg'],
     ['Image_186.jpg', 'Face Mask Dataset/fm_train/Image_186.jpg'],
     ['Image_187.jpg', 'Face Mask Dataset/fm_train/Image_187.jpg'],
     ['Image_188.jpg', 'Face Mask Dataset/fm_train/Image_188.jpg'],
     ['Image_189.jpg', 'Face Mask Dataset/fm_train/Image_189.jpg'],
     ['Image_190.jpg', 'Face Mask Dataset/fm_train/Image_190.jpg'],
     ['Image_191.jpg', 'Face Mask Dataset/fm_train/Image_191.jpg'],
     ['Image_192.jpg', 'Face Mask Dataset/fm_train/Image_192.jpg'],
     ['Image_193.jpg', 'Face Mask Dataset/fm_train/Image_193.jpg'],
     ['Image_194.jpg', 'Face Mask Dataset/fm_train/Image_194.jpg'],
     ['Image_195.jpg', 'Face Mask Dataset/fm_train/Image_195.jpg'],
     ['Image_196.jpg', 'Face Mask Dataset/fm_train/Image_196.jpg'],
     ['Image_197.jpg', 'Face Mask Dataset/fm_train/Image_197.jpg'],
     ['Image_198.jpg', 'Face Mask Dataset/fm_train/Image_198.jpg'],
     ['Image_199.jpg', 'Face Mask Dataset/fm_train/Image_199.jpg'],
     ['Image_200.jpg', 'Face Mask Dataset/fm_train/Image_200.jpg'],
     ['Image_201.jpg', 'Face Mask Dataset/fm_train/Image_201.jpg'],
     ['Image_202.jpg', 'Face Mask Dataset/fm_train/Image_202.jpg'],
     ['Image_203.jpg', 'Face Mask Dataset/fm_train/Image_203.jpg'],
     ['Image_204.jpg', 'Face Mask Dataset/fm_train/Image_204.jpg'],
     ['Image_205.jpg', 'Face Mask Dataset/fm_train/Image_205.jpg'],
     ['Image_206.jpg', 'Face Mask Dataset/fm_train/Image_206.jpg'],
     ['Image_207.jpg', 'Face Mask Dataset/fm_train/Image_207.jpg'],
     ['Image_208.jpg', 'Face Mask Dataset/fm_train/Image_208.jpg'],
     ['Image_209.jpg', 'Face Mask Dataset/fm_train/Image_209.jpg'],
     ['Image_210.jpg', 'Face Mask Dataset/fm_train/Image_210.jpg'],
     ['Image_211.jpg', 'Face Mask Dataset/fm_train/Image_211.jpg'],
     ['Image_212.jpg', 'Face Mask Dataset/fm_train/Image_212.jpg'],
     ['Image_213.jpg', 'Face Mask Dataset/fm_train/Image_213.jpg'],
     ['Image_214.jpg', 'Face Mask Dataset/fm_train/Image_214.jpg'],
     ['Image_215.jpg', 'Face Mask Dataset/fm_train/Image_215.jpg'],
     ['Image_216.jpg', 'Face Mask Dataset/fm_train/Image_216.jpg'],
     ['Image_217.jpg', 'Face Mask Dataset/fm_train/Image_217.jpg'],
     ['Image_218.jpg', 'Face Mask Dataset/fm_train/Image_218.jpg'],
     ['Image_219.jpg', 'Face Mask Dataset/fm_train/Image_219.jpg'],
     ['Image_220.jpg', 'Face Mask Dataset/fm_train/Image_220.jpg'],
     ['Image_221.jpg', 'Face Mask Dataset/fm_train/Image_221.jpg'],
     ['Image_222.jpg', 'Face Mask Dataset/fm_train/Image_222.jpg'],
     ['Image_223.jpg', 'Face Mask Dataset/fm_train/Image_223.jpg'],
     ['Image_224.jpg', 'Face Mask Dataset/fm_train/Image_224.jpg'],
     ['Image_225.jpg', 'Face Mask Dataset/fm_train/Image_225.jpg'],
     ['Image_226.jpg', 'Face Mask Dataset/fm_train/Image_226.jpg'],
     ['Image_227.jpg', 'Face Mask Dataset/fm_train/Image_227.jpg'],
     ['Image_228.jpg', 'Face Mask Dataset/fm_train/Image_228.jpg'],
     ['Image_229.jpg', 'Face Mask Dataset/fm_train/Image_229.jpg'],
     ['Image_230.jpg', 'Face Mask Dataset/fm_train/Image_230.jpg'],
     ['Image_231.jpg', 'Face Mask Dataset/fm_train/Image_231.jpg'],
     ['Image_232.jpg', 'Face Mask Dataset/fm_train/Image_232.jpg'],
     ['Image_233.jpg', 'Face Mask Dataset/fm_train/Image_233.jpg'],
     ['Image_234.jpg', 'Face Mask Dataset/fm_train/Image_234.jpg'],
     ['Image_235.jpg', 'Face Mask Dataset/fm_train/Image_235.jpg'],
     ['Image_236.jpg', 'Face Mask Dataset/fm_train/Image_236.jpg'],
     ['Image_237.jpg', 'Face Mask Dataset/fm_train/Image_237.jpg'],
     ['Image_238.jpg', 'Face Mask Dataset/fm_train/Image_238.jpg'],
     ['Image_239.jpg', 'Face Mask Dataset/fm_train/Image_239.jpg'],
     ['Image_240.jpg', 'Face Mask Dataset/fm_train/Image_240.jpg'],
     ['Image_241.jpg', 'Face Mask Dataset/fm_train/Image_241.jpg'],
     ['Image_242.jpg', 'Face Mask Dataset/fm_train/Image_242.jpg'],
     ['Image_243.jpg', 'Face Mask Dataset/fm_train/Image_243.jpg'],
     ['Image_244.jpg', 'Face Mask Dataset/fm_train/Image_244.jpg'],
     ['Image_245.jpg', 'Face Mask Dataset/fm_train/Image_245.jpg'],
     ['Image_246.jpg', 'Face Mask Dataset/fm_train/Image_246.jpg'],
     ['Image_247.jpg', 'Face Mask Dataset/fm_train/Image_247.jpg'],
     ['Image_248.jpg', 'Face Mask Dataset/fm_train/Image_248.jpg'],
     ['Image_249.jpg', 'Face Mask Dataset/fm_train/Image_249.jpg'],
     ['Image_250.jpg', 'Face Mask Dataset/fm_train/Image_250.jpg'],
     ['Image_251.jpg', 'Face Mask Dataset/fm_train/Image_251.jpg'],
     ['Image_252.jpg', 'Face Mask Dataset/fm_train/Image_252.jpg'],
     ['Image_253.jpg', 'Face Mask Dataset/fm_train/Image_253.jpg'],
     ['Image_254.jpg', 'Face Mask Dataset/fm_train/Image_254.jpg'],
     ['Image_255.jpg', 'Face Mask Dataset/fm_train/Image_255.jpg'],
     ['Image_256.jpg', 'Face Mask Dataset/fm_train/Image_256.jpg'],
     ['Image_257.jpg', 'Face Mask Dataset/fm_train/Image_257.jpg'],
     ['Image_258.jpg', 'Face Mask Dataset/fm_train/Image_258.jpg'],
     ['Image_259.jpg', 'Face Mask Dataset/fm_train/Image_259.jpg'],
     ['Image_260.jpg', 'Face Mask Dataset/fm_train/Image_260.jpg'],
     ['Image_261.jpg', 'Face Mask Dataset/fm_train/Image_261.jpg'],
     ['Image_262.jpg', 'Face Mask Dataset/fm_train/Image_262.jpg'],
     ['Image_263.jpg', 'Face Mask Dataset/fm_train/Image_263.jpg'],
     ['Image_264.jpg', 'Face Mask Dataset/fm_train/Image_264.jpg'],
     ['Image_265.jpg', 'Face Mask Dataset/fm_train/Image_265.jpg'],
     ['Image_266.jpg', 'Face Mask Dataset/fm_train/Image_266.jpg'],
     ['Image_267.jpg', 'Face Mask Dataset/fm_train/Image_267.jpg'],
     ['Image_268.jpg', 'Face Mask Dataset/fm_train/Image_268.jpg'],
     ['Image_269.jpg', 'Face Mask Dataset/fm_train/Image_269.jpg'],
     ['Image_270.jpg', 'Face Mask Dataset/fm_train/Image_270.jpg'],
     ['Image_271.jpg', 'Face Mask Dataset/fm_train/Image_271.jpg'],
     ['Image_272.jpg', 'Face Mask Dataset/fm_train/Image_272.jpg'],
     ['Image_273.jpg', 'Face Mask Dataset/fm_train/Image_273.jpg'],
     ['Image_274.jpg', 'Face Mask Dataset/fm_train/Image_274.jpg'],
     ['Image_275.jpg', 'Face Mask Dataset/fm_train/Image_275.jpg'],
     ['Image_276.jpg', 'Face Mask Dataset/fm_train/Image_276.jpg'],
     ['Image_277.jpg', 'Face Mask Dataset/fm_train/Image_277.jpg'],
     ['Image_278.jpg', 'Face Mask Dataset/fm_train/Image_278.jpg'],
     ['Image_279.jpg', 'Face Mask Dataset/fm_train/Image_279.jpg'],
     ['Image_280.jpg', 'Face Mask Dataset/fm_train/Image_280.jpg'],
     ['Image_281.jpg', 'Face Mask Dataset/fm_train/Image_281.jpg'],
     ['Image_282.jpg', 'Face Mask Dataset/fm_train/Image_282.jpg'],
     ['Image_283.jpg', 'Face Mask Dataset/fm_train/Image_283.jpg'],
     ['Image_284.jpg', 'Face Mask Dataset/fm_train/Image_284.jpg'],
     ['Image_285.jpg', 'Face Mask Dataset/fm_train/Image_285.jpg'],
     ['Image_286.jpg', 'Face Mask Dataset/fm_train/Image_286.jpg'],
     ['Image_287.jpg', 'Face Mask Dataset/fm_train/Image_287.jpg'],
     ['Image_288.jpg', 'Face Mask Dataset/fm_train/Image_288.jpg'],
     ['Image_289.jpg', 'Face Mask Dataset/fm_train/Image_289.jpg'],
     ['Image_290.jpg', 'Face Mask Dataset/fm_train/Image_290.jpg'],
     ['Image_291.jpg', 'Face Mask Dataset/fm_train/Image_291.jpg'],
     ['Image_292.jpg', 'Face Mask Dataset/fm_train/Image_292.jpg'],
     ['Image_293.jpg', 'Face Mask Dataset/fm_train/Image_293.jpg'],
     ['Image_294.jpg', 'Face Mask Dataset/fm_train/Image_294.jpg'],
     ['Image_295.jpg', 'Face Mask Dataset/fm_train/Image_295.jpg'],
     ['Image_296.jpg', 'Face Mask Dataset/fm_train/Image_296.jpg'],
     ['Image_297.jpg', 'Face Mask Dataset/fm_train/Image_297.jpg'],
     ['Image_298.jpg', 'Face Mask Dataset/fm_train/Image_298.jpg'],
     ['Image_299.jpg', 'Face Mask Dataset/fm_train/Image_299.jpg'],
     ['Image_300.jpg', 'Face Mask Dataset/fm_train/Image_300.jpg'],
     ['Image_301.jpg', 'Face Mask Dataset/fm_train/Image_301.jpg'],
     ['Image_302.jpg', 'Face Mask Dataset/fm_train/Image_302.jpg'],
     ['Image_303.jpg', 'Face Mask Dataset/fm_train/Image_303.jpg'],
     ['Image_304.jpg', 'Face Mask Dataset/fm_train/Image_304.jpg'],
     ['Image_305.jpg', 'Face Mask Dataset/fm_train/Image_305.jpg'],
     ['Image_306.jpg', 'Face Mask Dataset/fm_train/Image_306.jpg'],
     ['Image_307.jpg', 'Face Mask Dataset/fm_train/Image_307.jpg'],
     ['Image_308.jpg', 'Face Mask Dataset/fm_train/Image_308.jpg'],
     ['Image_309.jpg', 'Face Mask Dataset/fm_train/Image_309.jpg'],
     ['Image_310.jpg', 'Face Mask Dataset/fm_train/Image_310.jpg'],
     ['Image_311.jpg', 'Face Mask Dataset/fm_train/Image_311.jpg'],
     ['Image_312.jpg', 'Face Mask Dataset/fm_train/Image_312.jpg'],
     ['Image_313.jpg', 'Face Mask Dataset/fm_train/Image_313.jpg'],
     ['Image_314.jpg', 'Face Mask Dataset/fm_train/Image_314.jpg'],
     ['Image_315.jpg', 'Face Mask Dataset/fm_train/Image_315.jpg'],
     ['Image_316.jpg', 'Face Mask Dataset/fm_train/Image_316.jpg'],
     ['Image_317.jpg', 'Face Mask Dataset/fm_train/Image_317.jpg'],
     ['Image_318.jpg', 'Face Mask Dataset/fm_train/Image_318.jpg'],
     ['Image_319.jpg', 'Face Mask Dataset/fm_train/Image_319.jpg'],
     ['Image_320.jpg', 'Face Mask Dataset/fm_train/Image_320.jpg'],
     ['Image_321.jpg', 'Face Mask Dataset/fm_train/Image_321.jpg'],
     ['Image_322.jpg', 'Face Mask Dataset/fm_train/Image_322.jpg'],
     ['Image_323.jpg', 'Face Mask Dataset/fm_train/Image_323.jpg'],
     ['Image_324.jpg', 'Face Mask Dataset/fm_train/Image_324.jpg'],
     ['Image_325.jpg', 'Face Mask Dataset/fm_train/Image_325.jpg'],
     ['Image_326.jpg', 'Face Mask Dataset/fm_train/Image_326.jpg'],
     ['Image_327.jpg', 'Face Mask Dataset/fm_train/Image_327.jpg'],
     ['Image_328.jpg', 'Face Mask Dataset/fm_train/Image_328.jpg'],
     ['Image_329.jpg', 'Face Mask Dataset/fm_train/Image_329.jpg'],
     ['Image_330.jpg', 'Face Mask Dataset/fm_train/Image_330.jpg'],
     ['Image_331.jpg', 'Face Mask Dataset/fm_train/Image_331.jpg'],
     ['Image_332.jpg', 'Face Mask Dataset/fm_train/Image_332.jpg'],
     ['Image_333.jpg', 'Face Mask Dataset/fm_train/Image_333.jpg'],
     ['Image_334.jpg', 'Face Mask Dataset/fm_train/Image_334.jpg'],
     ['Image_335.jpg', 'Face Mask Dataset/fm_train/Image_335.jpg'],
     ['Image_336.jpg', 'Face Mask Dataset/fm_train/Image_336.jpg'],
     ['Image_337.jpg', 'Face Mask Dataset/fm_train/Image_337.jpg'],
     ['Image_338.jpg', 'Face Mask Dataset/fm_train/Image_338.jpg'],
     ['Image_339.jpg', 'Face Mask Dataset/fm_train/Image_339.jpg'],
     ['Image_340.jpg', 'Face Mask Dataset/fm_train/Image_340.jpg'],
     ['Image_341.jpg', 'Face Mask Dataset/fm_train/Image_341.jpg'],
     ['Image_342.jpg', 'Face Mask Dataset/fm_train/Image_342.jpg'],
     ['Image_343.jpg', 'Face Mask Dataset/fm_train/Image_343.jpg'],
     ['Image_344.jpg', 'Face Mask Dataset/fm_train/Image_344.jpg'],
     ['Image_345.jpg', 'Face Mask Dataset/fm_train/Image_345.jpg'],
     ['Image_346.jpg', 'Face Mask Dataset/fm_train/Image_346.jpg'],
     ['Image_347.jpg', 'Face Mask Dataset/fm_train/Image_347.jpg'],
     ['Image_348.jpg', 'Face Mask Dataset/fm_train/Image_348.jpg'],
     ['Image_349.jpg', 'Face Mask Dataset/fm_train/Image_349.jpg'],
     ['Image_350.jpg', 'Face Mask Dataset/fm_train/Image_350.jpg'],
     ['Image_351.jpg', 'Face Mask Dataset/fm_train/Image_351.jpg'],
     ['Image_352.jpg', 'Face Mask Dataset/fm_train/Image_352.jpg'],
     ['Image_353.jpg', 'Face Mask Dataset/fm_train/Image_353.jpg'],
     ['Image_354.jpg', 'Face Mask Dataset/fm_train/Image_354.jpg'],
     ['Image_355.jpg', 'Face Mask Dataset/fm_train/Image_355.jpg'],
     ['Image_356.jpg', 'Face Mask Dataset/fm_train/Image_356.jpg'],
     ['Image_357.jpg', 'Face Mask Dataset/fm_train/Image_357.jpg'],
     ['Image_358.jpg', 'Face Mask Dataset/fm_train/Image_358.jpg'],
     ['Image_359.jpg', 'Face Mask Dataset/fm_train/Image_359.jpg'],
     ['Image_360.jpg', 'Face Mask Dataset/fm_train/Image_360.jpg'],
     ['Image_361.jpg', 'Face Mask Dataset/fm_train/Image_361.jpg'],
     ['Image_362.jpg', 'Face Mask Dataset/fm_train/Image_362.jpg'],
     ['Image_363.jpg', 'Face Mask Dataset/fm_train/Image_363.jpg'],
     ['Image_364.jpg', 'Face Mask Dataset/fm_train/Image_364.jpg'],
     ['Image_365.jpg', 'Face Mask Dataset/fm_train/Image_365.jpg'],
     ['Image_366.jpg', 'Face Mask Dataset/fm_train/Image_366.jpg'],
     ['Image_367.jpg', 'Face Mask Dataset/fm_train/Image_367.jpg'],
     ['Image_368.jpg', 'Face Mask Dataset/fm_train/Image_368.jpg'],
     ['Image_369.jpg', 'Face Mask Dataset/fm_train/Image_369.jpg'],
     ['Image_370.jpg', 'Face Mask Dataset/fm_train/Image_370.jpg'],
     ['Image_371.jpg', 'Face Mask Dataset/fm_train/Image_371.jpg'],
     ['Image_372.jpg', 'Face Mask Dataset/fm_train/Image_372.jpg'],
     ['Image_373.jpg', 'Face Mask Dataset/fm_train/Image_373.jpg'],
     ['Image_374.jpg', 'Face Mask Dataset/fm_train/Image_374.jpg'],
     ['Image_375.jpg', 'Face Mask Dataset/fm_train/Image_375.jpg'],
     ['Image_376.jpg', 'Face Mask Dataset/fm_train/Image_376.jpg'],
     ['Image_377.jpg', 'Face Mask Dataset/fm_train/Image_377.jpg'],
     ['Image_378.jpg', 'Face Mask Dataset/fm_train/Image_378.jpg'],
     ['Image_379.jpg', 'Face Mask Dataset/fm_train/Image_379.jpg'],
     ['Image_380.jpg', 'Face Mask Dataset/fm_train/Image_380.jpg'],
     ['Image_381.jpg', 'Face Mask Dataset/fm_train/Image_381.jpg'],
     ['Image_382.jpg', 'Face Mask Dataset/fm_train/Image_382.jpg'],
     ['Image_383.jpg', 'Face Mask Dataset/fm_train/Image_383.jpg'],
     ['Image_384.jpg', 'Face Mask Dataset/fm_train/Image_384.jpg'],
     ['Image_385.jpg', 'Face Mask Dataset/fm_train/Image_385.jpg'],
     ['Image_386.jpg', 'Face Mask Dataset/fm_train/Image_386.jpg'],
     ['Image_387.jpg', 'Face Mask Dataset/fm_train/Image_387.jpg'],
     ['Image_388.jpg', 'Face Mask Dataset/fm_train/Image_388.jpg'],
     ['Image_389.jpg', 'Face Mask Dataset/fm_train/Image_389.jpg'],
     ['Image_390.jpg', 'Face Mask Dataset/fm_train/Image_390.jpg'],
     ['Image_391.jpg', 'Face Mask Dataset/fm_train/Image_391.jpg'],
     ['Image_392.jpg', 'Face Mask Dataset/fm_train/Image_392.jpg'],
     ['Image_393.jpg', 'Face Mask Dataset/fm_train/Image_393.jpg'],
     ['Image_394.jpg', 'Face Mask Dataset/fm_train/Image_394.jpg'],
     ['Image_395.jpg', 'Face Mask Dataset/fm_train/Image_395.jpg'],
     ['Image_396.jpg', 'Face Mask Dataset/fm_train/Image_396.jpg'],
     ['Image_397.jpg', 'Face Mask Dataset/fm_train/Image_397.jpg'],
     ['Image_398.jpg', 'Face Mask Dataset/fm_train/Image_398.jpg'],
     ['Image_399.jpg', 'Face Mask Dataset/fm_train/Image_399.jpg'],
     ['Image_400.jpg', 'Face Mask Dataset/fm_train/Image_400.jpg'],
     ['Image_401.jpg', 'Face Mask Dataset/fm_train/Image_401.jpg'],
     ['Image_402.jpg', 'Face Mask Dataset/fm_train/Image_402.jpg'],
     ['Image_403.jpg', 'Face Mask Dataset/fm_train/Image_403.jpg'],
     ['Image_404.jpg', 'Face Mask Dataset/fm_train/Image_404.jpg'],
     ['Image_405.jpg', 'Face Mask Dataset/fm_train/Image_405.jpg'],
     ['Image_406.jpg', 'Face Mask Dataset/fm_train/Image_406.jpg'],
     ['Image_407.jpg', 'Face Mask Dataset/fm_train/Image_407.jpg'],
     ['Image_408.jpg', 'Face Mask Dataset/fm_train/Image_408.jpg'],
     ['Image_409.jpg', 'Face Mask Dataset/fm_train/Image_409.jpg'],
     ['Image_410.jpg', 'Face Mask Dataset/fm_train/Image_410.jpg'],
     ['Image_411.jpg', 'Face Mask Dataset/fm_train/Image_411.jpg'],
     ['Image_412.jpg', 'Face Mask Dataset/fm_train/Image_412.jpg'],
     ['Image_413.jpg', 'Face Mask Dataset/fm_train/Image_413.jpg'],
     ['Image_414.jpg', 'Face Mask Dataset/fm_train/Image_414.jpg'],
     ['Image_415.jpg', 'Face Mask Dataset/fm_train/Image_415.jpg'],
     ['Image_416.jpg', 'Face Mask Dataset/fm_train/Image_416.jpg'],
     ['Image_417.jpg', 'Face Mask Dataset/fm_train/Image_417.jpg'],
     ['Image_418.jpg', 'Face Mask Dataset/fm_train/Image_418.jpg'],
     ['Image_419.jpg', 'Face Mask Dataset/fm_train/Image_419.jpg'],
     ['Image_420.jpg', 'Face Mask Dataset/fm_train/Image_420.jpg'],
     ['Image_421.jpg', 'Face Mask Dataset/fm_train/Image_421.jpg'],
     ['Image_422.jpg', 'Face Mask Dataset/fm_train/Image_422.jpg'],
     ['Image_423.jpg', 'Face Mask Dataset/fm_train/Image_423.jpg'],
     ['Image_424.jpg', 'Face Mask Dataset/fm_train/Image_424.jpg'],
     ['Image_425.jpg', 'Face Mask Dataset/fm_train/Image_425.jpg'],
     ['Image_426.jpg', 'Face Mask Dataset/fm_train/Image_426.jpg'],
     ['Image_427.jpg', 'Face Mask Dataset/fm_train/Image_427.jpg'],
     ['Image_428.jpg', 'Face Mask Dataset/fm_train/Image_428.jpg'],
     ['Image_429.jpg', 'Face Mask Dataset/fm_train/Image_429.jpg'],
     ['Image_430.jpg', 'Face Mask Dataset/fm_train/Image_430.jpg'],
     ['Image_431.jpg', 'Face Mask Dataset/fm_train/Image_431.jpg'],
     ['Image_432.jpg', 'Face Mask Dataset/fm_train/Image_432.jpg'],
     ['Image_433.jpg', 'Face Mask Dataset/fm_train/Image_433.jpg'],
     ['Image_434.jpg', 'Face Mask Dataset/fm_train/Image_434.jpg'],
     ['Image_435.jpg', 'Face Mask Dataset/fm_train/Image_435.jpg'],
     ['Image_436.jpg', 'Face Mask Dataset/fm_train/Image_436.jpg'],
     ['Image_437.jpg', 'Face Mask Dataset/fm_train/Image_437.jpg'],
     ['Image_438.jpg', 'Face Mask Dataset/fm_train/Image_438.jpg'],
     ['Image_439.jpg', 'Face Mask Dataset/fm_train/Image_439.jpg'],
     ['Image_440.jpg', 'Face Mask Dataset/fm_train/Image_440.jpg'],
     ['Image_441.jpg', 'Face Mask Dataset/fm_train/Image_441.jpg'],
     ['Image_442.jpg', 'Face Mask Dataset/fm_train/Image_442.jpg'],
     ['Image_443.jpg', 'Face Mask Dataset/fm_train/Image_443.jpg'],
     ['Image_444.jpg', 'Face Mask Dataset/fm_train/Image_444.jpg'],
     ['Image_445.jpg', 'Face Mask Dataset/fm_train/Image_445.jpg'],
     ['Image_446.jpg', 'Face Mask Dataset/fm_train/Image_446.jpg'],
     ['Image_447.jpg', 'Face Mask Dataset/fm_train/Image_447.jpg'],
     ['Image_448.jpg', 'Face Mask Dataset/fm_train/Image_448.jpg'],
     ['Image_449.jpg', 'Face Mask Dataset/fm_train/Image_449.jpg'],
     ['Image_450.jpg', 'Face Mask Dataset/fm_train/Image_450.jpg'],
     ['Image_451.jpg', 'Face Mask Dataset/fm_train/Image_451.jpg'],
     ['Image_452.jpg', 'Face Mask Dataset/fm_train/Image_452.jpg'],
     ['Image_453.jpg', 'Face Mask Dataset/fm_train/Image_453.jpg'],
     ['Image_454.jpg', 'Face Mask Dataset/fm_train/Image_454.jpg'],
     ['Image_455.jpg', 'Face Mask Dataset/fm_train/Image_455.jpg'],
     ['Image_456.jpg', 'Face Mask Dataset/fm_train/Image_456.jpg'],
     ['Image_457.jpg', 'Face Mask Dataset/fm_train/Image_457.jpg'],
     ['Image_458.jpg', 'Face Mask Dataset/fm_train/Image_458.jpg'],
     ['Image_459.jpg', 'Face Mask Dataset/fm_train/Image_459.jpg'],
     ['Image_460.jpg', 'Face Mask Dataset/fm_train/Image_460.jpg'],
     ['Image_461.jpg', 'Face Mask Dataset/fm_train/Image_461.jpg'],
     ['Image_462.jpg', 'Face Mask Dataset/fm_train/Image_462.jpg'],
     ['Image_463.jpg', 'Face Mask Dataset/fm_train/Image_463.jpg'],
     ['Image_464.jpg', 'Face Mask Dataset/fm_train/Image_464.jpg'],
     ['Image_465.jpg', 'Face Mask Dataset/fm_train/Image_465.jpg'],
     ['Image_466.jpg', 'Face Mask Dataset/fm_train/Image_466.jpg'],
     ['Image_467.jpg', 'Face Mask Dataset/fm_train/Image_467.jpg'],
     ['Image_468.jpg', 'Face Mask Dataset/fm_train/Image_468.jpg'],
     ['Image_469.jpg', 'Face Mask Dataset/fm_train/Image_469.jpg'],
     ['Image_470.jpg', 'Face Mask Dataset/fm_train/Image_470.jpg'],
     ['Image_471.jpg', 'Face Mask Dataset/fm_train/Image_471.jpg'],
     ['Image_472.jpg', 'Face Mask Dataset/fm_train/Image_472.jpg'],
     ['Image_473.jpg', 'Face Mask Dataset/fm_train/Image_473.jpg'],
     ['Image_474.jpg', 'Face Mask Dataset/fm_train/Image_474.jpg'],
     ['Image_475.jpg', 'Face Mask Dataset/fm_train/Image_475.jpg'],
     ['Image_476.jpg', 'Face Mask Dataset/fm_train/Image_476.jpg'],
     ['Image_477.jpg', 'Face Mask Dataset/fm_train/Image_477.jpg'],
     ['Image_478.jpg', 'Face Mask Dataset/fm_train/Image_478.jpg'],
     ['Image_479.jpg', 'Face Mask Dataset/fm_train/Image_479.jpg'],
     ['Image_480.jpg', 'Face Mask Dataset/fm_train/Image_480.jpg'],
     ['Image_481.jpg', 'Face Mask Dataset/fm_train/Image_481.jpg'],
     ['Image_482.jpg', 'Face Mask Dataset/fm_train/Image_482.jpg'],
     ['Image_483.jpg', 'Face Mask Dataset/fm_train/Image_483.jpg'],
     ['Image_484.jpg', 'Face Mask Dataset/fm_train/Image_484.jpg'],
     ['Image_485.jpg', 'Face Mask Dataset/fm_train/Image_485.jpg'],
     ['Image_486.jpg', 'Face Mask Dataset/fm_train/Image_486.jpg'],
     ['Image_487.jpg', 'Face Mask Dataset/fm_train/Image_487.jpg'],
     ['Image_488.jpg', 'Face Mask Dataset/fm_train/Image_488.jpg'],
     ['Image_489.jpg', 'Face Mask Dataset/fm_train/Image_489.jpg'],
     ['Image_490.jpg', 'Face Mask Dataset/fm_train/Image_490.jpg'],
     ['Image_491.jpg', 'Face Mask Dataset/fm_train/Image_491.jpg'],
     ['Image_492.jpg', 'Face Mask Dataset/fm_train/Image_492.jpg'],
     ['Image_493.jpg', 'Face Mask Dataset/fm_train/Image_493.jpg'],
     ['Image_494.jpg', 'Face Mask Dataset/fm_train/Image_494.jpg'],
     ['Image_495.jpg', 'Face Mask Dataset/fm_train/Image_495.jpg'],
     ['Image_496.jpg', 'Face Mask Dataset/fm_train/Image_496.jpg'],
     ['Image_497.jpg', 'Face Mask Dataset/fm_train/Image_497.jpg'],
     ['Image_498.jpg', 'Face Mask Dataset/fm_train/Image_498.jpg'],
     ['Image_499.jpg', 'Face Mask Dataset/fm_train/Image_499.jpg'],
     ['Image_500.jpg', 'Face Mask Dataset/fm_train/Image_500.jpg'],
     ['Image_501.jpg', 'Face Mask Dataset/fm_train/Image_501.jpg'],
     ['Image_502.jpg', 'Face Mask Dataset/fm_train/Image_502.jpg'],
     ['Image_503.jpg', 'Face Mask Dataset/fm_train/Image_503.jpg'],
     ['Image_504.jpg', 'Face Mask Dataset/fm_train/Image_504.jpg'],
     ['Image_505.jpg', 'Face Mask Dataset/fm_train/Image_505.jpg'],
     ['Image_506.jpg', 'Face Mask Dataset/fm_train/Image_506.jpg'],
     ['Image_507.jpg', 'Face Mask Dataset/fm_train/Image_507.jpg'],
     ['Image_508.jpg', 'Face Mask Dataset/fm_train/Image_508.jpg'],
     ['Image_509.jpg', 'Face Mask Dataset/fm_train/Image_509.jpg'],
     ['Image_510.jpg', 'Face Mask Dataset/fm_train/Image_510.jpg'],
     ['Image_511.jpg', 'Face Mask Dataset/fm_train/Image_511.jpg'],
     ['Image_512.jpg', 'Face Mask Dataset/fm_train/Image_512.jpg'],
     ['Image_513.jpg', 'Face Mask Dataset/fm_train/Image_513.jpg'],
     ['Image_514.jpg', 'Face Mask Dataset/fm_train/Image_514.jpg'],
     ['Image_515.jpg', 'Face Mask Dataset/fm_train/Image_515.jpg'],
     ['Image_516.jpg', 'Face Mask Dataset/fm_train/Image_516.jpg'],
     ['Image_517.jpg', 'Face Mask Dataset/fm_train/Image_517.jpg'],
     ['Image_518.jpg', 'Face Mask Dataset/fm_train/Image_518.jpg'],
     ['Image_519.jpg', 'Face Mask Dataset/fm_train/Image_519.jpg'],
     ['Image_520.jpg', 'Face Mask Dataset/fm_train/Image_520.jpg'],
     ['Image_521.jpg', 'Face Mask Dataset/fm_train/Image_521.jpg'],
     ['Image_522.jpg', 'Face Mask Dataset/fm_train/Image_522.jpg'],
     ['Image_523.jpg', 'Face Mask Dataset/fm_train/Image_523.jpg'],
     ['Image_524.jpg', 'Face Mask Dataset/fm_train/Image_524.jpg'],
     ['Image_525.jpg', 'Face Mask Dataset/fm_train/Image_525.jpg'],
     ['Image_526.jpg', 'Face Mask Dataset/fm_train/Image_526.jpg'],
     ['Image_527.jpg', 'Face Mask Dataset/fm_train/Image_527.jpg'],
     ['Image_528.jpg', 'Face Mask Dataset/fm_train/Image_528.jpg'],
     ['Image_529.jpg', 'Face Mask Dataset/fm_train/Image_529.jpg'],
     ['Image_530.jpg', 'Face Mask Dataset/fm_train/Image_530.jpg'],
     ['Image_531.jpg', 'Face Mask Dataset/fm_train/Image_531.jpg'],
     ['Image_532.jpg', 'Face Mask Dataset/fm_train/Image_532.jpg'],
     ['Image_533.jpg', 'Face Mask Dataset/fm_train/Image_533.jpg'],
     ['Image_534.jpg', 'Face Mask Dataset/fm_train/Image_534.jpg'],
     ['Image_535.jpg', 'Face Mask Dataset/fm_train/Image_535.jpg'],
     ['Image_536.jpg', 'Face Mask Dataset/fm_train/Image_536.jpg'],
     ['Image_537.jpg', 'Face Mask Dataset/fm_train/Image_537.jpg'],
     ['Image_538.jpg', 'Face Mask Dataset/fm_train/Image_538.jpg'],
     ['Image_539.jpg', 'Face Mask Dataset/fm_train/Image_539.jpg'],
     ['Image_540.jpg', 'Face Mask Dataset/fm_train/Image_540.jpg'],
     ['Image_541.jpg', 'Face Mask Dataset/fm_train/Image_541.jpg'],
     ['Image_542.jpg', 'Face Mask Dataset/fm_train/Image_542.jpg'],
     ['Image_543.jpg', 'Face Mask Dataset/fm_train/Image_543.jpg'],
     ['Image_544.jpg', 'Face Mask Dataset/fm_train/Image_544.jpg'],
     ['Image_545.jpg', 'Face Mask Dataset/fm_train/Image_545.jpg'],
     ['Image_546.jpg', 'Face Mask Dataset/fm_train/Image_546.jpg'],
     ['Image_547.jpg', 'Face Mask Dataset/fm_train/Image_547.jpg'],
     ['Image_548.jpg', 'Face Mask Dataset/fm_train/Image_548.jpg'],
     ['Image_549.jpg', 'Face Mask Dataset/fm_train/Image_549.jpg'],
     ['Image_550.jpg', 'Face Mask Dataset/fm_train/Image_550.jpg'],
     ['Image_551.jpg', 'Face Mask Dataset/fm_train/Image_551.jpg'],
     ['Image_552.jpg', 'Face Mask Dataset/fm_train/Image_552.jpg'],
     ['Image_553.jpg', 'Face Mask Dataset/fm_train/Image_553.jpg'],
     ['Image_554.jpg', 'Face Mask Dataset/fm_train/Image_554.jpg'],
     ['Image_555.jpg', 'Face Mask Dataset/fm_train/Image_555.jpg'],
     ['Image_556.jpg', 'Face Mask Dataset/fm_train/Image_556.jpg'],
     ['Image_557.jpg', 'Face Mask Dataset/fm_train/Image_557.jpg'],
     ['Image_558.jpg', 'Face Mask Dataset/fm_train/Image_558.jpg'],
     ['Image_559.jpg', 'Face Mask Dataset/fm_train/Image_559.jpg'],
     ['Image_560.jpg', 'Face Mask Dataset/fm_train/Image_560.jpg'],
     ['Image_561.jpg', 'Face Mask Dataset/fm_train/Image_561.jpg'],
     ['Image_562.jpg', 'Face Mask Dataset/fm_train/Image_562.jpg'],
     ['Image_563.jpg', 'Face Mask Dataset/fm_train/Image_563.jpg'],
     ['Image_564.jpg', 'Face Mask Dataset/fm_train/Image_564.jpg'],
     ['Image_565.jpg', 'Face Mask Dataset/fm_train/Image_565.jpg'],
     ['Image_566.jpg', 'Face Mask Dataset/fm_train/Image_566.jpg'],
     ['Image_567.jpg', 'Face Mask Dataset/fm_train/Image_567.jpg'],
     ['Image_568.jpg', 'Face Mask Dataset/fm_train/Image_568.jpg'],
     ['Image_569.jpg', 'Face Mask Dataset/fm_train/Image_569.jpg'],
     ['Image_570.jpg', 'Face Mask Dataset/fm_train/Image_570.jpg'],
     ['Image_571.jpg', 'Face Mask Dataset/fm_train/Image_571.jpg'],
     ['Image_572.jpg', 'Face Mask Dataset/fm_train/Image_572.jpg'],
     ['Image_573.jpg', 'Face Mask Dataset/fm_train/Image_573.jpg'],
     ['Image_574.jpg', 'Face Mask Dataset/fm_train/Image_574.jpg'],
     ['Image_575.jpg', 'Face Mask Dataset/fm_train/Image_575.jpg'],
     ['Image_576.jpg', 'Face Mask Dataset/fm_train/Image_576.jpg'],
     ['Image_577.jpg', 'Face Mask Dataset/fm_train/Image_577.jpg'],
     ['Image_578.jpg', 'Face Mask Dataset/fm_train/Image_578.jpg'],
     ['Image_579.jpg', 'Face Mask Dataset/fm_train/Image_579.jpg'],
     ['Image_580.jpg', 'Face Mask Dataset/fm_train/Image_580.jpg'],
     ['Image_581.jpg', 'Face Mask Dataset/fm_train/Image_581.jpg'],
     ['Image_582.jpg', 'Face Mask Dataset/fm_train/Image_582.jpg'],
     ['Image_583.jpg', 'Face Mask Dataset/fm_train/Image_583.jpg'],
     ['Image_584.jpg', 'Face Mask Dataset/fm_train/Image_584.jpg'],
     ['Image_585.jpg', 'Face Mask Dataset/fm_train/Image_585.jpg'],
     ['Image_586.jpg', 'Face Mask Dataset/fm_train/Image_586.jpg'],
     ['Image_587.jpg', 'Face Mask Dataset/fm_train/Image_587.jpg'],
     ['Image_588.jpg', 'Face Mask Dataset/fm_train/Image_588.jpg'],
     ['Image_589.jpg', 'Face Mask Dataset/fm_train/Image_589.jpg'],
     ['Image_590.jpg', 'Face Mask Dataset/fm_train/Image_590.jpg'],
     ['Image_591.jpg', 'Face Mask Dataset/fm_train/Image_591.jpg'],
     ['Image_592.jpg', 'Face Mask Dataset/fm_train/Image_592.jpg'],
     ['Image_593.jpg', 'Face Mask Dataset/fm_train/Image_593.jpg'],
     ['Image_594.jpg', 'Face Mask Dataset/fm_train/Image_594.jpg'],
     ['Image_595.jpg', 'Face Mask Dataset/fm_train/Image_595.jpg'],
     ['Image_596.jpg', 'Face Mask Dataset/fm_train/Image_596.jpg'],
     ['Image_597.jpg', 'Face Mask Dataset/fm_train/Image_597.jpg'],
     ['Image_598.jpg', 'Face Mask Dataset/fm_train/Image_598.jpg'],
     ['Image_599.jpg', 'Face Mask Dataset/fm_train/Image_599.jpg'],
     ['Image_600.jpg', 'Face Mask Dataset/fm_train/Image_600.jpg'],
     ['Image_601.jpg', 'Face Mask Dataset/fm_train/Image_601.jpg'],
     ['Image_602.jpg', 'Face Mask Dataset/fm_train/Image_602.jpg'],
     ['Image_603.jpg', 'Face Mask Dataset/fm_train/Image_603.jpg'],
     ['Image_604.jpg', 'Face Mask Dataset/fm_train/Image_604.jpg'],
     ['Image_605.jpg', 'Face Mask Dataset/fm_train/Image_605.jpg'],
     ['Image_606.jpg', 'Face Mask Dataset/fm_train/Image_606.jpg'],
     ['Image_607.jpg', 'Face Mask Dataset/fm_train/Image_607.jpg'],
     ['Image_608.jpg', 'Face Mask Dataset/fm_train/Image_608.jpg'],
     ['Image_609.jpg', 'Face Mask Dataset/fm_train/Image_609.jpg'],
     ['Image_610.jpg', 'Face Mask Dataset/fm_train/Image_610.jpg'],
     ['Image_611.jpg', 'Face Mask Dataset/fm_train/Image_611.jpg'],
     ['Image_612.jpg', 'Face Mask Dataset/fm_train/Image_612.jpg'],
     ['Image_613.jpg', 'Face Mask Dataset/fm_train/Image_613.jpg'],
     ['Image_614.jpg', 'Face Mask Dataset/fm_train/Image_614.jpg'],
     ['Image_615.jpg', 'Face Mask Dataset/fm_train/Image_615.jpg'],
     ['Image_616.jpg', 'Face Mask Dataset/fm_train/Image_616.jpg'],
     ['Image_617.jpg', 'Face Mask Dataset/fm_train/Image_617.jpg'],
     ['Image_618.jpg', 'Face Mask Dataset/fm_train/Image_618.jpg'],
     ['Image_619.jpg', 'Face Mask Dataset/fm_train/Image_619.jpg'],
     ['Image_620.jpg', 'Face Mask Dataset/fm_train/Image_620.jpg'],
     ['Image_621.jpg', 'Face Mask Dataset/fm_train/Image_621.jpg'],
     ['Image_622.jpg', 'Face Mask Dataset/fm_train/Image_622.jpg'],
     ['Image_623.jpg', 'Face Mask Dataset/fm_train/Image_623.jpg'],
     ['Image_624.jpg', 'Face Mask Dataset/fm_train/Image_624.jpg'],
     ['Image_625.jpg', 'Face Mask Dataset/fm_train/Image_625.jpg'],
     ['Image_626.jpg', 'Face Mask Dataset/fm_train/Image_626.jpg'],
     ['Image_627.jpg', 'Face Mask Dataset/fm_train/Image_627.jpg'],
     ['Image_628.jpg', 'Face Mask Dataset/fm_train/Image_628.jpg'],
     ['Image_629.jpg', 'Face Mask Dataset/fm_train/Image_629.jpg'],
     ['Image_630.jpg', 'Face Mask Dataset/fm_train/Image_630.jpg'],
     ['Image_631.jpg', 'Face Mask Dataset/fm_train/Image_631.jpg'],
     ['Image_632.jpg', 'Face Mask Dataset/fm_train/Image_632.jpg'],
     ['Image_633.jpg', 'Face Mask Dataset/fm_train/Image_633.jpg'],
     ['Image_634.jpg', 'Face Mask Dataset/fm_train/Image_634.jpg'],
     ['Image_635.jpg', 'Face Mask Dataset/fm_train/Image_635.jpg'],
     ['Image_636.jpg', 'Face Mask Dataset/fm_train/Image_636.jpg'],
     ['Image_637.jpg', 'Face Mask Dataset/fm_train/Image_637.jpg'],
     ['Image_638.jpg', 'Face Mask Dataset/fm_train/Image_638.jpg'],
     ['Image_639.jpg', 'Face Mask Dataset/fm_train/Image_639.jpg'],
     ['Image_640.jpg', 'Face Mask Dataset/fm_train/Image_640.jpg'],
     ['Image_641.jpg', 'Face Mask Dataset/fm_train/Image_641.jpg'],
     ['Image_642.jpg', 'Face Mask Dataset/fm_train/Image_642.jpg'],
     ['Image_643.jpg', 'Face Mask Dataset/fm_train/Image_643.jpg'],
     ['Image_644.jpg', 'Face Mask Dataset/fm_train/Image_644.jpg'],
     ['Image_645.jpg', 'Face Mask Dataset/fm_train/Image_645.jpg'],
     ['Image_646.jpg', 'Face Mask Dataset/fm_train/Image_646.jpg'],
     ['Image_647.jpg', 'Face Mask Dataset/fm_train/Image_647.jpg'],
     ['Image_648.jpg', 'Face Mask Dataset/fm_train/Image_648.jpg'],
     ['Image_649.jpg', 'Face Mask Dataset/fm_train/Image_649.jpg'],
     ['Image_650.jpg', 'Face Mask Dataset/fm_train/Image_650.jpg'],
     ['Image_651.jpg', 'Face Mask Dataset/fm_train/Image_651.jpg'],
     ['Image_652.jpg', 'Face Mask Dataset/fm_train/Image_652.jpg'],
     ['Image_653.jpg', 'Face Mask Dataset/fm_train/Image_653.jpg'],
     ['Image_654.jpg', 'Face Mask Dataset/fm_train/Image_654.jpg'],
     ['Image_655.jpg', 'Face Mask Dataset/fm_train/Image_655.jpg'],
     ['Image_656.jpg', 'Face Mask Dataset/fm_train/Image_656.jpg'],
     ['Image_657.jpg', 'Face Mask Dataset/fm_train/Image_657.jpg'],
     ['Image_658.jpg', 'Face Mask Dataset/fm_train/Image_658.jpg'],
     ['Image_659.jpg', 'Face Mask Dataset/fm_train/Image_659.jpg'],
     ['Image_660.jpg', 'Face Mask Dataset/fm_train/Image_660.jpg'],
     ['Image_661.jpg', 'Face Mask Dataset/fm_train/Image_661.jpg'],
     ['Image_662.jpg', 'Face Mask Dataset/fm_train/Image_662.jpg'],
     ['Image_663.jpg', 'Face Mask Dataset/fm_train/Image_663.jpg'],
     ['Image_664.jpg', 'Face Mask Dataset/fm_train/Image_664.jpg'],
     ['Image_665.jpg', 'Face Mask Dataset/fm_train/Image_665.jpg'],
     ['Image_666.jpg', 'Face Mask Dataset/fm_train/Image_666.jpg'],
     ['Image_667.jpg', 'Face Mask Dataset/fm_train/Image_667.jpg'],
     ['Image_668.jpg', 'Face Mask Dataset/fm_train/Image_668.jpg'],
     ['Image_669.jpg', 'Face Mask Dataset/fm_train/Image_669.jpg'],
     ['Image_670.jpg', 'Face Mask Dataset/fm_train/Image_670.jpg'],
     ['Image_671.jpg', 'Face Mask Dataset/fm_train/Image_671.jpg'],
     ['Image_672.jpg', 'Face Mask Dataset/fm_train/Image_672.jpg'],
     ['Image_673.jpg', 'Face Mask Dataset/fm_train/Image_673.jpg'],
     ['Image_674.jpg', 'Face Mask Dataset/fm_train/Image_674.jpg'],
     ['Image_675.jpg', 'Face Mask Dataset/fm_train/Image_675.jpg'],
     ['Image_676.jpg', 'Face Mask Dataset/fm_train/Image_676.jpg'],
     ['Image_677.jpg', 'Face Mask Dataset/fm_train/Image_677.jpg'],
     ['Image_678.jpg', 'Face Mask Dataset/fm_train/Image_678.jpg'],
     ['Image_679.jpg', 'Face Mask Dataset/fm_train/Image_679.jpg'],
     ['Image_680.jpg', 'Face Mask Dataset/fm_train/Image_680.jpg'],
     ['Image_681.jpg', 'Face Mask Dataset/fm_train/Image_681.jpg'],
     ['Image_682.jpg', 'Face Mask Dataset/fm_train/Image_682.jpg'],
     ['Image_683.jpg', 'Face Mask Dataset/fm_train/Image_683.jpg'],
     ['Image_684.jpg', 'Face Mask Dataset/fm_train/Image_684.jpg'],
     ['Image_685.jpg', 'Face Mask Dataset/fm_train/Image_685.jpg'],
     ['Image_686.jpg', 'Face Mask Dataset/fm_train/Image_686.jpg'],
     ['Image_687.jpg', 'Face Mask Dataset/fm_train/Image_687.jpg'],
     ['Image_688.jpg', 'Face Mask Dataset/fm_train/Image_688.jpg'],
     ['Image_689.jpg', 'Face Mask Dataset/fm_train/Image_689.jpg'],
     ['Image_690.jpg', 'Face Mask Dataset/fm_train/Image_690.jpg'],
     ['Image_691.jpg', 'Face Mask Dataset/fm_train/Image_691.jpg'],
     ['Image_692.jpg', 'Face Mask Dataset/fm_train/Image_692.jpg'],
     ['Image_693.jpg', 'Face Mask Dataset/fm_train/Image_693.jpg'],
     ['Image_694.jpg', 'Face Mask Dataset/fm_train/Image_694.jpg'],
     ['Image_695.jpg', 'Face Mask Dataset/fm_train/Image_695.jpg'],
     ['Image_696.jpg', 'Face Mask Dataset/fm_train/Image_696.jpg'],
     ['Image_697.jpg', 'Face Mask Dataset/fm_train/Image_697.jpg'],
     ['Image_698.jpg', 'Face Mask Dataset/fm_train/Image_698.jpg'],
     ['Image_699.jpg', 'Face Mask Dataset/fm_train/Image_699.jpg'],
     ['Image_700.jpg', 'Face Mask Dataset/fm_train/Image_700.jpg'],
     ['Image_701.jpg', 'Face Mask Dataset/fm_train/Image_701.jpg'],
     ['Image_702.jpg', 'Face Mask Dataset/fm_train/Image_702.jpg'],
     ['Image_703.jpg', 'Face Mask Dataset/fm_train/Image_703.jpg'],
     ['Image_704.jpg', 'Face Mask Dataset/fm_train/Image_704.jpg'],
     ['Image_705.jpg', 'Face Mask Dataset/fm_train/Image_705.jpg'],
     ['Image_706.jpg', 'Face Mask Dataset/fm_train/Image_706.jpg'],
     ['Image_707.jpg', 'Face Mask Dataset/fm_train/Image_707.jpg'],
     ['Image_708.jpg', 'Face Mask Dataset/fm_train/Image_708.jpg'],
     ['Image_709.jpg', 'Face Mask Dataset/fm_train/Image_709.jpg'],
     ['Image_710.jpg', 'Face Mask Dataset/fm_train/Image_710.jpg'],
     ['Image_711.jpg', 'Face Mask Dataset/fm_train/Image_711.jpg'],
     ['Image_712.jpg', 'Face Mask Dataset/fm_train/Image_712.jpg'],
     ['Image_713.jpg', 'Face Mask Dataset/fm_train/Image_713.jpg'],
     ['Image_714.jpg', 'Face Mask Dataset/fm_train/Image_714.jpg'],
     ['Image_715.jpg', 'Face Mask Dataset/fm_train/Image_715.jpg'],
     ['Image_716.jpg', 'Face Mask Dataset/fm_train/Image_716.jpg'],
     ['Image_717.jpg', 'Face Mask Dataset/fm_train/Image_717.jpg'],
     ['Image_718.jpg', 'Face Mask Dataset/fm_train/Image_718.jpg'],
     ['Image_719.jpg', 'Face Mask Dataset/fm_train/Image_719.jpg'],
     ['Image_720.jpg', 'Face Mask Dataset/fm_train/Image_720.jpg'],
     ['Image_721.jpg', 'Face Mask Dataset/fm_train/Image_721.jpg'],
     ['Image_722.jpg', 'Face Mask Dataset/fm_train/Image_722.jpg'],
     ['Image_723.jpg', 'Face Mask Dataset/fm_train/Image_723.jpg'],
     ['Image_724.jpg', 'Face Mask Dataset/fm_train/Image_724.jpg'],
     ['Image_725.jpg', 'Face Mask Dataset/fm_train/Image_725.jpg'],
     ['Image_726.jpg', 'Face Mask Dataset/fm_train/Image_726.jpg'],
     ['Image_727.jpg', 'Face Mask Dataset/fm_train/Image_727.jpg'],
     ['Image_728.jpg', 'Face Mask Dataset/fm_train/Image_728.jpg'],
     ['Image_729.jpg', 'Face Mask Dataset/fm_train/Image_729.jpg'],
     ['Image_730.jpg', 'Face Mask Dataset/fm_train/Image_730.jpg'],
     ['Image_731.jpg', 'Face Mask Dataset/fm_train/Image_731.jpg'],
     ['Image_732.jpg', 'Face Mask Dataset/fm_train/Image_732.jpg'],
     ['Image_733.jpg', 'Face Mask Dataset/fm_train/Image_733.jpg'],
     ['Image_734.jpg', 'Face Mask Dataset/fm_train/Image_734.jpg'],
     ['Image_735.jpg', 'Face Mask Dataset/fm_train/Image_735.jpg'],
     ['Image_736.jpg', 'Face Mask Dataset/fm_train/Image_736.jpg'],
     ['Image_737.jpg', 'Face Mask Dataset/fm_train/Image_737.jpg'],
     ['Image_738.jpg', 'Face Mask Dataset/fm_train/Image_738.jpg'],
     ['Image_739.jpg', 'Face Mask Dataset/fm_train/Image_739.jpg'],
     ['Image_740.jpg', 'Face Mask Dataset/fm_train/Image_740.jpg'],
     ['Image_741.jpg', 'Face Mask Dataset/fm_train/Image_741.jpg'],
     ['Image_742.jpg', 'Face Mask Dataset/fm_train/Image_742.jpg'],
     ['Image_743.jpg', 'Face Mask Dataset/fm_train/Image_743.jpg'],
     ['Image_744.jpg', 'Face Mask Dataset/fm_train/Image_744.jpg'],
     ['Image_745.jpg', 'Face Mask Dataset/fm_train/Image_745.jpg'],
     ['Image_746.jpg', 'Face Mask Dataset/fm_train/Image_746.jpg'],
     ['Image_747.jpg', 'Face Mask Dataset/fm_train/Image_747.jpg'],
     ['Image_748.jpg', 'Face Mask Dataset/fm_train/Image_748.jpg'],
     ['Image_749.jpg', 'Face Mask Dataset/fm_train/Image_749.jpg'],
     ['Image_750.jpg', 'Face Mask Dataset/fm_train/Image_750.jpg'],
     ['Image_751.jpg', 'Face Mask Dataset/fm_train/Image_751.jpg'],
     ['Image_752.jpg', 'Face Mask Dataset/fm_train/Image_752.jpg'],
     ['Image_753.jpg', 'Face Mask Dataset/fm_train/Image_753.jpg'],
     ['Image_754.jpg', 'Face Mask Dataset/fm_train/Image_754.jpg'],
     ['Image_755.jpg', 'Face Mask Dataset/fm_train/Image_755.jpg'],
     ['Image_756.jpg', 'Face Mask Dataset/fm_train/Image_756.jpg'],
     ['Image_757.jpg', 'Face Mask Dataset/fm_train/Image_757.jpg'],
     ['Image_758.jpg', 'Face Mask Dataset/fm_train/Image_758.jpg'],
     ['Image_759.jpg', 'Face Mask Dataset/fm_train/Image_759.jpg'],
     ['Image_760.jpg', 'Face Mask Dataset/fm_train/Image_760.jpg'],
     ['Image_761.jpg', 'Face Mask Dataset/fm_train/Image_761.jpg'],
     ['Image_762.jpg', 'Face Mask Dataset/fm_train/Image_762.jpg'],
     ['Image_763.jpg', 'Face Mask Dataset/fm_train/Image_763.jpg'],
     ['Image_764.jpg', 'Face Mask Dataset/fm_train/Image_764.jpg'],
     ['Image_765.jpg', 'Face Mask Dataset/fm_train/Image_765.jpg'],
     ['Image_766.jpg', 'Face Mask Dataset/fm_train/Image_766.jpg'],
     ['Image_767.jpg', 'Face Mask Dataset/fm_train/Image_767.jpg'],
     ['Image_768.jpg', 'Face Mask Dataset/fm_train/Image_768.jpg'],
     ['Image_769.jpg', 'Face Mask Dataset/fm_train/Image_769.jpg'],
     ['Image_770.jpg', 'Face Mask Dataset/fm_train/Image_770.jpg'],
     ['Image_771.jpg', 'Face Mask Dataset/fm_train/Image_771.jpg'],
     ['Image_772.jpg', 'Face Mask Dataset/fm_train/Image_772.jpg'],
     ['Image_773.jpg', 'Face Mask Dataset/fm_train/Image_773.jpg'],
     ['Image_774.jpg', 'Face Mask Dataset/fm_train/Image_774.jpg'],
     ['Image_775.jpg', 'Face Mask Dataset/fm_train/Image_775.jpg'],
     ['Image_776.jpg', 'Face Mask Dataset/fm_train/Image_776.jpg'],
     ['Image_777.jpg', 'Face Mask Dataset/fm_train/Image_777.jpg'],
     ['Image_778.jpg', 'Face Mask Dataset/fm_train/Image_778.jpg'],
     ['Image_779.jpg', 'Face Mask Dataset/fm_train/Image_779.jpg'],
     ['Image_780.jpg', 'Face Mask Dataset/fm_train/Image_780.jpg'],
     ['Image_781.jpg', 'Face Mask Dataset/fm_train/Image_781.jpg'],
     ['Image_782.jpg', 'Face Mask Dataset/fm_train/Image_782.jpg'],
     ['Image_783.jpg', 'Face Mask Dataset/fm_train/Image_783.jpg'],
     ['Image_784.jpg', 'Face Mask Dataset/fm_train/Image_784.jpg'],
     ['Image_785.jpg', 'Face Mask Dataset/fm_train/Image_785.jpg'],
     ['Image_786.jpg', 'Face Mask Dataset/fm_train/Image_786.jpg'],
     ['Image_787.jpg', 'Face Mask Dataset/fm_train/Image_787.jpg'],
     ['Image_788.jpg', 'Face Mask Dataset/fm_train/Image_788.jpg'],
     ['Image_789.jpg', 'Face Mask Dataset/fm_train/Image_789.jpg'],
     ['Image_790.jpg', 'Face Mask Dataset/fm_train/Image_790.jpg'],
     ['Image_791.jpg', 'Face Mask Dataset/fm_train/Image_791.jpg'],
     ['Image_792.jpg', 'Face Mask Dataset/fm_train/Image_792.jpg'],
     ['Image_793.jpg', 'Face Mask Dataset/fm_train/Image_793.jpg'],
     ['Image_794.jpg', 'Face Mask Dataset/fm_train/Image_794.jpg'],
     ['Image_795.jpg', 'Face Mask Dataset/fm_train/Image_795.jpg'],
     ['Image_796.jpg', 'Face Mask Dataset/fm_train/Image_796.jpg'],
     ['Image_797.jpg', 'Face Mask Dataset/fm_train/Image_797.jpg'],
     ['Image_798.jpg', 'Face Mask Dataset/fm_train/Image_798.jpg'],
     ['Image_799.jpg', 'Face Mask Dataset/fm_train/Image_799.jpg'],
     ['Image_800.jpg', 'Face Mask Dataset/fm_train/Image_800.jpg'],
     ['Image_801.jpg', 'Face Mask Dataset/fm_train/Image_801.jpg'],
     ['Image_802.jpg', 'Face Mask Dataset/fm_train/Image_802.jpg'],
     ['Image_803.jpg', 'Face Mask Dataset/fm_train/Image_803.jpg'],
     ['Image_804.jpg', 'Face Mask Dataset/fm_train/Image_804.jpg'],
     ['Image_805.jpg', 'Face Mask Dataset/fm_train/Image_805.jpg'],
     ['Image_806.jpg', 'Face Mask Dataset/fm_train/Image_806.jpg'],
     ['Image_807.jpg', 'Face Mask Dataset/fm_train/Image_807.jpg'],
     ['Image_808.jpg', 'Face Mask Dataset/fm_train/Image_808.jpg'],
     ['Image_809.jpg', 'Face Mask Dataset/fm_train/Image_809.jpg'],
     ['Image_810.jpg', 'Face Mask Dataset/fm_train/Image_810.jpg'],
     ['Image_811.jpg', 'Face Mask Dataset/fm_train/Image_811.jpg'],
     ['Image_812.jpg', 'Face Mask Dataset/fm_train/Image_812.jpg'],
     ['Image_813.jpg', 'Face Mask Dataset/fm_train/Image_813.jpg'],
     ['Image_814.jpg', 'Face Mask Dataset/fm_train/Image_814.jpg'],
     ['Image_815.jpg', 'Face Mask Dataset/fm_train/Image_815.jpg'],
     ['Image_816.jpg', 'Face Mask Dataset/fm_train/Image_816.jpg'],
     ['Image_817.jpg', 'Face Mask Dataset/fm_train/Image_817.jpg'],
     ['Image_818.jpg', 'Face Mask Dataset/fm_train/Image_818.jpg'],
     ['Image_819.jpg', 'Face Mask Dataset/fm_train/Image_819.jpg'],
     ['Image_820.jpg', 'Face Mask Dataset/fm_train/Image_820.jpg'],
     ['Image_821.jpg', 'Face Mask Dataset/fm_train/Image_821.jpg'],
     ['Image_822.jpg', 'Face Mask Dataset/fm_train/Image_822.jpg'],
     ['Image_823.jpg', 'Face Mask Dataset/fm_train/Image_823.jpg'],
     ['Image_824.jpg', 'Face Mask Dataset/fm_train/Image_824.jpg'],
     ['Image_825.jpg', 'Face Mask Dataset/fm_train/Image_825.jpg'],
     ['Image_826.jpg', 'Face Mask Dataset/fm_train/Image_826.jpg'],
     ['Image_827.jpg', 'Face Mask Dataset/fm_train/Image_827.jpg'],
     ['Image_828.jpg', 'Face Mask Dataset/fm_train/Image_828.jpg'],
     ['Image_829.jpg', 'Face Mask Dataset/fm_train/Image_829.jpg'],
     ['Image_830.jpg', 'Face Mask Dataset/fm_train/Image_830.jpg'],
     ['Image_831.jpg', 'Face Mask Dataset/fm_train/Image_831.jpg'],
     ['Image_832.jpg', 'Face Mask Dataset/fm_train/Image_832.jpg'],
     ['Image_833.jpg', 'Face Mask Dataset/fm_train/Image_833.jpg'],
     ['Image_834.jpg', 'Face Mask Dataset/fm_train/Image_834.jpg'],
     ['Image_835.jpg', 'Face Mask Dataset/fm_train/Image_835.jpg'],
     ['Image_836.jpg', 'Face Mask Dataset/fm_train/Image_836.jpg'],
     ['Image_837.jpg', 'Face Mask Dataset/fm_train/Image_837.jpg'],
     ['Image_838.jpg', 'Face Mask Dataset/fm_train/Image_838.jpg'],
     ['Image_839.jpg', 'Face Mask Dataset/fm_train/Image_839.jpg'],
     ['Image_840.jpg', 'Face Mask Dataset/fm_train/Image_840.jpg'],
     ['Image_841.jpg', 'Face Mask Dataset/fm_train/Image_841.jpg'],
     ['Image_842.jpg', 'Face Mask Dataset/fm_train/Image_842.jpg'],
     ['Image_843.jpg', 'Face Mask Dataset/fm_train/Image_843.jpg'],
     ['Image_844.jpg', 'Face Mask Dataset/fm_train/Image_844.jpg'],
     ['Image_845.jpg', 'Face Mask Dataset/fm_train/Image_845.jpg'],
     ['Image_846.jpg', 'Face Mask Dataset/fm_train/Image_846.jpg'],
     ['Image_847.jpg', 'Face Mask Dataset/fm_train/Image_847.jpg'],
     ['Image_848.jpg', 'Face Mask Dataset/fm_train/Image_848.jpg'],
     ['Image_849.jpg', 'Face Mask Dataset/fm_train/Image_849.jpg'],
     ['Image_850.jpg', 'Face Mask Dataset/fm_train/Image_850.jpg'],
     ['Image_851.jpg', 'Face Mask Dataset/fm_train/Image_851.jpg'],
     ['Image_852.jpg', 'Face Mask Dataset/fm_train/Image_852.jpg'],
     ['Image_853.jpg', 'Face Mask Dataset/fm_train/Image_853.jpg'],
     ['Image_854.jpg', 'Face Mask Dataset/fm_train/Image_854.jpg'],
     ['Image_855.jpg', 'Face Mask Dataset/fm_train/Image_855.jpg'],
     ['Image_856.jpg', 'Face Mask Dataset/fm_train/Image_856.jpg'],
     ['Image_857.jpg', 'Face Mask Dataset/fm_train/Image_857.jpg'],
     ['Image_858.jpg', 'Face Mask Dataset/fm_train/Image_858.jpg'],
     ['Image_859.jpg', 'Face Mask Dataset/fm_train/Image_859.jpg'],
     ['Image_860.jpg', 'Face Mask Dataset/fm_train/Image_860.jpg'],
     ['Image_861.jpg', 'Face Mask Dataset/fm_train/Image_861.jpg'],
     ['Image_862.jpg', 'Face Mask Dataset/fm_train/Image_862.jpg'],
     ['Image_863.jpg', 'Face Mask Dataset/fm_train/Image_863.jpg'],
     ['Image_864.jpg', 'Face Mask Dataset/fm_train/Image_864.jpg'],
     ['Image_865.jpg', 'Face Mask Dataset/fm_train/Image_865.jpg'],
     ['Image_866.jpg', 'Face Mask Dataset/fm_train/Image_866.jpg'],
     ['Image_867.jpg', 'Face Mask Dataset/fm_train/Image_867.jpg'],
     ['Image_868.jpg', 'Face Mask Dataset/fm_train/Image_868.jpg'],
     ['Image_869.jpg', 'Face Mask Dataset/fm_train/Image_869.jpg'],
     ['Image_870.jpg', 'Face Mask Dataset/fm_train/Image_870.jpg'],
     ['Image_871.jpg', 'Face Mask Dataset/fm_train/Image_871.jpg'],
     ['Image_872.jpg', 'Face Mask Dataset/fm_train/Image_872.jpg'],
     ['Image_873.jpg', 'Face Mask Dataset/fm_train/Image_873.jpg'],
     ['Image_874.jpg', 'Face Mask Dataset/fm_train/Image_874.jpg'],
     ['Image_875.jpg', 'Face Mask Dataset/fm_train/Image_875.jpg'],
     ['Image_876.jpg', 'Face Mask Dataset/fm_train/Image_876.jpg'],
     ['Image_877.jpg', 'Face Mask Dataset/fm_train/Image_877.jpg'],
     ['Image_878.jpg', 'Face Mask Dataset/fm_train/Image_878.jpg'],
     ['Image_879.jpg', 'Face Mask Dataset/fm_train/Image_879.jpg'],
     ['Image_880.jpg', 'Face Mask Dataset/fm_train/Image_880.jpg'],
     ['Image_881.jpg', 'Face Mask Dataset/fm_train/Image_881.jpg'],
     ['Image_882.jpg', 'Face Mask Dataset/fm_train/Image_882.jpg'],
     ['Image_883.jpg', 'Face Mask Dataset/fm_train/Image_883.jpg'],
     ['Image_884.jpg', 'Face Mask Dataset/fm_train/Image_884.jpg'],
     ['Image_885.jpg', 'Face Mask Dataset/fm_train/Image_885.jpg'],
     ['Image_886.jpg', 'Face Mask Dataset/fm_train/Image_886.jpg'],
     ['Image_887.jpg', 'Face Mask Dataset/fm_train/Image_887.jpg'],
     ['Image_888.jpg', 'Face Mask Dataset/fm_train/Image_888.jpg'],
     ['Image_889.jpg', 'Face Mask Dataset/fm_train/Image_889.jpg'],
     ['Image_890.jpg', 'Face Mask Dataset/fm_train/Image_890.jpg'],
     ['Image_891.jpg', 'Face Mask Dataset/fm_train/Image_891.jpg'],
     ['Image_892.jpg', 'Face Mask Dataset/fm_train/Image_892.jpg'],
     ['Image_893.jpg', 'Face Mask Dataset/fm_train/Image_893.jpg'],
     ['Image_894.jpg', 'Face Mask Dataset/fm_train/Image_894.jpg'],
     ['Image_895.jpg', 'Face Mask Dataset/fm_train/Image_895.jpg'],
     ['Image_896.jpg', 'Face Mask Dataset/fm_train/Image_896.jpg'],
     ['Image_897.jpg', 'Face Mask Dataset/fm_train/Image_897.jpg'],
     ['Image_898.jpg', 'Face Mask Dataset/fm_train/Image_898.jpg'],
     ['Image_899.jpg', 'Face Mask Dataset/fm_train/Image_899.jpg'],
     ['Image_900.jpg', 'Face Mask Dataset/fm_train/Image_900.jpg'],
     ['Image_901.jpg', 'Face Mask Dataset/fm_train/Image_901.jpg'],
     ['Image_902.jpg', 'Face Mask Dataset/fm_train/Image_902.jpg'],
     ['Image_903.jpg', 'Face Mask Dataset/fm_train/Image_903.jpg'],
     ['Image_904.jpg', 'Face Mask Dataset/fm_train/Image_904.jpg'],
     ['Image_905.jpg', 'Face Mask Dataset/fm_train/Image_905.jpg'],
     ['Image_906.jpg', 'Face Mask Dataset/fm_train/Image_906.jpg'],
     ['Image_907.jpg', 'Face Mask Dataset/fm_train/Image_907.jpg'],
     ['Image_908.jpg', 'Face Mask Dataset/fm_train/Image_908.jpg'],
     ['Image_909.jpg', 'Face Mask Dataset/fm_train/Image_909.jpg'],
     ['Image_910.jpg', 'Face Mask Dataset/fm_train/Image_910.jpg'],
     ['Image_911.jpg', 'Face Mask Dataset/fm_train/Image_911.jpg'],
     ['Image_912.jpg', 'Face Mask Dataset/fm_train/Image_912.jpg'],
     ['Image_913.jpg', 'Face Mask Dataset/fm_train/Image_913.jpg'],
     ['Image_914.jpg', 'Face Mask Dataset/fm_train/Image_914.jpg'],
     ['Image_915.jpg', 'Face Mask Dataset/fm_train/Image_915.jpg'],
     ['Image_916.jpg', 'Face Mask Dataset/fm_train/Image_916.jpg'],
     ['Image_917.jpg', 'Face Mask Dataset/fm_train/Image_917.jpg'],
     ['Image_918.jpg', 'Face Mask Dataset/fm_train/Image_918.jpg'],
     ['Image_919.jpg', 'Face Mask Dataset/fm_train/Image_919.jpg'],
     ['Image_920.jpg', 'Face Mask Dataset/fm_train/Image_920.jpg'],
     ['Image_921.jpg', 'Face Mask Dataset/fm_train/Image_921.jpg'],
     ['Image_922.jpg', 'Face Mask Dataset/fm_train/Image_922.jpg'],
     ['Image_923.jpg', 'Face Mask Dataset/fm_train/Image_923.jpg'],
     ['Image_924.jpg', 'Face Mask Dataset/fm_train/Image_924.jpg'],
     ['Image_925.jpg', 'Face Mask Dataset/fm_train/Image_925.jpg'],
     ['Image_926.jpg', 'Face Mask Dataset/fm_train/Image_926.jpg'],
     ['Image_927.jpg', 'Face Mask Dataset/fm_train/Image_927.jpg'],
     ['Image_928.jpg', 'Face Mask Dataset/fm_train/Image_928.jpg'],
     ['Image_929.jpg', 'Face Mask Dataset/fm_train/Image_929.jpg'],
     ['Image_930.jpg', 'Face Mask Dataset/fm_train/Image_930.jpg'],
     ['Image_931.jpg', 'Face Mask Dataset/fm_train/Image_931.jpg'],
     ['Image_932.jpg', 'Face Mask Dataset/fm_train/Image_932.jpg'],
     ['Image_933.jpg', 'Face Mask Dataset/fm_train/Image_933.jpg'],
     ['Image_934.jpg', 'Face Mask Dataset/fm_train/Image_934.jpg'],
     ['Image_935.jpg', 'Face Mask Dataset/fm_train/Image_935.jpg'],
     ['Image_936.jpg', 'Face Mask Dataset/fm_train/Image_936.jpg'],
     ['Image_937.jpg', 'Face Mask Dataset/fm_train/Image_937.jpg'],
     ['Image_938.jpg', 'Face Mask Dataset/fm_train/Image_938.jpg'],
     ['Image_939.jpg', 'Face Mask Dataset/fm_train/Image_939.jpg'],
     ['Image_940.jpg', 'Face Mask Dataset/fm_train/Image_940.jpg'],
     ['Image_941.jpg', 'Face Mask Dataset/fm_train/Image_941.jpg'],
     ['Image_942.jpg', 'Face Mask Dataset/fm_train/Image_942.jpg'],
     ['Image_943.jpg', 'Face Mask Dataset/fm_train/Image_943.jpg'],
     ['Image_944.jpg', 'Face Mask Dataset/fm_train/Image_944.jpg'],
     ['Image_945.jpg', 'Face Mask Dataset/fm_train/Image_945.jpg'],
     ['Image_946.jpg', 'Face Mask Dataset/fm_train/Image_946.jpg'],
     ['Image_947.jpg', 'Face Mask Dataset/fm_train/Image_947.jpg'],
     ['Image_948.jpg', 'Face Mask Dataset/fm_train/Image_948.jpg'],
     ['Image_949.jpg', 'Face Mask Dataset/fm_train/Image_949.jpg'],
     ['Image_950.jpg', 'Face Mask Dataset/fm_train/Image_950.jpg'],
     ['Image_951.jpg', 'Face Mask Dataset/fm_train/Image_951.jpg'],
     ['Image_952.jpg', 'Face Mask Dataset/fm_train/Image_952.jpg'],
     ['Image_953.jpg', 'Face Mask Dataset/fm_train/Image_953.jpg'],
     ['Image_954.jpg', 'Face Mask Dataset/fm_train/Image_954.jpg'],
     ['Image_955.jpg', 'Face Mask Dataset/fm_train/Image_955.jpg'],
     ['Image_956.jpg', 'Face Mask Dataset/fm_train/Image_956.jpg'],
     ['Image_957.jpg', 'Face Mask Dataset/fm_train/Image_957.jpg'],
     ['Image_958.jpg', 'Face Mask Dataset/fm_train/Image_958.jpg'],
     ['Image_959.jpg', 'Face Mask Dataset/fm_train/Image_959.jpg'],
     ['Image_960.jpg', 'Face Mask Dataset/fm_train/Image_960.jpg'],
     ['Image_961.jpg', 'Face Mask Dataset/fm_train/Image_961.jpg'],
     ['Image_962.jpg', 'Face Mask Dataset/fm_train/Image_962.jpg'],
     ['Image_963.jpg', 'Face Mask Dataset/fm_train/Image_963.jpg'],
     ['Image_964.jpg', 'Face Mask Dataset/fm_train/Image_964.jpg'],
     ['Image_965.jpg', 'Face Mask Dataset/fm_train/Image_965.jpg'],
     ['Image_966.jpg', 'Face Mask Dataset/fm_train/Image_966.jpg'],
     ['Image_967.jpg', 'Face Mask Dataset/fm_train/Image_967.jpg'],
     ['Image_968.jpg', 'Face Mask Dataset/fm_train/Image_968.jpg'],
     ['Image_969.jpg', 'Face Mask Dataset/fm_train/Image_969.jpg'],
     ['Image_970.jpg', 'Face Mask Dataset/fm_train/Image_970.jpg'],
     ['Image_971.jpg', 'Face Mask Dataset/fm_train/Image_971.jpg'],
     ['Image_972.jpg', 'Face Mask Dataset/fm_train/Image_972.jpg'],
     ['Image_973.jpg', 'Face Mask Dataset/fm_train/Image_973.jpg'],
     ['Image_974.jpg', 'Face Mask Dataset/fm_train/Image_974.jpg'],
     ['Image_975.jpg', 'Face Mask Dataset/fm_train/Image_975.jpg'],
     ['Image_976.jpg', 'Face Mask Dataset/fm_train/Image_976.jpg'],
     ['Image_977.jpg', 'Face Mask Dataset/fm_train/Image_977.jpg'],
     ['Image_978.jpg', 'Face Mask Dataset/fm_train/Image_978.jpg'],
     ['Image_979.jpg', 'Face Mask Dataset/fm_train/Image_979.jpg'],
     ['Image_980.jpg', 'Face Mask Dataset/fm_train/Image_980.jpg'],
     ['Image_981.jpg', 'Face Mask Dataset/fm_train/Image_981.jpg'],
     ['Image_982.jpg', 'Face Mask Dataset/fm_train/Image_982.jpg'],
     ['Image_983.jpg', 'Face Mask Dataset/fm_train/Image_983.jpg'],
     ['Image_984.jpg', 'Face Mask Dataset/fm_train/Image_984.jpg'],
     ['Image_985.jpg', 'Face Mask Dataset/fm_train/Image_985.jpg'],
     ['Image_986.jpg', 'Face Mask Dataset/fm_train/Image_986.jpg'],
     ['Image_987.jpg', 'Face Mask Dataset/fm_train/Image_987.jpg'],
     ['Image_988.jpg', 'Face Mask Dataset/fm_train/Image_988.jpg'],
     ['Image_989.jpg', 'Face Mask Dataset/fm_train/Image_989.jpg'],
     ['Image_990.jpg', 'Face Mask Dataset/fm_train/Image_990.jpg'],
     ['Image_991.jpg', 'Face Mask Dataset/fm_train/Image_991.jpg'],
     ['Image_992.jpg', 'Face Mask Dataset/fm_train/Image_992.jpg'],
     ['Image_993.jpg', 'Face Mask Dataset/fm_train/Image_993.jpg'],
     ['Image_994.jpg', 'Face Mask Dataset/fm_train/Image_994.jpg'],
     ['Image_995.jpg', 'Face Mask Dataset/fm_train/Image_995.jpg'],
     ['Image_996.jpg', 'Face Mask Dataset/fm_train/Image_996.jpg'],
     ['Image_997.jpg', 'Face Mask Dataset/fm_train/Image_997.jpg'],
     ['Image_998.jpg', 'Face Mask Dataset/fm_train/Image_998.jpg'],
     ['Image_999.jpg', 'Face Mask Dataset/fm_train/Image_999.jpg'],
     ['Image_1000.jpg', 'Face Mask Dataset/fm_train/Image_1000.jpg'],
     ...]



#### Number of labels equals number of images


```python
if len(labels) == len(file_paths):
    print('Number of labels i.e. ', len(labels), 'matches the number of filenames i.e. ', len(file_paths))
else:
    print('Number of labels does not match the number of filenames')
```

    Number of labels i.e.  11264 matches the number of filenames i.e.  11264


#### Sample image from data


```python
from IPython.display import Image
Image('Face Mask Dataset/fm_train/Image_998.jpg')
```




    
![jpeg](Face-Mask-Detection-CNN-MLP-VGG99_files/Face-Mask-Detection-CNN-MLP-VGG99_38_0.jpg)
    



#### Converting the file_paths to dataframe


```python
images = pd.DataFrame(file_paths, columns=['filename', 'filepaths'])
images.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filename</th>
      <th>filepaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Image_1.jpg</td>
      <td>Face Mask Dataset/fm_train/Image_1.jpg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Image_2.jpg</td>
      <td>Face Mask Dataset/fm_train/Image_2.jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Image_3.jpg</td>
      <td>Face Mask Dataset/fm_train/Image_3.jpg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Image_4.jpg</td>
      <td>Face Mask Dataset/fm_train/Image_4.jpg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Image_5.jpg</td>
      <td>Face Mask Dataset/fm_train/Image_5.jpg</td>
    </tr>
  </tbody>
</table>
</div>



#### Combining labels with images


```python
train_data = pd.merge(images, labels, how = 'inner', on = 'filename')
train_data.head()  
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filename</th>
      <th>filepaths</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Image_1.jpg</td>
      <td>Face Mask Dataset/fm_train/Image_1.jpg</td>
      <td>without_mask</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Image_2.jpg</td>
      <td>Face Mask Dataset/fm_train/Image_2.jpg</td>
      <td>without_mask</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Image_3.jpg</td>
      <td>Face Mask Dataset/fm_train/Image_3.jpg</td>
      <td>without_mask</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Image_4.jpg</td>
      <td>Face Mask Dataset/fm_train/Image_4.jpg</td>
      <td>without_mask</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Image_5.jpg</td>
      <td>Face Mask Dataset/fm_train/Image_5.jpg</td>
      <td>without_mask</td>
    </tr>
  </tbody>
</table>
</div>



### Data Preprocessing


```python
import cv2
data = []     
image_size = 100      # image size taken is 100 here. one can take other size too
for i in range(len(train_data)):

    img_array = cv2.imread(train_data['filepaths'][i], cv2.IMREAD_GRAYSCALE)   # converting the image to gray scale

    new_img_array = cv2.resize(img_array, (image_size, image_size))      # resizing the image array

    # encoding the labels. with_mask = 1 and without_mask = 0
    if train_data['label'][i] == 'with_mask':
        data.append([new_img_array, 1])
    else:
        data.append([new_img_array, 0])
```


```python
# image pixels of a image
data[0]
```




    [array([[119, 119, 119, ...,  36,  44,  48],
            [126, 127, 127, ...,  43,  36,  45],
            [132, 132, 133, ...,  42,  35,  41],
            ...,
            [129, 117, 140, ...,  75,  51,  25],
            [119, 117, 136, ...,  79,  59,  31],
            [118, 129, 137, ...,  83,  65,  37]], dtype=uint8),
     0]




```python
data = np.array(data)
data[0][0].shape
```




    (100, 100)



#### Shuffling

Images needed to shuffle as first half has data with mask and second half has without mask data. Model needs to train with both the categories in order to detect them.


```python
np.random.shuffle(data)
```


```python
import matplotlib.pyplot as plt
```


```python
#view the images
num_rows, num_cols = 2, 5
f, ax = plt.subplots(num_rows, num_cols, figsize=(12,5),
                     gridspec_kw={'wspace':0.03, 'hspace':0.01}, 
                     squeeze=True)

for r in range(num_rows):
    for c in range(num_cols):
      
        image_index = r * 100 + c
        ax[r,c].axis("off")
        ax[r,c].imshow( data[image_index][0], cmap='gray')
        if data[image_index][1] == 0:
            ax[r,c].set_title('without_mask')
        else:
            ax[r,c].set_title('with_mask')
plt.show()
plt.close()
```


    
![png](Face-Mask-Detection-CNN-MLP-VGG99_files/Face-Mask-Detection-CNN-MLP-VGG99_51_0.png)
    


### Separating image and labels


```python
x = []
y = []
for image in data:
    x.append(image[0])
    y.append(image[1])

# converting x & y to numpy array as they are list
x = np.array(x)
y = np.array(y)
```


```python
np.unique(y, return_counts=True)
```




    (array([0, 1]), array([5632, 5632]))




```python
x = x / 255

# Why divided by 255?
# The pixel value lie in the range 0 - 255 representing the RGB (Red Green Blue) value.
```


```python
# spliting the data
X_train, X_val, y_train, y_val = train_test_split(x,y,test_size=0.3, random_state = 42)
```

### Training Model

- Fully connected layers uses softmax activation function in output layer
- Every node in previous layer connected to every node in next layer (I/P -> Hidden -> O/P)
- Convolution and Pooling model used to identify high level image features 
- Based on features MLP model classifies the image using training dataset



```python
import warnings

```


```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100)),    # flattening the image
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size = 20)
```

    Epoch 1/10
    395/395 [==============================] - 1s 2ms/step - loss: 0.7028 - accuracy: 0.6741
    Epoch 2/10
    395/395 [==============================] - 1s 2ms/step - loss: 0.4171 - accuracy: 0.8193
    Epoch 3/10
    395/395 [==============================] - 1s 2ms/step - loss: 0.3598 - accuracy: 0.8435
    Epoch 4/10
    395/395 [==============================] - 1s 2ms/step - loss: 0.3278 - accuracy: 0.8576
    Epoch 5/10
    395/395 [==============================] - 1s 2ms/step - loss: 0.3103 - accuracy: 0.8723
    Epoch 6/10
    395/395 [==============================] - 1s 2ms/step - loss: 0.2956 - accuracy: 0.8734
    Epoch 7/10
    395/395 [==============================] - 1s 3ms/step - loss: 0.2784 - accuracy: 0.8838
    Epoch 8/10
    395/395 [==============================] - 1s 3ms/step - loss: 0.3067 - accuracy: 0.8656
    Epoch 9/10
    395/395 [==============================] - 1s 3ms/step - loss: 0.2837 - accuracy: 0.8789
    Epoch 10/10
    395/395 [==============================] - 1s 3ms/step - loss: 0.2580 - accuracy: 0.8873





    <tensorflow.python.keras.callbacks.History at 0x7f7f3d0e4190>



### MLP Model Validation


```python
model.evaluate(X_val, y_val)
```

    106/106 [==============================] - 0s 1ms/step - loss: 0.2815 - accuracy: 0.8781





    [0.2815481424331665, 0.8781065344810486]



- A two layer backpropagation network with hidden nodes proven to be universal approximator
MLP do not make any assumptions regarding probability 
- Underlying probability density function
- Any probabilistic information about pattern classes 
Required decision function yield directly via training 
- Preferred techniques for gesture recognition 


## 2.2 VGG19

The VGGNet architecture is based on CNN. It is said to one of the best architectures in use today

The VGG19 architecture – it consists of a series of Convolution layers followed by a max pooling layer

In the end there is a fully connected Neural Net with ReLU and a soft max activation which can have up to 1000 classes

<img src="Face Mask Dataset/figures/VGG Arch 1.png">

### The difference in the architectures of VGG16 and VGG19

<img src="Face Mask Dataset/figures/VGG 16 VGG 19.png">

The only difference is the depth to which is are built are different. Some classification problems may perform better with VGG16 while other may perform better with VGG19


```python
#Load train and test set
train_dir = './Face Mask Dataset/Train'
test_dir = './Face Mask Dataset/Test'
val_dir = './Face Mask Dataset/Validation'
```


```python
# Data augmentation

train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, zoom_range=0.2,shear_range=0.2)
train_generator = train_datagen.flow_from_directory(directory=train_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

val_datagen = ImageDataGenerator(rescale=1.0/255)
val_generator = val_datagen.flow_from_directory(directory=val_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(directory=test_dir,target_size=(128,128),class_mode='categorical',batch_size=32)
```

    Found 10000 images belonging to 2 classes.
    Found 800 images belonging to 2 classes.
    Found 992 images belonging to 2 classes.


For building the model – 

**weights**: ImageNet (weights which have been annotated through Image Net have been used. ImageNet is a dataset where images are pre-annotated)

**input_shape**: 128x128 (3 denotes RGB) 

**include_top**: False
input image is resized and the output classes will be 2 (1000 is default)

**activation**: sigmoid (only 2 classes)


```python
vgg19 = VGG19(weights='imagenet',include_top=False,input_shape=(128,128,3))

for layer in vgg19.layers:
    layer.trainable = False
    
model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(2,activation='sigmoid'))
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    vgg19 (Functional)           (None, 4, 4, 512)         20024384  
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 8192)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 2)                 16386     
    =================================================================
    Total params: 20,040,770
    Trainable params: 16,386
    Non-trainable params: 20,024,384
    _________________________________________________________________



```python
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics ="accuracy")
```


```python
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=len(train_generator)//32,
                              epochs=20,validation_data=val_generator,
                              validation_steps=len(val_generator)//32)
```

    Epoch 1/20
    9/9 [==============================] - 20s 2s/step - loss: 0.6771 - accuracy: 0.6314
    Epoch 2/20
    9/9 [==============================] - 20s 2s/step - loss: 0.3053 - accuracy: 0.8543
    Epoch 3/20
    9/9 [==============================] - 21s 2s/step - loss: 0.1789 - accuracy: 0.9349
    Epoch 4/20
    9/9 [==============================] - 21s 2s/step - loss: 0.1075 - accuracy: 0.9741
    Epoch 5/20
    9/9 [==============================] - 22s 2s/step - loss: 0.0890 - accuracy: 0.9722
    Epoch 6/20
    9/9 [==============================] - 21s 2s/step - loss: 0.0685 - accuracy: 0.9805
    Epoch 7/20
    9/9 [==============================] - 21s 2s/step - loss: 0.0607 - accuracy: 0.9824
    Epoch 8/20
    9/9 [==============================] - 21s 2s/step - loss: 0.0941 - accuracy: 0.9426
    Epoch 9/20
    9/9 [==============================] - 20s 2s/step - loss: 0.0798 - accuracy: 0.9933
    Epoch 10/20
    9/9 [==============================] - 21s 2s/step - loss: 0.0777 - accuracy: 0.9798
    Epoch 11/20
    9/9 [==============================] - 21s 2s/step - loss: 0.0917 - accuracy: 0.9674
    Epoch 12/20
    9/9 [==============================] - 22s 2s/step - loss: 0.0699 - accuracy: 0.9770
    Epoch 13/20
    9/9 [==============================] - 22s 2s/step - loss: 0.0775 - accuracy: 0.9629
    Epoch 14/20
    9/9 [==============================] - 22s 2s/step - loss: 0.1287 - accuracy: 0.9498
    Epoch 15/20
    9/9 [==============================] - 25s 3s/step - loss: 0.0334 - accuracy: 0.9888
    Epoch 16/20
    9/9 [==============================] - 26s 3s/step - loss: 0.0433 - accuracy: 0.9875
    Epoch 17/20
    9/9 [==============================] - 22s 2s/step - loss: 0.0497 - accuracy: 0.9902
    Epoch 18/20
    9/9 [==============================] - 22s 2s/step - loss: 0.0746 - accuracy: 0.9834
    Epoch 19/20
    9/9 [==============================] - 23s 2s/step - loss: 0.0401 - accuracy: 0.9847
    Epoch 20/20
    9/9 [==============================] - 23s 3s/step - loss: 0.0587 - accuracy: 0.9838



```python
model.evaluate_generator(test_generator)
```




    [0.04233855754137039, 0.9858871102333069]



### Model Validation


```python
sample_mask_img1 = cv2.imread('./Face Mask Dataset/Test/WithMask/187.png')
sample_mask_img1 = cv2.resize(sample_mask_img1,(128,128))
plt.imshow(sample_mask_img1)
sample_mask_img1 = np.reshape(sample_mask_img1,[1,128,128,3])
sample_mask_img1 = sample_mask_img1/255.0
```


    
![png](Face-Mask-Detection-CNN-MLP-VGG99_files/Face-Mask-Detection-CNN-MLP-VGG99_78_0.png)
    



```python
model.predict(sample_mask_img1)
```




    array([[0.98006535, 0.07107526]], dtype=float32)



Here we can see that there is a 0.71 probability of the person wearing a mask


```python
sample_mask_img2 = cv2.imread('./Face Mask Dataset/Test/WithoutMask/147.png')
sample_mask_img2 = cv2.resize(sample_mask_img2,(128,128))
plt.imshow(sample_mask_img2)
sample_mask_img2 = np.reshape(sample_mask_img2,[1,128,128,3])
sample_mask_img2 = sample_mask_img2/255.0
```


    
![png](Face-Mask-Detection-CNN-MLP-VGG99_files/Face-Mask-Detection-CNN-MLP-VGG99_81_0.png)
    



```python
model.predict(sample_mask_img2)
```




    array([[0.07868743, 0.9842355 ]], dtype=float32)



Here we can see that there is a 0.97 probability that the person is not wearing a mask

## 2.3 MobileNet

- MobileNet is a CNN architecture model for Image Classification and Mobile Vision.
- MobileNet requires very less computation power to run or apply transfer learning to.
- perfect fit for Mobile devices,embedded systems and computers without GPU or low computational efficiency
- uses depth wise separable convolutions to build light weight deep neural networks.


The full MobileNet V2 architecture, then, consists of 17" building blocks" in a row. This is followed by a regular 1×1 convolution, a global average pooling layer, and a classification layer
MACs are multiply-accumulate operations, which measure how many calculations are needed to perform inference on a single 224×224 RGB image. V2 requires approx 300 such calculations.
This model offers us Reduced network size ,Reduced number of parameters ,Faster in performance and are useful for mobile applications.,Small, low-latency convolutional neural network.


<img src="Face Mask Dataset/figures/mn1.png">

<img src="Face Mask Dataset/figures/mn2.png">


```python
train_dir = "Face Mask Dataset/Train/"
test_dir = "Face Mask Dataset/Test/"
val_dir = './Face Mask Dataset/Validation'
```


```python
#with Mask
plt.figure(figsize=(12,7))
for i in range(5):
    sample = random.choice(os.listdir(train_dir+"WithMask/"))
    plt.subplot(1,5,i+1)
    img = load_img(train_dir+"WithMask/"+sample)
    plt.subplots_adjust(hspace=0.001)
    plt.xlabel("With Mask")
    plt.imshow(img)
plt.show()

#without Mask
plt.figure(figsize=(12,7))
for i in range(5):
    sample = random.choice(os.listdir(train_dir+"WithoutMask/"))
    plt.subplot(1,5,i+1)
    img = load_img(train_dir+"WithoutMask/"+sample)
    plt.subplots_adjust(hspace=0.001)
    plt.xlabel("Without Mask")
    plt.imshow(img)
plt.show()
```


    
![png](Face-Mask-Detection-CNN-MLP-VGG99_files/Face-Mask-Detection-CNN-MLP-VGG99_87_0.png)
    



    
![png](Face-Mask-Detection-CNN-MLP-VGG99_files/Face-Mask-Detection-CNN-MLP-VGG99_87_1.png)
    



```python
height = 150
width=150
train_datagen = ImageDataGenerator(rescale=1.0/255,validation_split=0.2,shear_range = 0.2,zoom_range=0.2,horizontal_flip=True)
train = train_datagen.flow_from_directory(directory=train_dir,target_size=(height,width),
                                          class_mode="categorical",batch_size=32,subset = "training")

valid_datagen = ImageDataGenerator(rescale=1.0/255)

valid = train_datagen.flow_from_directory(directory=train_dir,target_size=(height,width),
                                          class_mode="categorical",batch_size=32,subset="validation")
```

    Found 8000 images belonging to 2 classes.
    Found 2000 images belonging to 2 classes.



```python
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

mobilenet = MobileNetV2(weights = "imagenet",include_top = False,input_shape=(150,150,3))
```

    WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
    9412608/9406464 [==============================] - 1s 0us/step



```python
for layer in mobilenet.layers:
    layer.trainable = False
```


```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense
model = Sequential()
model.add(mobilenet)
model.add(Flatten())
model.add(Dense(2,activation="sigmoid"))
```


```python
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    mobilenetv2_1.00_224 (Functi (None, 5, 5, 1280)        2257984   
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 32000)             0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 2)                 64002     
    =================================================================
    Total params: 2,321,986
    Trainable params: 64,002
    Non-trainable params: 2,257,984
    _________________________________________________________________



```python
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics ="accuracy")
```


```python
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
checkpoint = ModelCheckpoint("moblenet_facemask.h5",monitor="val_accuracy",save_best_only=True,verbose=1)
earlystop = EarlyStopping(monitor="val_accuracy",patience=5,verbose=1)
```


```python
history = model.fit_generator(generator=train,steps_per_epoch=len(train)// 32,validation_data=valid,
                             validation_steps = len(valid)//32,callbacks=[checkpoint,earlystop],epochs=15)
```

    Epoch 1/15
    7/7 [==============================] - 6s 575ms/step - loss: 4.8479 - accuracy: 0.6541 - val_loss: 1.0105 - val_accuracy: 0.8438
    
    Epoch 00001: val_accuracy improved from -inf to 0.84375, saving model to moblenet_facemask.h5
    Epoch 2/15
    7/7 [==============================] - 3s 445ms/step - loss: 0.4651 - accuracy: 0.9399 - val_loss: 1.9744e-07 - val_accuracy: 1.0000
    
    Epoch 00002: val_accuracy improved from 0.84375 to 1.00000, saving model to moblenet_facemask.h5
    Epoch 3/15
    7/7 [==============================] - 3s 386ms/step - loss: 0.4585 - accuracy: 0.9741 - val_loss: 0.5303 - val_accuracy: 0.9688
    
    Epoch 00003: val_accuracy did not improve from 1.00000
    Epoch 4/15
    7/7 [==============================] - 3s 438ms/step - loss: 0.1053 - accuracy: 0.9884 - val_loss: 0.7955 - val_accuracy: 0.9688
    
    Epoch 00004: val_accuracy did not improve from 1.00000
    Epoch 5/15
    7/7 [==============================] - 3s 446ms/step - loss: 0.0615 - accuracy: 0.9947 - val_loss: 8.9903e-05 - val_accuracy: 1.0000
    
    Epoch 00005: val_accuracy did not improve from 1.00000
    Epoch 6/15
    7/7 [==============================] - 3s 434ms/step - loss: 0.0122 - accuracy: 0.9965 - val_loss: 0.4482 - val_accuracy: 0.9062
    
    Epoch 00006: val_accuracy did not improve from 1.00000
    Epoch 7/15
    7/7 [==============================] - 3s 418ms/step - loss: 0.1385 - accuracy: 0.9781 - val_loss: 0.0948 - val_accuracy: 0.9688
    
    Epoch 00007: val_accuracy did not improve from 1.00000
    Epoch 00007: early stopping



```python
model.evaluate_generator(valid)
```




    [0.22745804488658905, 0.9714999794960022]




```python
model.save("face_mask.h5")
pred = model.predict_classes(valid)
pred[:15]
```




    array([1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])



### Running Haar cascade to detect Face and mobileNet to detect Mask


```python
def mobileNet_mask_id(image_name):
    
    mask = "Face Mask Dataset/Validation"
    plt.figure(figsize=(8,7))
    label = {0:"With Mask",1:"Without Mask"}
    color_label = {0: (0,255,0),1 : (0,0,255)}
    cascade = cv2.CascadeClassifier("Face Mask Dataset/haarcascade_frontalface_default.xml")
    count = 0
    i = "Face Mask Dataset/Validation/" + image_name

    frame =cv2.imread(i)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        face_image = frame[y:y+h,x:x+w]
        resize_img  = cv2.resize(face_image,(150,150))
        normalized = resize_img/255.0
        reshape = np.reshape(normalized,(1,150,150,3))
        reshape = np.vstack([reshape])
        result = model.predict_classes(reshape)

        if result == 0:
            cv2.rectangle(frame,(x,y),(x+w,y+h),color_label[0],1)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            plt.imshow(frame)
        elif result == 1:
            cv2.rectangle(frame,(x,y),(x+w,y+h),color_label[1],1)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            plt.imshow(frame)
        plt.imshow(frame)
    plt.show()
    cv2.destroyAllWindows()
```

### Model Validation


```python
mobileNet_mask_id('WithoutMask/431.png')
```


    
![png](Face-Mask-Detection-CNN-MLP-VGG99_files/Face-Mask-Detection-CNN-MLP-VGG99_101_0.png)
    



```python
mobileNet_mask_id('WithMask/431.png')
```


    
![png](Face-Mask-Detection-CNN-MLP-VGG99_files/Face-Mask-Detection-CNN-MLP-VGG99_102_0.png)
    

