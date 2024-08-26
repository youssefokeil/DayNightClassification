# Preprocessing 
## File Structure
First I had to process the dataset folder, so that it could work well with the ImageFolder() function.
  So I had to change the file structure from something like this:
  ```
  dataset-005
      └───day-night images
          ├───day
          ├───night_2nd_round
          │   ├───folder_1
          │   ├───folder_10
          └───night_3rd_round
              ├───folder_110
              ├───folder_114
              ├───folder_12
  .....
  ```
  to something like this
  ```
  ├───day
  └───night
  ```
## Downsample
The data was unbalanced.
```
Number of day files is: 2762
Number of night files is: 22819
```
So I downsample the night files to match the day files.
# SVM Model
## Model Definition
```
svc=svm.SVC(probability=True)
param_grid={'C':[0.1,1,10,100], 
            'gamma':[0.0001,0.001,0.1,1], 
            'kernel':['rbf','poly']} 
model_svm=RandomizedSearchCV(svc,param_grid,n_iter=5)
```
## Performance
after 90 mins of training on CPU, the model performed so well (in a worrying way).
```
              precision    recall  f1-score   support

         day       1.00      1.00      1.00       556
       night       1.00      1.00      1.00       549

    accuracy                           1.00      1105
   macro avg       1.00      1.00      1.00      1105
weighted avg       1.00      1.00      1.00      1105
```
# PyTorch Model (no.1)
## Model Definition (Same As TensoFlow)
```
class CNNModule(nn.Module):
    def __init__(self):
        super().__init__()
        #our first convolution layer
        self.conv1=nn.Conv2d(3,16,3,padding=1) 
        self.conv2=nn.Conv2d(16,32,3,padding=1)
        self.conv3=nn.Conv2d(32,64,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(16*16*64,128)
        self.fc2=nn.Linear(128,1)
        self.sigmoid=nn.Sigmoid()
        self.dropout=nn.Dropout(p=0.2)

    def forward(self,x):
        # input=128 x 128 x 3, output=64 x 64 x 16
        x=self.pool(F.relu(self.conv1(x)))
        # input=64 x 64 x 16, output=32 x 32 x 32
        x=self.pool(F.relu(self.conv2(x)))
        # input=32 x 32 x 32, output=16 x 16 x 64
        x=self.pool(F.relu(self.conv3(x)))
        # first dropout layer
        x=self.dropout(x)
        # flatten
        x=x.view(-1,16*16*64)        
        # fully connected layers
        x=F.relu(self.fc1(x))
        # second dropout layer
        x=self.dropout(x)
        x=self.fc2(x)
        return x     

```
## Performance
Last 5 epochs of 30 epoch training.
```
For epoch 25
, Validation loss is 0.02024, Training Loss is 0.00014, Accuracy is 99.554%
For epoch 26
, Validation loss is 0.02578, Training Loss is 0.00015, Accuracy is 99.380%
For epoch 27
, Validation loss is 0.01351, Training Loss is 0.00038, Accuracy is 99.609%
For epoch 28
, Validation loss is 0.01130, Training Loss is 0.00039, Accuracy is 99.547%
For epoch 29
, Validation loss is 0.02791, Training Loss is 0.00007, Accuracy is 98.822%
For epoch 30
, Validation loss is 0.01755, Training Loss is 0.00011, Accuracy is 99.107%
```
Model Complexity Graph
!(Pytorch_Plot)[https://github.com/youssefokeil/DayNightClassification/blob/main/Files_Github/Pytorch_ModelComplexity.jpeg]

## Model Redefiniton Using HSV color space and BatchNorm2d
### Custom Dataset HSV
I defined my own custom dataset to make transforms, change color space of image to hsv and output v only. HSV will make us separate the v field, which corresponds to value and will give higher importance to brightness in image. 
This is extremely helpful with a model that will focus on day & night classifcation, since brightness is a pretty impoprtant feature. 

A day image will look something like this in v color space.
![HSV_image](https://github.com/youssefokeil/DayNightClassification/blob/main/Files_Github/HSV_img.jpeg)

### Performance of Redefined Model
```
For epoch 1
, Validation loss is 47.03125, Training Loss is 0.00000, Accuracy is 94.754%
For epoch 2
, Validation loss is 3.93750, Training Loss is 0.00000, Accuracy is 99.696%
For epoch 3
, Validation loss is 3.00781, Training Loss is 0.00000, Accuracy is 99.396%
```
# Tensorflow Model
## Model Definition
```
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, name="outputs")
])
```
## Performance 
Model performed really well, at the end it had a 99% to 100% accuracy
```
Epoch 9/10
35/35 [==============================] - 31s 864ms/step - loss: 0.0043 - accuracy: 0.9986 - val_loss: 0.0023 - val_accuracy: 0.9991
Epoch 10/10
35/35 [==============================] - 35s 968ms/step - loss: 0.0059 - accuracy: 0.9980 - val_loss: 0.0015 - val_accuracy: 1.0000
```

A plot of the history of my Tensorflow model:
![tf_history](https://github.com/youssefokeil/DayNightClassification/blob/main/Files_Github/TF_history.jpeg)
