#   **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[commaai]: ./saved_images/commaai.tiff
[nvidia]: ./saved_images/nvidia.tiff
[3views]: ./saved_images/3views.png
[3bright]: ./saved_images/3bright.png
[3shadow]: ./saved_images/3shadow.png
[flip]: ./saved_images/flip.png
[shift]: ./saved_images/shift.png
[full_histo]: ./saved_images/full_histo.png
[gen_histo]: ./saved_images/gen_histo.png
[raw_histo]: ./saved_images/raw_histo.png
[gen_histo_true_random]: ./saved_images/gen_histo_true_random.png
[histo]: ./saved_images/histo.png
[pred1]: ./saved_images/pred1.png
[pred2]: ./saved_images/pred2.png
[pred3]: ./saved_images/pred3.png
[pred4]: ./saved_images/pred4.png
[pred5]: ./saved_images/pred5.png
-----

## Exploration notes

First, recorded 1 lap using arrows. Nvidia architecture. Couldn't get past the bridge.
Then added the "recovery" recordings to teach the car how to recenter itself. The car behaved worse.

I then recorded 1 lap using the trackpad smoothing the recorded steering angle used for labelling.



It enabled me to reduce the loss significantly:
    training loss: 0.0149 - validation loss: 0.0640 in 2 epochs
to
    training loss:  - validation loss:  in 5 epochs

![alt text][3bright]


model_1 fine until 2nd turn after bridge




---

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The augmenting and preprocessing pipeline I used for training and validating the model is in data.py, and it contains comments to explain how the code works. I perform some tests in test.py. The code in that file allowed me to output the figures in this report.
The code has been separated for clarity.

### Model Architecture

#### 1. Models Experimented

I have been really impressed by the work Nvidia has been doing using convolutionnal neural network to teach a car how to drive by itself by copying the driving habits of a human driver as described in this [paper](https://arxiv.org/pdf/1604.07316v1.pdf).


The Nvidia model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64.

* First, I added a lambda layer to normalize the image between -0.5 and 0.5.
* I use the cropping function provided by keras to remove the parts of the image that are not relevant (70 pixels at the top and 25 pixels at the bottom of the image, keep the full width).
* The rest of the model is the same as used by nvidia accept that my input image is not the same as theirs.
* The model includes ELU layers to introduce nonlinearity.

![alt text][nvidia]

On the other hand, George Hotz and its company Comma.ai have achieved groundbreaking results before releasing the architecture of their CovNet. I testing this model as well.

![alt text][commaai]

We can note that there is 10x the numbers of trainable parameters in that model compared to the one used by Nvidia.

#### 2. Attempts to reduce overfitting in the model

Both models contain dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. In that case, it doesn't make sense to keep a set of the data dedicated for testing. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

Both model used an adam optimizer, so the learning rate was not tuned manually.


### Training Strategy

#### 1. Appropriate training data

At first, I drove the car in the center of the road as accurately as possible for 2 laps. I also added some recovery data, recordings where the car happen to be on the side of the road and steer back to the center of the road. I found that process tricky because the by doing so, the car can learn from bad data.

After testing I found out that the training data with which I first succeeded to close an autonomously driven gap on the first track was the following:

* 4 laps where the vehicle is driving in the center of the road.
* Recovering from the left and right sides of the road was **not** recorded.

#### 2. Data presentation

Here is the 3 images that are generated by the simulation software provided by Udacity

![alt text][3views]

The steering angle of the first image is the one recorded. I then derive the steering angle to be applied for the left and right cameras.

#### 3. Preprocessing pipeline

The preprocessing pipeline is the transformations that every images need to be applied onto every images before entering the neural network, would that be for training or for performing an actual prediction once on the road.

```python
def preprocess(image):
    '''
    Preprocessing pipeline
    Cropping done with a keras call.
    '''
    image = GaussianBlur(image)
    image = RGB2YUV(image)
    return image
```

* Gaussian blur is used to smooth out the image.
* The conversion to YUV format is recommended by nvidia.

#### 4. Augmentation pipeline

The augmentation pipeline is the set of transformations applied to data to obtain more of it, reduce overfitting and help the model generalize.

```python
def augment(center, left, right, steering_angle, parameter):
    '''
    Augment the data to improve the model accuracy, help generalize and prevent overfitting
    '''
    # The steering_angle is the one from the center image so far
    image , steer = random_image_pick(center, left, right, steering_angle, parameter)
    # Now, the steering_angle is adapted to the chosen image
    if np.random.choice(2) == 0:
        image, steer = horizontal_flip(image, steer)
    if parameter['color_augmentation']:
        image = random_brightness(image)
        image = random_shadow(image)
    if parameter['shift_augmentation']:
        image, steer = random_shift(image, steer)
    return image, steer
```

First of all, since I am recording data always driving in the same direction on the track, the data is heavily biased to the left.

![alt text][raw_histo]

My approach was to have a 50% chance to flip the image horizontally so that it cancels that bias over a high number of iterations. The `horizontal_flip(image, steering_angle):` function outputs:

![alt text][flip]

That enables the follwing more balanced histogram:

![alt text][histo]

I obtain data that is heavily centered. At that point I decided to add the side cameras with a bias of **0.25**. That number started at .2 and changed over time as I tested my models. The `random_image_pick()` function comes into play.

![alt text][full_histo]

At that point, we can see that the data is strongly unevenly balanced and that causes problem for the model that learns to steer at values representing those 3 peaks more than everywhere else. I needed to balance my dataset. I introduced the following augmenting function:

```python
def random_shift(image, steering_angle):
    '''
    Shifts the image horizontally and vertically
    '''
    X_range = 50
    Y_range = 10
    dX = X_range * np.random.uniform() - X_range / 2
    steering_angle += dX * .01
    dY = Y_range * np.random.uniform() - Y_range / 2
    shift = np.float32([[1,0,dX],[0,1,dY]])
    image = cv2.warpAffine(image,shift,(image.shape[1], image.shape[0]))
    return image, steering_angle
```

I obtain the following images:

![alt text][shift]

and the corresponding histogram when the data is shifted using that function. The data is directly obtained from the generator so that is represents well what the model is receiving.


![alt text][gen_histo_true_random]


Lastly I wanted to test whether my model could generalize to the 2nd track. To do so, I needed to augment my images a little bit more. First because the 2nd track is very hilly. The model needs to learn to detect on the bottom and on the top of the image. In the `random_shift(image, steering_angle):` function, a vertical shift takes into account this.

On the second track, the brightness is varying and in some places, shadows diminish the the luminosity quite significantly. I created the `random_shadow(image):` and `random_brightness(image):` to augment the training data even more. Here are the outputs to these functions.

![alt text][3shadow]

![alt text][3bright]



### Final Solution

#### 1. Solution Design Approach

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I added the possibility tfor an early termination. If the validation loss doesn't decrease after the following 2 training epochs, I rewind the model to keep the weights at that epoch.

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (clone.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process


The final data distribution that I fould would make the model work is the following:

![alt text][gen_histo]

As a comparison to the last histogram higher in this file, this data distribution is more evenlly distributed between -.25 and +.25. As a result the car seems more encline to steer when it approches a turn whereas the previous one failed in the 2nd sharp turn after the bridge.



To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


#### 4. Output video

![](./recorded_runs/run1.mp4)   
