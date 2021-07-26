# Fatigue Face Detector

The goal of this project is to detect any fatigue (eyes closing, mouth open, and yawning) from a person's face in real time. Specific areas/features from a face that are important would be eyes and mouth.

<p align="center">
   <img src="https://compote.slate.com/images/be86a141-95de-40ef-9352-3514aef96fa9.jpg" width="50%" height="50%">
</p>

## Methods

**Two** methods that are being used in this Project are:

- Calculating the Eye Aspect Ration and Mouth using Dlib. [EAR](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)

- Convolutional Neural Network. [CNN](https://towardsdatascience.com/drowsiness-detection-using-convolutional-neural-networks-face-recognition-and-tensorflow-56cdfc8315ad)

<p align="center">
   <img src="https://miro.medium.com/max/1400/1*moqDUbHEkoXJdSmBzkgdkQ.png" width="80%">
</p>

## Calculating the Eye Aspect Ration and Mouth using Dlib

Based on this paper [1](https://towardsdatascience.com/drowsiness-detection-using-convolutional-neural-networks-face-recognition-and-tensorflow-56cdfc8315ad). A real-time eye and mouth closing can be detected using Dlib. Dlib has 68 key points in (x,y) coordinates to help recognize facial features. The main concept is to calculate the distance between each of the interesting key points. In this project, the important key points are **37-42 (left eye)**, **43-48 (right eye)**, and **49-68 (mouth)**.

<p align="center">
   <img src="https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg" width="50%" height="50%">
</p>

### Eye Aspect Ratio

All the key points in the eye area are necessary to detect if the eyes are closed or not, the distance between the key points with their own's respectively opposite will be calculated.

<p align="center">
   <img src="https://lh5.googleusercontent.com/NEdpVPSIHlb6vKjJ86d3Q_spX0MrYB33GeMvdn3J3k4B4kr87Jpy7YBw4shn1JfwpXEOfNzjIhEHpsDh-dndx2j-riFGiDgbqk7diPEGl5mA__sgDKUuczbJd5tCUKSALwIJ6zp3" width="50%" height="50%">
</p>
The Eye Aspect Ratio usually hold a constant value when the eye is currently open, but when the eye is closed it will falls to almost 0 depending on the location of camera and the face.

<p align="center">
   <img src="https://lh6.googleusercontent.com/qoeyiEcyQI4jfi3Bu2WTXX0rWsPYixvJjmqSjQ6ChvPpi2tCLBNXQCLedJNhaq4B-_U9vyk70e5vpOChxlPZNCUGAfv9A30pXsGXgarmUAmryM-M91hUS0Bgy2Yle1J7SX2NYSln" width="50%" height="50%">
</p>

### Mouth Aspect Ratio

The same principles as previously also apply to detecting if the mouth is currently closed or open. The distance between the middle part of the upper lip (**51-53 and 62-64**) with the lower lip (**66-68 and 57-59**) will be calculated.

<p align="center">
   <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRPRQjFHRnrUoLuF_IS8uRzheaX7fmuWITrqA&usqp=CAU" width="50%" height="50%">
</p>

### Implementation

Once the facial features of the person are already recognized with all the 68 key points the distance can be directly calculated to check if the eyes/mouth are closed. Below is the threeshold for both mouth and eyes.

> EYE_AR_THRESH = 0.17
> YAWN_THRESH = 20

If by any chance the eye-aspect-ratio based on the calculation is lower than the threshold, it is safe to assume that the eyes are currently closed. The opposite apply also for the mouth, if the distance between the upper and lower lip is bigger than the threshold, most likely the mouth is currently open.

> Run the distance_dlib.py file to try it.

## Convolutional Neural Network

### Architecture's model

The main architecture's model is based on [this](https://towardsdatascience.com/drowsiness-detection-using-convolutional-neural-networks-face-recognition-and-tensorflow-56cdfc8315ad)

<p align="center">
   <img src="https://miro.medium.com/max/1400/1*moqDUbHEkoXJdSmBzkgdkQ.png" width="90%">
</p>

### Dataset

The model will be trained using [dataset](https://www.kaggle.com/serenaraju/yawn-eye-dataset-new).

<p align="center">
   <img src="https://storage.googleapis.com/kagglesdsdata/datasets/762074/1315164/dataset_new/test/Open/_20.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210726%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210726T153451Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=04fdcbed542cfa096fdc13a08b4ee1b22b63fb568c51c349fe714f17a94f0fe6fb35b46a05acccf56facdc8d0035301b8348b3f6a3f36ac9092a190b682d4a68fb18a950cb737e768c7cefae2adc5f87b858a7a45fcaa19e74ea3aec7bcd2f2ca1bfa17908e687498b5cf42d81975c250aafdcb7093d99bbcd9c76c3b641a8925780712bb3aa1968ace677892796e91f12e5277f89cc7b672f9328aba4071f92d2a7223db3744721c1398ec59e06c51a7fb231871cf3133fd7cd417a7ab42e261a7e441f34f4f78d863596b3fd18ab41b17cf8fd621f7ca6794e0e6f86a5105f350975a0004c23ab7ad12f79cbb5fea7ff6ead500e041b00043573079ba59c3d" height="30%" width="30%">
    <img src="https://storage.googleapis.com/kagglesdsdata/datasets/762074/1315164/dataset_new/test/Closed/_120.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210726%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210726T153637Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=8f610c23c4715ef432fcc99cfbce82b46236380dec93f034c18e242f4ed1d988ef922e1cf8e959e11e7be262f296b587a7dbe20fd00223d15d2dafd922c7e352689bc3e94ff2c1c1110e565e4f80df7af6bc1ce35bd19259b7e4727497ecdafb246650b2c64b4a7aecd35afe32292551d3948a801f1c2cde037db80d5fbdf4597b34a30f78a1f706974dfb01f62fa4da66f124f12dff705dac063be3d056a88398c18f9238c4bf39ac1d80f94ecb5a759c6d10bd7bf581eb9d1f87fb26e4a81058111e7c90fbee35c6d41465b15e7e461082cabfa6bfce50cdcaa2356938b4de86552e4532e3ce07142d335ccc09dbc62ae12220a99105feed064ef62d70878f" height="10%" width="25%">
</p>
<p align="center">
<img src="https://storage.googleapis.com/kagglesdsdata/datasets/762074/1315164/dataset_new/train/no_yawn/1.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210726%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210726T154053Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=698e93ccb36f725517da712c9170a3f91f5092e16f4871e0069e2d7ba0c09733e6db8cb83d38be1937285f420673edf01909ee70821bef923ff5d19c3677bb8d9a0423b1534d43cea1099e31bcc2753d65148189a74679f940850da26f880828a8e81f683d11f167f3d73fd31d8ca925dc6ae247b1b8cf1b35e9e9876b829212448bf71bac9a1c838686c3b1a61232832ed1a409296d65abdf68ea46bec4af926a2f8a1a937c808009ccc713faa118be3f4aed2bdf1b2f9fe5dc9ba968dfd15feae0d39ca79d09b6cc6b5cda08e2b8e20d00bfcc5a54994fd9cfe00514d93bfdb56b00082d2fa7feda7b353da3f842f03ff9602f51e6a82f982c47ada9530a50" height="10%" width="28%">
  <img src="https://storage.googleapis.com/kagglesdsdata/datasets/762074/1315164/dataset_new/train/yawn/1.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210726%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210726T154150Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=46beb76d6d0bb7fce73da96e0bb81af7c7d415c1d1e321bc5b6ee61c9f48b224dbee9d5313d6ff2a9a45dd0a770c4d9cfbc54aa2077e498ff277fcaa04629bc2bc2970789c7f8528831db35e08483aa52fa7b8de21ab9c88d663e5420444818b0882a0224321e991588fb19fd98e653ca3615559888cca0f3fa35ffbe3a24a82b0b3159762e62b94917b00e3e9dd943517095eb0cb48545dda5a91ff0aa3f9e830a81cff8d02f65c348ef2cfab0e3883daed7ec1708cea1e4a6921f14f710b774881f704791bbfb1e2f70920693a417ff10b106303e73a327419b4c915da635469d4dcb4cad82c55a2976dea3f0fb13fb2bbaf8e4b143af2760d4a5f306063a2" height="10%" width="28%">
 </p>

Because the dataset only has the full face and not directly the mouth part like the eyes dataset. A python script modified from [this](https://stackoverflow.com/questions/51097659/script-for-identifying-landmarks-and-cropping-mouth-from-images-using-opencv-doe) using dlib is written and will be needed to crop the image.

### Testing with different Hyperparameters

The default parameter for CNN can be seen below.

> Filters=32,
> kernel_size=(3,3),
> pool_size=(2,2),
> dense_nodes=256,
> decrease_dense=True, loss='binary_crossentropy',
> optimizer='adam',
> activation='sigmoid', drop=0.3,
> metrics=keras.metrics.AUC(curve = 'PR'))

#### Result with the default parameters:

> For all the plots, left is the accuracy and right is the loss

Eyes

<p align="center">
   <img src="/Plots/Sigmoid_300_15_acc.png" height="100%" width="45%">
   <img src="/Plots/Sigmoid_300_15_loss.png" height="100%" width="45%">
 </p>
Mouth
<p align="center">
   <img src="/Plots/mouth/Sigmoid_300_15_acc.png" height="100%" width="45%">
   <img src="/Plots/mouth/Sigmoid_300_15_loss.png" height="100%" width="45%">
 </p>

#### Mouth result without densing the layer:

> The layer will remain as 256 Layers then before shrinking it down to 1 for activation.

<p align="center">
   <img src="/Plots/mouth/Sigmoid_300_15_acc_False.png" height="100%" width="45%">
   <img src="/Plots/mouth/Sigmoid_300_15_loss_False.png" height="100%" width="45%">
 </p>

#### Result with 1 Filters:

Eyes

<p align="center">
   <img src="/Plots/Sigmoid_300_15_acc_1.png" height="100%" width="45%">
   <img src="/Plots/Sigmoid_300_15_loss_1.png" height="100%" width="45%">
 </p>

Mouth

<p align="center">
   <img src="/Plots/mouth/Sigmoid_300_15_acc_1.png" height="100%" width="45%">
   <img src="/Plots/mouth/Sigmoid_300_15_loss_1.png" height="100%" width="45%">
 </p>
 
#### Result with 16 Filters:
Eyes
<p align="center">
   <img src="/Plots/Sigmoid_300_15_acc_16.png" height="100%" width="45%">
   <img src="/Plots/Sigmoid_300_15_loss_16.png" height="100%" width="45%">
 </p>
Mouth
<p align="center">
<img src="/Plots/mouth/Sigmoid_300_15_acc_16.png" height="100%" width="45%">
  <img src="/Plots/mouth/Sigmoid_300_15_loss_16.png" height="100%" width="45%">
 </p>

#### Result with 64 Filters:

> The layer is now double the size from the default.

Eyes

<p align="center">
<img src="/Plots/Sigmoid_300_15_acc_64.png" height="100%" width="45%">
  <img src="/Plots/Sigmoid_300_15_loss_64.png" height="100%" width="45%">
 </p>

Mouth

<p align="center">
<img src="/Plots/mouth/Sigmoid_300_15_acc_64.png" height="100%" width="45%">
  <img src="/Plots/mouth/Sigmoid_300_15_loss_64.png" height="100%" width="45%">
 </p>

> Run the fatigue_cnn.py file to try it.

## Conclusion and thoughts

Both methods can detect any fatigues expression such as eyes closed and mouth opened very well. Respectively have their pros and cons.

### The Eye and Mouth Aspect Ratio

The Eye and Mouth Aspect Ratio is very fast in regards that once it detects the facial expression, the calculation will be running directly and can detect if a person currently shows any symptoms of tiredness. Dlib is a great Toolkit/library to use out of the box without having to train an additional model and preparing the datasets from scratch.

### Convolutional neural network

The Convolutional neural network on the other hand gives more room to properly tune the model, prepare a good dataset and train the model to the desired goal.
