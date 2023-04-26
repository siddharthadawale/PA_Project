from __future__ import print_function
import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import IPython.display as display
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import PIL.Image
import time
import functools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


import PIL.Image as Image
import matplotlib.pyplot as plt
import copy

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

import torchvision.transforms.functional as F
from torchvision.utils import make_grid
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
from glob import glob
from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope
import tensorflow_hub as hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    model = tf.keras.models.load_model(r"C:\Users\harsh\Downloads\model (2).h5")
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)
def load_img(img):
  max_dim = 512
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img
def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
def mask_content(content_img, generated_img, mask_img):
    width = mask_img.size(2)
    height = mask_img.size(3)
    mask_img = mask_img.data.cpu().numpy()
    content_img = content_img.data
    
    for i in range(width):
        for j in range(height):
            if mask_img[0, :, i, j].all() == 0:
                generated_img[0, :, i, j] = content_img[0, :, i, j]
    
    return generated_img
transform = transforms.Compose([transforms.ToTensor()])

def image_loader(image_name):
    image=image_name
    if transform is not None:
        image = transform(image)
        
    image = Variable(image)
    image = image.unsqueeze(0)
    
    return image.type(dtype)
def seg_image1(seg_path):
  """ Global parameters """
  H = 512
  W = 512

  """ Creating a directory """
  def create_dir(path):
      if not os.path.exists(path):
          os.makedirs(path)

  if __name__ == "__main__":
      """ Seeding """
      np.random.seed(42)
      tf.random.set_seed(42)

      """ Directory for storing files """
      create_dir("remove_bg")

      # model.summary()

      """ Load the dataset """
      data_x = glob(seg_path)

      for path in tqdm(data_x, total=len(data_x)):
          """ Extracting name """
          name = path.split("/")[-1].split(".")[0]

          """ Read the image """
          image = cv2.imread(path, cv2.IMREAD_COLOR)
          h, w, _ = image.shape
          x = cv2.resize(image, (W, H))
          x = x/255.0
          x = x.astype(np.float32)
          x = np.expand_dims(x, axis=0)

          """ Prediction """
          y = model.predict(x)[0]
          y = cv2.resize(y, (w, h))
          y = np.expand_dims(y, axis=-1)
          y = y > 0.5

          photo_mask = y
  return photo_mask*255

######Implementation#########
import streamlit as st
import PIL.Image as Image
def output(op1,op2,op3,i):
    st.title("Output Image:")
    content_path=op1
    style_path=op2

    content_image = load_img(tf.io.read_file(content_path))
    style_image = load_img(tf.io.read_file(style_path))
    img_array_2d = np.squeeze(seg_image1(content_path), axis=2)  # Convert to a 2D array
    img_array_2d = img_array_2d.astype(np.uint8)  # Convert to uint8
    # Create a PIL Image object from the 2D ndarray
    seg_img = Image.fromarray(img_array_2d)
    seg_img.save('image.png')
    seg_image = load_img(tf.io.read_file('image.png'))

    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

    if i==0:
        st.image(tensor_to_image(stylized_image))
    elif i==1:
        output = mask_content(image_loader(tensor_to_image(content_image)), image_loader(tensor_to_image(stylized_image)),image_loader(tensor_to_image(seg_image)) )
        torchvision.utils.save_image(output, 'output_masked.jpg')
        st.image(Image.open('output_masked.jpg'))
    elif i==2:
        output = mask_content(image_loader(tensor_to_image(stylized_image)), image_loader(tensor_to_image(content_image)),image_loader(tensor_to_image(seg_image)) )
        torchvision.utils.save_image(output, 'output_masked.jpg')
        st.image(Image.open('output_masked.jpg'))
    elif i==3:
        style_path2=op3
        style_image2 = load_img(tf.io.read_file(style_path2))
        stylized_image2 = hub_model(tf.constant(content_image), tf.constant(style_image2))[0]
        output = mask_content(image_loader(tensor_to_image(stylized_image2)), image_loader(tensor_to_image(stylized_image)),image_loader(tensor_to_image(seg_image)) )
        torchvision.utils.save_image(output, 'output_masked.jpg')
        st.image(Image.open('output_masked.jpg'))
def take_user():
    for image in content_image:
        st.image(Image.open(content_image[image]), width=200)
    option1 = st.selectbox("Choose an Content image",list(content_image.keys())+ ["Upload your own image"])
    if option1 == "Upload your own image":
        uploaded_file1 = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    # If the user chooses "Choose an image", display a default image
        if uploaded_file1 is not None:
            # Read the contents of the file
            image_bytes = uploaded_file1.read()
            # Save the image to a file
            with open("uploaded_file1.jpg", "wb") as f:
                f.write(image_bytes)
            st.image(uploaded_file1, width=200)
            option1='uploaded_file1.jpg'
    # If the user chooses "Upload your own image", allow them to upload an image
    else:
        st.image(Image.open(content_image[option1]),width=200)
        option1=content_image[option1]
    st.title("Select Style Image")
    for image in style_image:
        st.image(Image.open(style_image[image]), width=200)
    option2 = st.selectbox("Choose an Style image",list(style_image.keys())+ ["Upload your own image"])
    if option2 == "Upload your own image":
        uploaded_file2 = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    # If the user chooses "Choose an image", display a default image
        if uploaded_file2 is not None:
            # Read the contents of the file
            image_bytes = uploaded_file2.read()
            # Save the image to a file
            with open("uploaded_file2.jpg", "wb") as f:
                f.write(image_bytes)
            st.image(uploaded_file2, width=200)
            option1='uploaded_file2.jpg'
        # If the user chooses "Upload your own image", allow them to upload an image
    else:
        st.image(Image.open(style_image[option2]),width=200)
        option2=style_image[option2]
    return option1,option2
home_image = {
    "Image 1": r"C:\Users\harsh\Downloads\DS\1.png",
    "Image 2": r"C:\Users\harsh\Downloads\DS\2.png",
    "Image 3": r"C:\Users\harsh\Downloads\DS\3.png",
    "Image 4": r"C:\Users\harsh\Downloads\DS\4.png",
}

content_image = {
    "Image 1": r"C:\Users\harsh\Downloads\Hrithik-Roshan.jpg",
    "Image 2": r"C:\Users\harsh\Downloads\Famous-Bollywood-Actor-Akshay-Kumar.jpg",
    "Image 3": r"C:\Users\harsh\Downloads\461225-shah-rukh-khan1-1467362200.jpg",
}

style_image = {
    "Image 1": r"C:\Users\harsh\Downloads\Colour-preservation-in-Neural-Style-Transfer-a-Content-image-b-Style-image-c.png",
    "Image 2": r"C:\Users\harsh\Downloads\edtaonisl.jpg",
    "Image 3": r"C:\Users\harsh\Downloads\iStock-641678392.jpg",
}

# Define your different pages as separate functions
def page0():
    st.title("Masked Neural Style Transfer")
    st.header("ALL")
    st.image(Image.open(home_image["Image 1"]), width=800)
    st.header("Only Object")
    st.image(Image.open(home_image["Image 2"]), width=800)
    st.header("Only Background")
    st.image(Image.open(home_image["Image 3"]), width=800)
    st.header("Different Object and Background")
    st.image(Image.open(home_image["Image 4"]), width=800)
def page1():
    st.title("Neural Style Transfer(ALL)")
    option1,option2=take_user()
    if st.button("Continue"):
        output(option1,option2,0,0)


def page2():
    st.title("Masked Neural Style Transfer(Only Object)")
    option1,option2=take_user()
    if st.button("Continue"):
        output(option1,option2,0,1)

def page3():
    st.title("Masked Neural Style Transfer(Only Background)")
    option1,option2=take_user()
    if st.button("Continue"):
        output(option1,option2,0,2)
def page4():
    st.title("Masked Neural Style Transfer(different Object and Background)")
    option1,option2=take_user()
    option3 = st.selectbox("Choose an Style 2nd image",list(style_image.keys())+ ["Upload your own image"])
    if option3 == "Upload your own image":
        uploaded_file3 = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    # If the user chooses "Choose an image", display a default image
        if uploaded_file3 is not None:
            uploaded_file3.save('uploaded_file3.png')
            st.image(uploaded_file3, use_column_width=True)
            option3='uploaded_file3.png'
        # If the user chooses "Upload your own image", allow them to upload an image
    else:
        st.image(Image.open(style_image[option3]),width=200)
        option3=style_image[option3]
    if st.button("Continue"):
        output(option1,option2,option3,3)

page_list = ['---',"ALL","Only Object","Only Background","Different Object and Background"]
# Use the st.sidebar function to create a sidebar
with st.sidebar:
    st.header("Navigation")
    st.title("Select an Operation do you want to perform:")
    # Use the st.selectbox function to create a dropdown menu
    page = st.selectbox('Select Option',page_list)
    #if st.button("Continue"):
    # Use an if/elif statement to display the appropriate page based on user selection
if page == '---':
    page0()
elif page == "ALL":
    page1()
elif page == "Only Object":
    page2()
elif page == "Only Background":
    page3()
elif page == "Different Object and Background":
    page4()
