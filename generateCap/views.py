from django.shortcuts import render
from django.http import HttpResponse
from .forms import ImageForm
from .models import ImageModel
import os
from django.conf import settings
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image


# Create your views here.
def index(request):
    return render(request, "generateCap/index.html")
    # return HttpResponse("This is the home page.")



def extract_feature(filename):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature
    
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


def pred(path):
    # print(path)

    # path = settings.MEDIA_ROOT + '/images/' + str(path)
    # print(path)
    base_path = os.path.join(settings.BASE_DIR, "generateCap", "static", "models")

    tokenizer_path = os.path.join(base_path, 'tokenizer.pkl')
    model_path = os.path.join(base_path, 'model.h5')

    tokenizer = load(open(tokenizer_path, 'rb'))
    model = load_model(model_path)

    max_length = 34
    photo = extract_feature(path)
    description = generate_desc(model, tokenizer, photo, max_length)
    d = description.split(" ")
    d = d[1:-1]
    description = " ".join(d)
    return description.title()


def generate(request):
    '''' Accepts the image and returns the caption'''
    path=''
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        print('yes')
        print(form.is_valid())
        if form.is_valid():
            print('yes')
            # print(form['image'])
            form.save()
            path = form.cleaned_data['image']
            print(path)
            new_path = os.path.join(settings.MEDIA_ROOT, 'images', 'upload.jpg')

            # Check if 'upload.jpg' already exists and delete it
            if os.path.exists(new_path):
                os.remove(new_path)
            os.rename(settings.MEDIA_ROOT + '/images/' + str(path),
                      settings.MEDIA_ROOT + '/images/' + 'upload' + '.jpg')
            path= settings.MEDIA_ROOT + '/images/' + 'upload' + '.jpg'
            print(path)
            cap1=pred(path)
            print(cap1)
            cap2=predict_step([path])
            # return HttpResponse("Image uploaded successfully.")
        

    return render(request, "generateCap/generate.html", {'caption1':cap1,'caption2':cap2,'img':path})

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds[0].title()

#Generate Captions for a Fresh Image



