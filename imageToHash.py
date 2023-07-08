# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 02:55:15 2023

@author: jonat
"""
import cv2
import hashlib
from PIL import Image
import matplotlib.pyplot as plt

def write_to_file(pathMetrcis, nameImage_1 , data = " ", create_file=False):
    nameImage_1 = nameImage_1[:-4]
    pathFileResults = pathMetrcis + "/hashValues/" + nameImage_1 + "_ValueHash" + ".txt"
    if create_file == True:
        try:
            with open(pathFileResults, "w") as file:
                print("File txt successfully created.")
        except Exception as e:
                print("An error occurred while creating  the file:", str(e))
    else:
        try:
            with open(pathFileResults, "a+") as file:
                file.write(data + '\n')
        except Exception as e:
            print("An error occurred while writing to the file:", str(e))
            
def showImage(imageCV):
    plt.imshow(cv2.cvtColor(imageCV, cv2.COLOR_BGR2RGB))
    plt.axis("off") 
    plt.show()

def imageToHash (pathImage):
    nameHashNoise = pathImage[:-4] + "_hashNoise" + pathImage[-4:]
    image = cv2.imread(pathImage)
    # image = cv2.resize(image, (240, 240)) # Uncomment if you need to normlize the input image
    showImage(image)
    size1, size2, channels = image.shape
    sizeHashImage = (size2,size1)
    hash_object = hashlib.blake2s(image, digest_size=16)
    hash_value = hash_object.hexdigest()
    write_to_file(pathImage[:10], pathImage[20:], data = " ", create_file=True)
    write_to_file(pathImage[:10], pathImage[20:], ("Image Hash:" + hash_value))
    print("Image Hash:", hash_value)
    hash_bytes = hashlib.md5(hash_value.encode()).digest()
    imageHash = Image.new("RGB", sizeHashImage)
    pixels = []
    for i in range(sizeHashImage[0]):
        for j in range(sizeHashImage[1]):
            index = (i * sizeHashImage[1] + j) % len(hash_bytes)
            r = hash_bytes[index]
            g = hash_bytes[(index + 1) % len(hash_bytes)]
            b = hash_bytes[(index + 2) % len(hash_bytes)]
            pixels.append((r, g, b))
    imageHash.putdata(pixels)
    imageHash.save(nameHashNoise)
    imageHash = cv2.imread(nameHashNoise)
    showImage(imageHash)
 
pathImage = "../NoiseHashGAN/Key_image/person1_.jpg"
imageToHash (pathImage)






