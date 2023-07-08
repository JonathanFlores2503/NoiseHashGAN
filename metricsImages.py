#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 02:49:30 2023

@author: pc
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import structural_similarity as compare_ssim

def generate_dir(*args):
    if len(args) == 1:
        dir_general = args[0]
        try:
            os.mkdir(dir_general)
        except OSError:
            pass
            print("Failed to create directory %s or it already exists" % dir_general)
        else:
            pass
            print("Directory created: %s" % dir_general)
    if len(args) == 2:
        dir_general = args[0] + args[1]
        try:
            os.mkdir(dir_general)
        except OSError:
            pass
            print("Failed to create directory %s or it already exists" % dir_general)
        else:
            pass
            print("Directory created: %s" % dir_general)
    if len(args) == 3:
        dir_general = args[0] + args[1] + args[2]
        try:
            os.mkdir(dir_general)
        except OSError:
            pass
            print("Failed to create directory %s" % dir_general)
        else:
            pass
            print("Directory created: %s" % dir_general)
    return dir_general

def write_to_csv(file_path, data, create_file=False):
    mode = 'a' if not create_file else 'w'
    try:
        df = pd.DataFrame(data)
        df.to_csv(file_path, mode=mode, header=create_file, index=False)
        print("Data successfully added to the CSV file.")
    except Exception as e:
        print("An error occurred while writing to the CSV file:", str(e))
        
def write_to_file(nameImage_1, nameImage_2 , data = " ", create_file=False):
    nameImage_1 = nameImage_1[:-4]
    nameImage_2 = nameImage_2[:-4]
    pathFileResults = pathMetrcis + "/" + nameImage_1 + "-" + nameImage_2 + ".txt"
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
    
def rawDataImage (pathDirGeneral, nameImage_1, nameImage_2):
    image1 = cv2.imread(pathDirGeneral + nameImage_1)
    image2 = cv2.imread(pathDirGeneral + nameImage_2)
    pass_flag = 0
    length_1, width_1, channels_1 = image1.shape
    length_2, width_2, channels_2 = image2.shape
    write_to_file(nameImage_1, nameImage_2 , ("Raw data of image: " + str(nameImage_1)))
    write_to_file(nameImage_1, nameImage_2 , ("Width: " + str(width_1) + " Length: " + str(length_1) + " Number of channels: " + str(channels_1)))
    write_to_file(nameImage_1, nameImage_2 , ("Raw data of image: " + str(nameImage_2)))
    write_to_file(nameImage_1, nameImage_2 , ("Width: " + str(width_2) + " Length: " + str(length_2) + " Number of channels: " + str(channels_2)))
    if length_1 == length_2 and width_1 == width_2 and channels_1 == channels_2:
        write_to_file(nameImage_1, nameImage_2 , "The images have the same raw values")
        pass_flag = 1
        width = width_1
        length = length_1
        channels = channels_1
    else:
        write_to_file(nameImage_1, nameImage_2 , "The images don't have the same raw values")
    if pass_flag == 1:
        dataPixel(nameImage_1, nameImage_2,image1, image2, channels, width , length)
    return channels, width , length

def dataPixel(nameImage_1, nameImage_2, image1, image2, channels, width , length):
    image_1_numpy = np.array(image1)
    image_2_numpy = np.array(image2)
    test = True
    countPixelDifferent = 0
    for i in range(channels):
        for j in range(width):
            for k in range(length):
                if image_1_numpy[k, j, i] != image_2_numpy[k, j, i]:
                    test = False
                    countPixelDifferent += 1
    if test == True:
        write_to_file(nameImage_1, nameImage_2 , "The pixel values of the images are the same")
    else:
        totalPixel = channels * width * length
        pocentDifferentPixel = (totalPixel-countPixelDifferent) * 100 / totalPixel
        write_to_file(nameImage_1, nameImage_2 , "The pixel values of the images are different")
        write_to_file(nameImage_1, nameImage_2 , ("The percentage of equal pixels in images are: {:.3f}%".format(pocentDifferentPixel)))

def dataChannel(pathDirGeneral, nameImage_1, nameImage_2, channels, width , length, pathMetrcis):
    image1 = cv2.imread(pathDirGeneral + nameImage_1)
    image2 = cv2.imread(pathDirGeneral + nameImage_2)
    for i in range(channels):
        histogram_1, _ = np.histogram(image1[:, :, i].flatten(), bins=256, range=[0, 256])
        histogram_2, _ = np.histogram(image2[:, :, i].flatten(), bins=256, range=[0, 256])
        distance = cv2.norm(histogram_1, histogram_2, cv2.NORM_L2)
        if i == 0:
            channelName = "green"
        elif i == 2:
            channelName = "red"
        else:
            channelName = "blue"
        if distance == 0:
            write_to_file(nameImage_1, nameImage_2 , ("The histograms of the " + channelName + " channel are the same"))
        else:
            write_to_file(nameImage_1, nameImage_2 , ("The histograms of the " + channelName + " channel are different"))
        
        plt.figure()
        plt.plot(histogram_1, color='red')
        plt.plot(histogram_2, color='blue')
        plt.title("Comparison of histograms for the " + channelName + " channel of the images")
        plt.xlabel("Levels of the " + channelName + " channel")
        plt.ylabel("Frequency")
        plt.text(0.5, 0.9, 'Histogram of ' + nameImage_1, color='red', transform=plt.gca().transAxes)
        plt.text(0.5, 0.8, 'Histogram of ' + nameImage_2, color='blue', transform=plt.gca().transAxes)
        plt.savefig(pathMetrcis + "/Histogram_"+channelName+"_" + nameImage_1 + "-" + nameImage_2 + ".png")
        plt.show()

def dataChannelGray(pathDirGeneral, nameImage_1, nameImage_2, width , length, pathMetrcis):
    image1 = cv2.imread(pathDirGeneral + nameImage_1)
    image2 = cv2.imread(pathDirGeneral + nameImage_2)
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    histogram_1, _ = np.histogram(gray_image1.flatten(), bins=256, range=[0, 256])
    histogram_2, _ = np.histogram(gray_image2.flatten(), bins=256, range=[0, 256])
    distance = cv2.norm(histogram_1, histogram_2, cv2.NORM_L2)
    if distance == 0:
        write_to_file(nameImage_1, nameImage_2 , ("The histograms of the grayscale images are the same"))
    else:
        write_to_file(nameImage_1, nameImage_2 , ("The histograms of the grayscale images are different"))
    plt.figure()
    plt.plot(histogram_1, color='red')
    plt.plot(histogram_2, color='blue')
    plt.title('Histograms of the grayscale images')
    plt.xlabel('Gray Levels')
    plt.ylabel('Frequency')
    plt.text(0.5, 0.9, 'Histogram of ' + nameImage_1, color='red', transform=plt.gca().transAxes)
    plt.text(0.5, 0.8, 'Histogram of ' + nameImage_2, color='blue', transform=plt.gca().transAxes)
    plt.savefig(pathMetrcis + "/Histogram_GRAY_" + nameImage_1 + "-" + nameImage_2 + ".png")
    plt.show()

def metric_SSIM(pathDirGeneral, nameImage_1, nameImage_2):
    image1 = cv2.imread(pathDirGeneral + nameImage_1,cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(pathDirGeneral + nameImage_2,cv2.IMREAD_GRAYSCALE)
    ssim_value = ssim(image1, image2)
    write_to_file(nameImage_1, nameImage_2 , ("The SSIM value of " + nameImage_1 + " to " + nameImage_2 + " is: {:.5f}".format(ssim_value)))

def metric_DSSIM(pathDirGeneral, nameImage_1, nameImage_2):
    image1 = cv2.imread(pathDirGeneral + nameImage_1,cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(pathDirGeneral + nameImage_2,cv2.IMREAD_GRAYSCALE)
    dssim = 1 - compare_ssim(image1, image2, multichannel = False)
    write_to_file(nameImage_1, nameImage_2 , ("The DSSIM value of " + nameImage_1 + " to " + nameImage_2 + " is: {:.5f}".format(dssim)))
    
def metric_RMSE(pathDirGeneral, nameImage_1, nameImage_2):
    image1 = cv2.imread(pathDirGeneral + nameImage_1)
    image2 = cv2.imread(pathDirGeneral + nameImage_2)
    differenceImage = image1.astype(float) - image2.astype(float)
    rmse = np.sqrt(np.mean(differenceImage ** 2))
    write_to_file(nameImage_1, nameImage_2 , ("The RMSE value of " + nameImage_1 + " and " + nameImage_2 + " is: {:.5f}".format(rmse)))

def metric_MAE(pathDirGeneral, nameImage_1, nameImage_2):
    image1 = cv2.imread(pathDirGeneral + nameImage_1)
    image2 = cv2.imread(pathDirGeneral + nameImage_2)
    differenceImage = np.abs(image1.astype(float) - image2.astype(float))
    mae = np.mean(differenceImage)
    write_to_file(nameImage_1, nameImage_2 , ("The MAE value of " + nameImage_1 + " and " + nameImage_2 + " is: {:.5f}".format(mae)))
    
def main():
    generate_dir(pathMetrcis)
    write_to_file(nameImage_1, nameImage_2, create_file=True)
    channels, width , length = rawDataImage (pathDirGeneral, nameImage_1, nameImage_2)
    dataChannel(pathDirGeneral, nameImage_1, nameImage_2, channels, width , length, pathMetrcis)
    dataChannelGray(pathDirGeneral, nameImage_1, nameImage_2, width , length, pathMetrcis)
    metric_SSIM(pathDirGeneral, nameImage_1, nameImage_2)
    metric_DSSIM(pathDirGeneral, nameImage_1, nameImage_2)
    metric_RMSE(pathDirGeneral, nameImage_1, nameImage_2)
    metric_MAE(pathDirGeneral, nameImage_1, nameImage_2)
    
if __name__ == '__main__':
    pathDirGeneral = "../NoiseHashGAN/TestDemo/Comparative_2/Person/"
    filesNames = os.listdir(pathDirGeneral)
    for i in range(0, len(filesNames)-1):
        for j in range(i, len(filesNames)-1):
            print("i: {}".format(filesNames[i]) + " j: {}".format(filesNames[j+1]))
            # nameImage_1 = "person_1_GAN_7.png" # Changes
            # nameImage_2 = "person_1_GAN_8.png" # Changes
            nameImage_1 = filesNames[i] # Changes
            nameImage_2 = filesNames[j+1]
            pathMetrcis = pathDirGeneral + nameImage_1[:-4] + "-" + nameImage_2[:-4]
            main()

