# -*- coding:utf-8 -*-

import numpy as np
from captcha.image import ImageCaptcha
import os

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


def get_text(low=3, high=10, char_set=number+alphabet+ALPHABET):
    captcha_text = []
    for _ in range(np.random.randint(low, high)):
        char = np.random.choice(char_set)
        captcha_text.append(char)
    return ''.join(captcha_text)


def generate(num=1, path='./', low=3, high=10, char_set=number+alphabet+ALPHABET):
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(num):
        captcha_text = get_text(low, high, char_set)
        image = ImageCaptcha(30*len(captcha_text), 64)
        image.write(captcha_text, path+str(i)+'_'+captcha_text+'.png')

if __name__ == "__main__":
    generate(num=100, low=3, high=10, path='./captcha3-10_train/')
    generate(num=100000, low=3, high=10, path='./captcha3-10_test/')
