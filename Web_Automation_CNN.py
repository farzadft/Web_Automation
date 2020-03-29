from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import parser 
import xml
from nltk.tokenize import sent_tokenize, word_tokenize
import os 
from PIL import Image
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as scaler
import numpy as np

def crawl_images(name):
    if os.path.isdir('/Users/farzadtehrani/Desktop/'+name)==True:
        pass
    else:
        os.mkdir('/Users/farzadtehrani/Desktop/'+name)
        web_pages=['https://www.google.com/search?q={}&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjR3Mf1krToAhUUAZ0JHcOZBQoQ_AUoAXoECA8QAw&biw=1440&bih=821'.format(name),'https://ca.images.search.yahoo.com/search/images;_ylt=AwrJ7KK.jHteh7QALF_rFAx.;_ylu=X3oDMTB0N2Noc21lBGNvbG8DYmYxBHBvcwMxBHZ0aWQDBHNlYwNwaXZz?p={}&fr2=piv-web&fr=yfp-t'.format(name),
                  'https://www.google.com/search?q={}&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjR3Mf1krToAhUUAZ0JHcOZBQoQ_AUoAXoECA8QAw&biw=1440&bih=821'.format(name+'+'+'show more'+'+'),'https://ca.images.search.yahoo.com/search/images;_ylt=AwrJ7KK.jHteh7QALF_rFAx.;_ylu=X3oDMTB0N2Noc21lBGNvbG8DYmYxBHBvcwMxBHZ0aWQDBHNlYwNwaXZz?p={}&fr2=piv-web&fr=yfp-t'.format(name+'+'+'show more'+'+')]
        soup=[BeautifulSoup(requests.get(i).text,'html.parser').select('img') for i in web_pages]
        source=[]
        for i in soup:
            source.append(str(i).split())
        link=[]
        for i in source: 
            for j in range(len(i)):
                splitting=i[j].split()
        
                for k in range(len(splitting)):
                    if splitting[k].find('src="https:'):
                        pass
                    else:
                        m=splitting[k].replace('src="','')
                        link.append(m.replace('"',''))
    
        for index, img_link in enumerate(link):
            img_data=requests.get(img_link).content
            with open('/Users/farzadtehrani/Desktop/'+name+'/'+str(index+1)+'.jpg','wb+') as f:
                 f.write(img_data)
        
                
            
    return ('/Users/farzadtehrani/Desktop/'+name)

def resize(name):
    for c in os.listdir(crawl_images(name)):
        image = Image.open(crawl_images(name)+'/'+c)
        new_image = image.resize((200, 200))
        new_image.save(crawl_images(name)+'/'+c)
    return ('/Users/farzadtehrani/Desktop/'+name)
    
def main(test_set_folder):
    number_of_classes=int(input())
    x=[]
    
    x_t=[]
    classes=[]
    for i in range(number_of_classes):
        class_=str(input())
        classes.append(class_)
        for c in os.listdir(resize(class_)):
            k=mpimg.imread(resize(class_)+'/'+c),i
            x.append(k)
            
    for j in os.listdir(test_set_folder):
        image = Image.open(test_set_folder+'/'+j)
        new_image = image.resize((200, 200))
        new_image.save(test_set_folder+'/'+j)
           
    for m in os.listdir(test_set_folder):
        x_t.append(mpimg.imread(test_set_folder+'/'+m))
            
    np.random.shuffle(x)
    y=[x[i][1] for i in range(len(x))]
    x_f=[x[i][0] for i in range(len(x))]
    x_train= tf.keras.utils.normalize(x_f)
    x_test=tf.keras.utils.normalize(x_t)
    
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(x_train.shape[0],activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(x_train.shape[0],activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(number_of_classes,activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train,np.array(y), epochs=100)
    predictions=model.predict([x_test])
    pred=[np.argmax(i) for i in predictions]
    final_pred=[classes[i] for i in pred ]
    return pd.DataFrame(final_pred)
