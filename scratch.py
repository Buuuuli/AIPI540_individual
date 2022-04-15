from Call_Function import *
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import dash
import pandas as pd
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from Call_Function import *
import pickle as pkl
import torch

# ML Part
#filepath = 'Model/randomforest.sav'

#loaded_model = pkl.load(open(filepath, 'rb'))

#age_input = 55
#sex_input = 'male'
#localization_input = 'chest'


#a = meta_pipeline(age_input,sex_input,localization_input)


#test_preds = loaded_model.predict(a).reshape(len(a),1)


#meta__prob = loaded_model.predict_proba(a)



#if test_preds[0][0] ==4:
    #print('Including Melanoma')
#else:
    #print('Benign Keratosis')





#DL Part



model2 = torch.load('Model/deep_fullmodel_new.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image = 'webelement/ISIC_0024315.jpg'


b = image_pipeline(image)


j,k = test_model(model2, b, device)


