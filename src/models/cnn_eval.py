"""
Convolutional Neural Network to classify Amazon dataset
Predicting new images

Created on TUE Apr 30 2021     10:00:00

@author: micheldearaujo

"""

# Importing the library
import os
import sys

sys.path.append("./")
from src.utilities import *

model_name = 'cnn_%s_SGD.h5'%(targ_shape[0])

# Load the file with the images names and labels
mapping_csv = pd.read_csv(base_dir + '/' + 'train_classes.csv')

# Loading the compiling the previously trained model
modelo = load_model(base_dir+'/'+model_name, compile=False)
modelo.compile(optimizer=opt, loss='binary_crossentropy', metrics=[fbeta])

# Loading the image-label dictionary
mapping = create_file_mapping(mapping_csv)

# Loading the label-number dictionary
labels_map, inv_labels_map = create_tag_map(mapping_csv)

# Loading the testset
Xte, yte = load_dataset(test_dataset_name, False)

# Creating a list with all possible labels
all_labels = []
for i in range(len(inv_labels_map)):
    all_labels.append(inv_labels_map[i])

# Defining the thresholds
thresholds = [0.5]

# Creating some counters
TPl, FPl, TNl, FNl = [], [], [], []
precisionl, recalll, accl, f1_scorel = [], [], [], []

# Initialing a loop to classify all images throughout all thresholds
start_time = time.monotonic()
for threshold in thresholds:
    TP, FP, TN, FN = 0, 0, 0, 0
    print('Loop: ',threshold)
    #for image_no in range(len(Xte), 2*len(Xte)): #8096
    for img_name in os.listdir(test_dir):
        # Loading the test image
        #img_name = 'train_%s.jpg'%image_no
        img = load_img(test_dir+'/'+img_name, target_size=targ_shape)
        imgarray = img_to_array(img)
        imgarray = imgarray.reshape((1,)+imgarray.shape) # Alterando a dimensão, agora é um vetor unidimensional
        imgarray = imgarray/255

        # Classifying the new image
        predicted_labels = modelo.predict(imgarray)

        # Creating a list with images true labels
        true_labels = mapping[img_name[:-4]]

        # Creating a ordered list with the images true labels and the other labels
        true_labels_list = [0 for i in range(len(all_labels))]
        for label_ in true_labels:
            index_ = all_labels.index(label_)
            true_labels_list[index_] = 1

        # Creating a dataframe to organize all the info
        df_labels = pd.DataFrame(all_labels, columns=['Lbels'])
        df_labels['True_labels'] = pd.Series(true_labels_list)
        df_labels['Predicted_proba'] = pd.Series(predicted_labels[0])

        # Defining as 1 the labels with probability > threshold
        def enconder(probabilidade):
            if probabilidade>threshold:
                return 1
            else:
                return 0
        df_labels['Predicted_labels'] = df_labels['Predicted_proba'].apply(enconder)

        # Calculating the TP, FP, TN, FN
        TP += len(df_labels[(df_labels['True_labels'] == 1) & (df_labels['Predicted_labels'] == 1)])
        FP += len(df_labels[(df_labels['True_labels'] == 0) & (df_labels['Predicted_labels'] == 1)])
        TN += len(df_labels[(df_labels['True_labels'] == 0) & (df_labels['Predicted_labels'] == 0)])
        FN += len(df_labels[(df_labels['True_labels'] == 1) & (df_labels['Predicted_labels'] == 0)])
        end_time = time.monotonic()

        print('Evaluation time: ')
        print('\n')
        print(timedelta(seconds=end_time - start_time))

    # Calculating the metrics
    # Precision
    precision = round(TP / (TP + FP), 3)

    # Recall (Sensibility ou True Positive Rate)
    recall = round(TP / (TP + FN), 3)

    # Accuracy (Percentage of Trues)
    acc = round((TP+TN)/(TP + FP + TN + FN), 3)

    # F1 Score (Weighted average betewwn precision and recall)
    f1_score = round(2 * (precision * recall) / (precision + recall), 3)

    # Adding the results of the j-image to the lists
    TPl.append(TP)
    FPl.append(FP)
    TNl.append(TN)
    FNl.append(FN)
    precisionl.append(precision)
    recalll.append(recall)
    accl.append(acc)
    f1_scorel.append(f1_score)

# Creating a dataframe with all scores as function of threshold
dic = {'Avg TP':TPl,
       'Avg FP':FPl,
       'Avg TN':TNl,
       'Avg FN':FNl,
       'Avg Precision':precisionl,
       'Avg Recall':recalll,
       'Avg Accuracy':accl,
       'Avg F1_score':f1_scorel}
final_scores = pd.DataFrame(dic, index=thresholds)
final_scores.to_csv(base_dir+'/'+'cnn_scores_%s.csv'%(targ_shape[0]))
