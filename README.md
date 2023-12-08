# Heart_Rhythm_Classification_From_Raw_ECG_Signals_AML2
 Project 2 for the course of Advanced Machine Learning. Heart rhythm classification from manually perturbed ECG Signals. Evaluation is based on F1-score.

 #RESULTS 8th team out of 121 teams. Score: 0.845 (1st team score: 0.852. Last team score: 0.577).

 #REPORT 
  Preprocessing:
   - We implemented the paper "ENCASE: an ENsemble CLASsifiEr for ECG Classification using expert features and deep neural networks" to extract 64 features from the data
   - Furthermore we used the following approach to extract further features:
    - every row of the dataset
    - is stripped of its missing value
    - the resulting signal is cleaned using the fuction ecg_clean of neurokit
    - using the function ecg_peaks of neurokit and delineate, the peaks are extracted from the signal
    - from the resulting preprocessed signal, the main features are extracted
    - all the features are scaled appropriately using MinMaxScaler from sklearn
  
  Classification:
   - the classification task is done by majority voting based on 4 previous submissions, 1 of which was produced using xgboost, one using random forest, one using adaboost and one an ensemble implemented through StackingClassifier from sklearn of the following 5 models:
    - catboost (CatBoostClassifier from catboost with objective 'MultiClass', 300 iterations, random_state=0, eval_metric 'TotalF1:average=Micro')
    - extratrees (ExtraTreesClassifier from sklearn with 1000 estimators)
    - support vector machine (SVC from sklearn with default parametheres)
    - xgboost (XGBClassifier from xgboost with 300 estimators, objective='multi:softmax'; and metric f1_score)
    - random forest (RandomForestClassifier from sklearn with 300 estimators)
