#Database testing ---> python -m pytest
import managed_db

def database():
	managed_db.create_usertable()
	sample_data = [
	('abu','talha'),
	('adithya','raj')
	]

	managed_db.add_userdata(sample_data)

def test_database():
	assert len(list(managed_db.view_all_users())) == 2

#Model testing
import pickle
import pandas as pd
from sklearn import preprocessing
PATH = 'data/crop_recommendation1.csv'
df = pd.read_csv(PATH)
category_col = ['season']
labelEncoder = preprocessing.LabelEncoder() 
  
mapping_dict ={} 
for col in category_col: 
    df[col] = labelEncoder.fit_transform(df[col])
    
    le_name_mapping = dict(zip(labelEncoder.classes_,labelEncoder.transform(labelEncoder.classes_))) 
    
    mapping_dict[col]= le_name_mapping
features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'season', 'rainfall']]
target = df['label']
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
load_model = pickle.load(open('models/RandomForest1.pkl','rb'))
acc_score = load_model.score(Xtest,Ytest)

def test_random_forest():

	assert acc_score == 0.9022727272727272