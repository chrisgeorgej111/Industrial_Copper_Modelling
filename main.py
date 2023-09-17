import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score,f1_score,roc_auc_score,classification_report
import re

# reading the csv file and Data_cleaning,type_casting

df=pd.read_csv('Copper_Set.xlsx - Result 1.csv')
non_meaning=df['material_ref'].str.startswith('00000')
df['material_ref'][non_meaning==True]=np.NaN
df['material_ref'].fillna(0,inplace=True)
df.rename(columns={'quantity tons':'quantity_tons','item type':'item_type','delivery date':'delivery_date'},inplace=True)
df['quantity_tons'].iloc[173086]=0
df['quantity_tons'].iloc[173086] = df['quantity_tons'].median()
df['item_date']=pd.to_datetime(df['item_date'],format='%Y%m%d',errors='coerce').dt.date
df['quantity_tons']=df['quantity_tons'].astype(float)
df['delivery_date']=pd.to_datetime(df['delivery_date'],format='%Y%m%d',errors='coerce').dt.date
df_1=df.drop(columns=['id','material_ref'])

#Imputation

df_1['item_date']=df_1['item_date'].fillna(df_1['item_date'].mode()[0])
df_1['delivery_date']=df_1['delivery_date'].fillna(df_1['delivery_date'].mode()[0])

cat=['country','status','product_ref','item_type']
for i in cat:
    df_1[i].fillna(df_1[i].mode()[0],inplace=True)

neg_value = df_1['selling_price'] <= 0

df_1.loc[neg_value, 'selling_price'] = np.nan

neg_value_1= df_1['quantity_tons'] <= 0

df_1.loc[neg_value_1, 'quantity_tons'] = np.nan

numeric=['application','quantity_tons','customer','thickness','width','selling_price']
for i in numeric:
    df_1[i].fillna(df_1[i].mean(),inplace=True)
#Transformation of data
df_1['tr_selling_price']=np.log(df_1['selling_price'])
df_1['tr_quantity_tons']=np.log(df_1['quantity_tons'])
df_1['tr_thickness']=np.log(df_1['thickness'])
#Feature Engineering

df_1['delivery_time'] = (df_1['delivery_date'] - df_1['item_date']).abs().dt.days
df_1['total_amount'] = df_1['quantity_tons'] * df_1['selling_price']


# Regression
# Data Pre_Processing
x=df_1[['tr_quantity_tons','customer','country','status','item_type','application','tr_thickness','width','delivery_time','total_amount']]
ohe=OneHotEncoder(handle_unknown='ignore')
ohe.fit(x[['item_type']])
x_item = ohe.fit_transform(x[['item_type']]).toarray()
ohe_st = OneHotEncoder(handle_unknown='ignore')
ohe_st.fit(x[['status']])
x_st= ohe_st.fit_transform(x[['status']]).toarray()
X=np.concatenate((x[['tr_quantity_tons','customer','country','application','tr_thickness','width','delivery_time','total_amount']].values,x_item,x_st),axis=1)
Y=df_1['tr_selling_price']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

# Training and developing regression model for this data
reg= ExtraTreesRegressor(n_estimators=100,criterion='friedman_mse',max_features=10)
reg.fit(x_train,y_train)
Y_pred=reg.predict(x_test)

# Evaluation Metrics for this model
mse=mean_squared_error(y_test,Y_pred)
r2=r2_score(y_test,Y_pred)

# Saving the model
import pickle
with open('regressor.pkl', 'wb') as file:
    pickle.dump(reg, file)
with open('reg_encoder.pkl', 'wb') as f:
    pickle.dump(ohe, f)
with open('reg_encoder1.pkl', 'wb') as f:
    pickle.dump(ohe_st, f)


# Classification
# Data Preprocessing
df_2=df_1.copy()
not_required=df_2[(df_2['status']!='Won') &(df_2['status']!='Lost')].index
df_2.drop(not_required,inplace=True)
x=df_2[['tr_quantity_tons','customer','country','item_type','application','tr_thickness','width','delivery_time','total_amount','selling_price']]
ohe=OneHotEncoder(handle_unknown='ignore')
ohe.fit(x[['item_type']])
ohe_c_item=ohe.fit_transform(x[['item_type']]).toarray()
X=np.concatenate((x[['tr_quantity_tons','customer','country','application','tr_thickness','width','delivery_time','total_amount','selling_price']].values,ohe_c_item),axis=1)
Y=df_2['status']
x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size= 0.2, random_state=1)

# Training and developing classification model for this data
clas= ExtraTreesClassifier(n_estimators=100,criterion='entropy',max_features=10)
clas.fit(x_train,y_train)
y_pred=clas.predict(x_test)
acc_score=accuracy_score(y_test,y_pred)
classification_report=classification_report(y_test,y_pred)

#Saving the Model

import pickle
with open('classifier.pkl', 'wb') as file:
    pickle.dump(clas, file)
with open('classifier_encoder.pkl', 'wb') as f:
    pickle.dump(ohe, f)


st.set_page_config(layout="wide")

st.title("INDUSTRIAL COPPER MODELLING")

tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"])

with tab1:
    status_select = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered',
                      'Offerable']
    item_type_select = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_select = [25.0,26.0,27.0,28.0,30.0,32.0,38.0,39.0,40.0,77.0,78.0,79.0,80.0,84.0,89.0,107.0,113.0]
    application_select = [10., 41., 28., 59., 15.,4., 38, 56.,42., 26.,27., 19., 20., 66., 29.,22., 40., 25., 67., 79.,3.,
                           99.,  2.,  5., 39.,69., 70., 65., 58., 68.,25.61580851]

    with st.form("my_form"):
        col1, col2 = st.columns([3,3])
        with col1:
            st.write(' ')
            status = st.selectbox("Status", status_select, key=1)
            item_type = st.selectbox("Item Type", item_type_select, key=2)
            country = st.selectbox("Country", country_select, key=3)
            application = st.selectbox("Application", sorted(application_select), key=4)

        with col2:
            quantity_tons = st.text_input("Enter Quantity Tons (Min:0 & Max:1000000000)")
            thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            width = st.text_input("Enter width (Min:1, Max:2990)")
            customer = st.text_input("customer ID (Min:12458, Max:30408185)")
            delivery_time=st.text_input("Time (Min:0, Max:689)")
            total_amount=st.text_input("amount (Min:0, Max:583000000000)")

            submit= st.form_submit_button(label="PREDICT SELLING PRICE")

        flag = 0
        pattern = '[0-9]*\.?[0-9]+'
        for i in [quantity_tons, thickness, width, customer,delivery_time,total_amount]:
            if re.match(pattern, i):
                pass
            else:
                flag = 1
                break

    if submit and flag == 1:
        if len(i) == 0:
            st.write("please enter a valid number space not allowed")
        else:
            st.write("You have entered an invalid value: ", i)

    if submit and flag == 0:
        import pickle

        with open(r"regressor.pkl", 'rb') as file:
            loaded_model = pickle.load(file)

        with open(r"reg_encoder.pkl", 'rb') as f:
            oh_loaded = pickle.load(f)

        with open(r"reg_encoder1.pkl", 'rb') as f:
            oh_1_loaded = pickle.load(f)

        new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)), float(width),
                                country, float(customer),int(delivery_time), int(total_amount),item_type, status]])
        new_sample_ohe = oh_loaded.transform(new_sample[:, [8]]).toarray()
        new_sample_ohe_1 = oh_1_loaded.transform(new_sample[:, [9]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5,6,7]], new_sample_ohe, new_sample_ohe_1), axis=1)
        new_pred = loaded_model.predict(new_sample)[0]
        st.write('## :green[Predicted selling price:] ', round(np.exp(new_pred)))

with tab2:
    with st.form("my_form1"):
        col1, col2 = st.columns([3, 3])
        with col1:
            c_quantity_tons = st.text_input("Enter Quantity Tons (Min:0 & Max:1000000000)")
            c_thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            c_width = st.text_input("Enter width (Min:1, Max:2990)")
            c_customer = st.text_input("customer ID (Min:12458, Max:30408185)")
            c_selling = st.text_input("Selling Price (Min:1, Max:100001015)")
            c_delivery_time=st.text_input("Time (Min:0, Max:689)")
            c_total_amount=st.text_input("amount (Min:0, Max:583000000000)")

        with col2:
            st.write(' ')
            c_item_type = st.selectbox("Item Type", item_type_select, key=5)
            c_country = st.selectbox("Country", country_select, key=6)
            c_application = st.selectbox("Application", sorted(application_select), key=7)
            c_submit = st.form_submit_button(label="PREDICT STATUS")

        c_flag = 0
        pattern = '[0-9]*\.?[0-9]+'
        for k in [c_quantity_tons, c_thickness, c_width, c_customer, c_selling, c_delivery_time,c_total_amount]:
            if re.match(pattern, k):
                pass
            else:
                c_flag=1
                break
    if c_submit and flag == 1:
        if len(k) == 0:
            st.write("please enter a valid number space not allowed")
        else:
            st.write("You have entered an invalid value: ", k)

    if c_submit and flag == 0:
        import pickle

    with open(r"classifier.pkl", 'rb') as file:
        c_loaded_model = pickle.load(file)

    with open(r"classifier_encoder.pkl", 'rb') as f:
        oh_loaded = pickle.load(f)

        new_sample = np.array([[np.log(float(c_quantity_tons)), np.log(float(c_selling)), c_application,
                                np.log(float(c_thickness)), float(c_width), c_country, int(c_customer),int(c_delivery_time),int(c_total_amount),
                                c_item_type]])
        new_sample_ohe = oh_loaded.transform(new_sample[:, [9]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7,8]], new_sample_ohe), axis=1)

        new_pred = c_loaded_model.predict(new_sample)
        if new_pred.any() == 1:
            st.write('## :green[The Status is Won] ')
        else:
            st.write('## :red[The status is Lost] ')




















































