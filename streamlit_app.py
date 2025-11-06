import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
le = LabelEncoder()

st.title('🤖 Machine Learning App')

st.info('This is app builds a machine learning model!')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_excel('https://github.com/afrozsamee/Predictions_price_of_FlightTickets/blob/master/Data_Train.xlsx')
  df
  
  df.dropna(inplace = True)
  df['Date_of_Journey']=pd.to_datetime(df['Date_of_Journey'])
  df['Day']=df['Date_of_Journey'].apply(lambda x: x.day)
  df['Month']=df['Date_of_Journey'].apply(lambda x: x.month)
  df['Year']=df['Date_of_Journey'].apply(lambda x: x.year)
  
  
  df['Arrival_Time']=pd.to_datetime(df['Arrival_Time'])
  df['Arrival Time']=df['Arrival_Time'].apply(lambda x: x.hour)
  df['Arrival Day']=df['Arrival_Time'].apply(lambda x: x.day)
  df['Arrival Month']=df['Arrival_Time'].apply(lambda x: x.month)
  df['Arrival Year']=df['Arrival_Time'].apply(lambda x: x.year)
  
  df['Airline']=le.fit_transform(df['Airline'])
  df['Source']=le.fit_transform(df['Source'])
  df['Destination']=le.fit_transform(df['Destination'])
  df['Route']=le.fit_transform(df['Route'])
  df['Dep_Time']=le.fit_transform(df['Dep_Time'])
  df['Duration']=le.fit_transform(df['Duration'])
  df['Total_Stops']=le.fit_transform(df['Total_Stops'])
  df['Additional_Info']=le.fit_transform(df['Additional_Info'])

  st.write('**X**')
  X_raw = df.drop('Price', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.Price
  y_raw

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Input features
with st.sidebar:
  st.header('Input features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Gender', ('male', 'female'))
  
  # Create a DataFrame for the input features
  data = {'island': island,
          'bill_length_mm': bill_length_mm,
          'bill_depth_mm': bill_depth_mm,
          'flipper_length_mm': flipper_length_mm,
          'body_mass_g': body_mass_g,
          'sex': gender}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
  st.write('**Input penguin**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins


# Data preparation
# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y
target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (input penguin)**')
  input_row
  st.write('**Encoded y**')
  y


# Model training and inference
## Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={0: 'Adelie',
                                 1: 'Chinstrap',
                                 2: 'Gentoo'})

# Display predicted species
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba,
             column_config={
               'Adelie': st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Chinstrap': st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Gentoo': st.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)


penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0]))
