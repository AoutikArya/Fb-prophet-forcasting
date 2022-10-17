"""System module."""
import base64
import pandas as pd
from prophet import Prophet
import streamlit as st

st.title('Future Forcasting')
st.write('Made using *[Fbprophet](https://facebook.github.io/prophet/)!*:sunglasses:')
df = st.file_uploader("upload CSV file", type={"csv"})

col1, col2, col3 = st.columns(3)
if df is not None:
    df = pd.read_csv(df)
    ctm=list(df.columns.values)
    ctm=[e.lower() for e in ctm ]
    if 'date' not in ctm :
        st.write('Invalid dataset no feature like date')
    else:
        with col1:
            y=st.selectbox('Select a feature to be forcasted',
                           df.columns.values[df.columns.str.lower()!='date'])
        with col2:
            seasonality=st.selectbox('Choose seasonality',['Daily','Monthly','Yearly'])
        with col3:
            if seasonality=='Daily':
                period=st.selectbox('Period',range(1,3651))
            elif seasonality=='Monthly':
                period=st.selectbox('Period',range(1,121))
            else:
                period=st.selectbox('Period',range(1,11))
    df.columns= df.columns.str.lower()
    df.rename(columns={'date':'ds',y.lower():'y'},inplace=True)
    df=df[['ds','y']]
    df['ds']=pd.to_datetime(df['ds'])
    idx=len(df)-1
    model=Prophet()
    model.fit(df)
    if seasonality=='Daily':
        future=model.make_future_dataframe(periods=period,freq='D')
    elif seasonality=='Monthly':
        future=model.make_future_dataframe(periods=period,freq='M')
    else:
        future=model.make_future_dataframe(periods=period,freq='Y')
    predict=model.predict(future)
    idx2=len(predict)-1
    if st.button('Predict'):
        st.write(f"Value changed from   **{round(df['y'][idx],2)}**   as on {df['ds'][idx]} to 
                 ***{round(predict['yhat'][idx2],2)}***   as on {predict['ds'][idx2]}")
        st.pyplot(model.plot(predict,xlabel='Date',ylabel=y,include_legend=True))
        st.pyplot(model.plot_components(predict))
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f'''
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    ''',
    unsafe_allow_html=True
    )
add_bg_from_local('2.jpg')
