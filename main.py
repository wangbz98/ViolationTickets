import streamlit as st
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

st.title('Am I getting a :red[_ticket_] for a traffic stop in WA? 😥')

### SECTION 1 - INTRODUCTION ###
st.header(':blue[_BACKGROUND AND MOTIVATION_]')
st.markdown(
"""
* Police officers make more than 50,000 traffic stops per day in the United States (_From the “Stanford Open Policing Project”_)
* The consequence of that violation depends on the punishment that the officer gives to the driver
* Factors from drivers that lead to traffic violations
    * lacking knowledge or experience
    * driver’s demographic
    * Type of violation and the number of violations they have made
* The same traffic violation might result in different punishments which depend upon the **police’s judgment**
* This study aims to find relations between :orange[**officers’ demographic and driver factors that relate to punishment in traffic stops**]
"""
)

### SECTION 2 - DATA ###
st.header(':blue[_DATA_]')
st.subheader('Data Details')
st.markdown(
"""
* The Stanford Open Policing Project - Washington State
* 8 millions records of traffic stops by officers
* 850,289 records from 2013 – 2016
"""
)
### Use to be working, but later get "HTTP Error 403: Forbidden" error
#@st.cache
#def load_raw_data():
#    url='https://drive.google.com/file/d/1SWMYje4FvPgzRBvLnOS1jXMLI9b1-OmQ/view?usp=share_link'
#    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
#    return pd.read_csv(url)
#raw_data = load_raw_data()

raw_data = pd.read_csv('sample_raw_data.csv')
st.dataframe(raw_data)
st.caption('Table 1: Sample of raw data')

st.subheader('Recode Data')
st.markdown(
"""
* Dependent variables
    * Stop outcome - Verbal or Written
* Independent variables
    * Using driver’s demographics – race, gender, and age
    * Date – days of week
    * Time – Rush or Normal
    * Violation – Significant only and Count 
* The ‘Significant violation’ variable contained 15 categories was grouped as follows:
    * Movement = [Speeding, Safe movement, Moving violation, Stop sign/light]
    * Driver = [Cell phone, Seat belt,DUI]
    * Vehicle = [Equipment, Truck, Lights]
    * Others = [Paperwork,License,Registration/plates, Other, Other (non-mapped)] 
* Group by officer’s demographics
    * Race x Gender
    * 10 groups
* Coefficients from Binary Logit Model 
"""
)
#Show variable detail table
### Use to be working, but later get "HTTP Error 403: Forbidden" error
#url='https://drive.google.com/file/d/1W2feR6A0bIcn0P2DSmKFLOM7-f9V3v1k/view?usp=share_link'
#url='https://drive.google.com/uc?id=' + url.split('/')[-2]
#variable = pd.read_csv(url, encoding= 'unicode_escape')

variable = pd.read_csv('variable_table.csv', encoding= 'unicode_escape')
st.caption('Table 2: Independent and dependent variable')
st.dataframe(variable)

#Show data management plan
st.subheader('Data Management Plan')
st.write("Three tables are used in this study. As shown in figure 1, Tabel I is raw data that contains all relevant records. Then, each event will be grouped together in table II based on the officer's demographic. The binay logistic regression model fitting were performed for each table II. The coefficient from each group will be store in Table III, which will later used for visulization and interpretation.")
#Image need to be embedded. Convert share link to embedded at https://www.labnol.org/embed/google/drive/
st.image("https://lh5.googleusercontent.com/DJbCIQLOIDQhR9PSx7Wt-NGq1oZaeOmQBK8pvSkdSXoA8cllL-FGqUbc3GEKawNaGh4=w2400", caption='Figure 1: Data Management Plan')

### SECTION 3 - ANALYSIS
st.header(':blue[_ANALYSIS_]')
st.subheader('The Binary Logistic Regression Model')
st.markdown(
"""
* The binary logistic regression model is used for exploring the relationship between a set of independent variables and a binary dependent variable 
* In this study, the dependent variable is converted from the punishment outcome,  which is either getting a written warning (denoted as 1), or a verbal warning (denoted as 0)
"""
)
st.latex(r''' log(\frac{p}{1-p})\ = b_{0} + b_{1}X_{1}+...+ b_{k}X_{k}  ''')
st.text("""
Where
\tp = Probability that Y = 1 given X 
\tY = Dependent Variable 
\tX1,X2,...,Xk = Independent variables
\tb0,b1,...,bk = Coefficient of model
""")

#Show Logistic Regression using logit function
st.subheader('Model Fitting')
st.markdown(
"""
The statsmodels.formula.api model was used for fitting the binary logit regression model. The syntax of smf.logit is as follows
"""
)

code = '''import statsmodels.formula.api as smf
predict_model = smf.logit(formula = 'stop_outcome ~ day_of_week + rush_hour + driver_age + driver_gender + driver_age + sig_violation + count_of_violation', data = df).fit() '''
st.code(code, language='python')

#Show summary
st.markdown(
"""
The model.summary() function was used for getting the analysis result. The output shows details of the fitted model
"""
)
code2 = '''predict_model.summary()'''
st.code(code2, language='python')
st.image("https://lh6.googleusercontent.com/dtQDgH1V35xz5-7Q4u6KI7YN2B5cw3FQVlJFJGSc6Kt2f_uqNGmQ2kDrZTtq8Af0UmA=w2400", caption='Figure 2: Model analysis result')

#Show coefficien result
st.subheader('Analysis Result')
st.markdown(
"""
In this study, a coefficient and P-Value for each variable are considered. Below is the plot of the relationship between officer from each group to the coefficient of each variable
"""
)
st.caption('Figure 3: The plot of relationship between officer from each group to the coefficient of each variable')

coeff = pd.read_csv('coefficient_final.csv', encoding= 'unicode_escape')
coeff = coeff.drop('Group', axis=1)

fig = plt.figure()
x = ['White M','Black M','Asian M','Hispanic M','Other M','White F','Black F','Asian F','Hispanic F','Other F']
plt.xticks(rotation=45)
Intercept = coeff.loc[:,'Intercept']
plt.plot(x,Intercept,label='Intercept')
TUE = coeff.loc[:,'[MON|TUE]']
plt.plot(x,TUE,color='green',label='[MON|TUE]')
WED = coeff.loc[:,'[MON|WED]']
plt.plot(x,WED,color='green',label='[MON|WED]')
THU = coeff.loc[:,'[MON|THU]']
plt.plot(x,THU,color='green',label='[MON|THU]')
FRI = coeff.loc[:,'[MON|FRI]']
plt.plot(x,FRI,color='green',label='[MON|FRI]')
SAT = coeff.loc[:,'[MON|SAT]']
plt.plot(x,SAT,color='green',label='[MON|SAT]')
SUN= coeff.loc[:,'[MON|SUN]']
plt.plot(x,SAT,color='green',label='[MON|SUN]')
GenderF = coeff.loc[:,'Gender[F|M]']
plt.plot(x,GenderF,color='yellow',label='Gender[F|M]')
Violation_DRIVER_MOVEMENT = coeff.loc[:,'Violation[DRIVER|MOVEMENT]']
plt.plot(x,Violation_DRIVER_MOVEMENT,color='cyan',label='Violation[DRIVER|MOVEMENT]')
Violation_DRIVER_Others = coeff.loc[:,'Violation[DRIVER|Others]']
plt.plot(x,Violation_DRIVER_Others,color='cyan',label='Violation[DRIVER|Others]')
Violation_DRIVER_VEHICLE = coeff.loc[:,'Violation[DRIVER|VEHICLE]']
plt.plot(x,Violation_DRIVER_VEHICLE,color='cyan',label='Violation[DRIVER|VEHICLE]')
Is_rush_hour = coeff.loc[:,'Is Rush hour']
plt.plot(x,Is_rush_hour,color='magenta',label='Is Rush hour')
Age = coeff.loc[:,'driver age']
plt.plot(x,Age,label='driver age')
Count_of_violation = coeff.loc[:,'Count of violation']
plt.plot(x,Count_of_violation,color='blue',label='Count of violation')
#plt.gca().legend((1,2,3,4,5,6,7,8,9,10))
fig.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels=['Intercept','[MON|TUE]','[MON|WED]','[MON|THU]','[MON|FRI]','[MON|SAT]','[MON|SUN]','Gender[F|M]','Violation[DRIVER|MOVEMENT]','Violation[DRIVER|MOVEMENT]','Violation[DRIVER|Others]','Violation[DRIVER|VEHICLE]','Is Rush hour','driver age','Count of violation'])
st.pyplot(fig)

#Show P-Value result
st.markdown(
"""
 The variables with p-value less than 0.05 are considered to be statistically significant. 
"""
)
st.caption('Table 3: P-value for each variable')
p = pd.read_csv('p-value_final.csv', encoding= 'unicode_escape')
p = p.drop('Group', axis=1)
p.index = list(['White M','Black M','Asian M','Hispanic M','Other M','White F','Black F','Asian F','Hispanic F','Other F'])
st.dataframe(p.style.highlight_between(left=0, right=0.05))
