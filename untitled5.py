import pandas as pd 
import numpy as np
from pandas._libs import index
import sklearn 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from tkinter import *
#from tkinter import ttk
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tkinter as tk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("C:\\Users\\PC\\Downloads\\CustomersDataset (1).csv")
print(df.loc[5, :])

#preprocessing

df['Partner'].replace(['Yes','No'],[1,0], inplace=True)
df['gender'].replace(['Female','Male'],[1,0],inplace=True)
df['Dependents'].replace(['Yes','No'],[1,0],inplace=True)
df['PhoneService'].replace(['Yes','No'],[1,0],inplace=True)
df['PaperlessBilling'].replace(['Yes','No'],[1,0],inplace=True)
df['Churn'].replace(['Yes','No'],[1,0],inplace=True)
df['MultipleLines'].replace(['No phone service','No','Yes'],[2,0,1],inplace=True)
df['InternetService'].replace(['DSL','Fiber optic','No'],[2,1,0],inplace=True)
df['OnlineSecurity'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
df['DeviceProtection'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
df['TechSupport'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
df['StreamingMovies'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
df['StreamingTV'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
df['Contract'].replace(['Month-to-month','One year','Two year'],[0,1,2],inplace=True)
df['PaymentMethod'].replace(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'],[0,1,2,3],inplace=True)
df['OnlineBackup'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)

df = df.drop('customerID' , axis = 1)

df['TotalCharges'] = LabelEncoder().fit_transform(df['TotalCharges'])



#data scaling
colms=['SeniorCitizen','tenure','MonthlyCharges','gender','Churn','OnlineBackup','PaymentMethod','Contract','StreamingTV','StreamingMovies','TechSupport','DeviceProtection','OnlineSecurity','InternetService','MultipleLines','PaperlessBilling','PhoneService','Dependents','Partner']
from sklearn.preprocessing import StandardScaler
std_scalar= StandardScaler()
Stand_Sc = std_scalar.fit_transform(df[colms].iloc[:,range(0,19)].values)
sns.kdeplot(Stand_Sc[:,5],fill=True,color='red')
plt.xlabel('stand value -PaymentMethod')
plt.show()



#feature extraction    
#df = df.drop('customerID', axis='columns')
#df['TotalCharges'] = LabelEncoder().fit_transform(df['TotalCharges'])

X = df.drop('Churn', axis='columns')
Y = df['Churn']
X = pd.get_dummies(X, prefix_sep='_')
Y = LabelEncoder().fit_transform
X = StandardScaler().fit_transform(X)



'''
def forest_test(X, Y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,
                                                        test_size = 0.30,
                                                        random_state = 101)
    start = time.process_time()
    trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train,Y_Train)
    #print(time.process_time() - start)
    predictionforest = trainedforest.predict(X_Test)
    #print(confusion_matrix(Y_Test,predictionforest))
    #print(classification_report(Y_Test,predictionforest))


forest_test(X, Y)
'''
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
PCA_df = pd.concat([PCA_df, df['Churn']], axis = 1)
PCA_df['Churn'] = LabelEncoder().fit_transform(PCA_df['Churn'])
PCA_df.head() 



#ID3
#Scaled the data
x = df.drop('Churn', axis='columns')
y = df['Churn']


x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=321, test_size=0.2)


modell = DecisionTreeClassifier(criterion="entropy", max_depth=6)
modell.fit(x_train, y_train)

predictions = modell.predict(x_test)
#print('classification_report : ', classification_report(y_test, predictions))
accuracy = accuracy_score(y_test, predictions)
#print('accuracy : ', accuracy)


#print('your prediction is : ', predictions ,'accuracy is : ', accuracy)
ID3Train= modell.score(x_train,y_train);  
ID3Test=accuracy




#SVM
y = df['Churn']
x = df.drop('Churn' , axis=1) 
x = StandardScaler().fit_transform(x)
x_train, x_test ,y_train, y_test = train_test_split(x, y , test_size=0.2, random_state = 0)
m = SVC(kernel = 'linear' , random_state=0 ,C=1)
m.fit(x_train , y_train) 
y_pred = m.predict(x_test)
SVMTrain = m.score(x_train, y_train);
SVMTest=metrics.accuracy_score(y_test, y_pred);




#LogisticRegression
from sklearn.metrics import accuracy_score
#Scaled the data
x = df.drop('Churn' , axis=1) 
y = df['Churn']
x = StandardScaler().fit_transform(x)

#Split the dara into 80% training and 20% testing
x_train, x_test ,y_train, y_test = train_test_split(x, y , test_size=0.2, random_state = 42)

#create the model 
model = LogisticRegression()
#train the model
model.fit(x_train ,y_train)
#create the prediction
predictions = model.predict(x_test)

#print the predictions
#print(predictions)

    #print accuracy
LogisticTest =accuracy_score(y_test,predictions);
LogisticTrain = model.score(x_train, y_train);       
   
    

            
        
#knn
model2 = KNeighborsClassifier(n_neighbors=50)
model2.fit(x_train, y_train)
pred = model2.predict(x_test)
KnnTest = accuracy_score(y_test, pred)
knnTrain = model2.score(x_train, y_train)
                
        
def testLOGISTIC ():

    print ("accuracy of LOGISTIC(test) :" ,LogisticTest)
    
    
def TRAINLOGISTIC ():
    
    print ("accuracy of LOGISTIC(train) :" , LogisticTrain)
        
    
    
def trainSVM () :   
   
    print ("accuracy of SVM(train) :" , SVMTrain)
    
    
    
def testSVM () :  
    
    print ("accuracy of SVM(test) :" , SVMTest)
 
    
    
def trainID3 () :   
  print ("accuracy of id3(train) :" , ID3Train)
   
  
  
def testID3 ()   :  
  print ("accuracy of id3(test) :" , ID3Test)  



def testKnn ()   :  
  print ("accuracy of knn(test) :" , KnnTest)
  
  
  
def trainKnn ()   :  
  print ("accuracy of knn(train) :" , knnTrain)

        
    
#gui    
root=Tk()
root.title('Service Cancellation Predictor')
root.geometry('800x400')
root.maxsize(1200,800)
root.minsize(600,200)

#root.configure(background='violet')

b1=Button(root,text="test logistic",width="10",height="1",command=testLOGISTIC)
b1.grid(column=1,row=3)

b2=Button(root,text="test svm",width="10",height="1",command=testSVM)
b2.grid(column=2,row=3)

c1=Button(root,text="test id3",width="10",height="1" , command=testID3)
c1.grid(column=3,row=3)

c2=Button(root,text="train logistic",width="10",height="1", command=TRAINLOGISTIC )
c2.grid(column=1,row=2)

c3=Button(root,text="train svm",width="10",height="1", command=trainSVM)
c3.grid(column=2,row=2)

c4=Button(root,text="train id3",width="10",height="1", command=trainID3)
c4.grid(column=3,row=2)

c5=Button(root,text="train knn",width="10",height="1", command=trainKnn)
c5.grid(column=4,row=2)

c6=Button(root,text="test knn",width="10",height="1", command=testKnn)
c6.grid(column=4,row=3)


t1=tk.Label(text='mehtodology')
t1.grid(column=1,row=1)        

t1=tk.Label(text='custumer data')
t1.grid(column=1,row=4)
   
t1=tk.Label(text='custumer id')
t1.grid(column=1,row=5)
e1=Entry(root)
e1.grid(column=2,row=5)

t2=tk.Label(text='gender')
t2.grid(column=3,row=5)

e2=Entry(root)
e2.grid(column=4,row=5)

t1=tk.Label(text='senior citizen')
t1.grid(column=5,row=5)

e3=Entry(root)
e3.grid(column=6,row=5)


l1=tk.Label(root, text='Partner')
l1.grid(row=6,column=1)
l2=tk.Label(root, text='Dependents')
l2.grid(row=6,column=3)
l3=tk.Label(root, text='tenure')
l3.grid(row=6,column=5)

e4 = Entry(root)
e4.grid(row=6, column=2)
e5 = Entry(root)
e5.grid(row=6, column=4)
e6 = Entry(root)
e6.grid(row=6, column=6)

l4=tk.Label(root, text='PhoneService')
l4.grid(row=7,column=1)
l5=tk.Label(root, text='MultipleLines')
l5.grid(row=7,column=3)
l6=Label(root, text='InternetService')

l6.grid(row=7,column=5)
e7 = Entry(root)
e7.grid(row=7, column=2)
e8 = Entry(root)
e8.grid(row=7, column=4)
e9 = Entry(root)
e9.grid(row=7, column=6)

l7=tk.Label(root, text='OnlineSecurity')
l7.grid(row=8,column=1)
l8=tk.Label(root, text='OnlineBackup')
l8.grid(row=8,column=3)
l9=tk.Label(root, text='DeviceProtection')
l9.grid(row=8,column=5)

e10 = Entry(root)
e10.grid(row=8, column=2)
e11= Entry(root)
e11.grid(row=8, column=4)
e12 = Entry(root)
e12.grid(row=8, column=6)

l10=tk.Label(root, text='TechSupport')
l10.grid(row=9,column=1)
l11=tk.Label(root, text='StreamingTV')
l11.grid(row=9,column=3)
l12=tk.Label(root, text='StreamingMovies')
l12.grid(row=9,column=5)

e13 = Entry(root)
e13.grid(row=9, column=2)
e14= Entry(root)
e14.grid(row=9, column=4)
e15 = Entry(root)
e15.grid(row=9, column=6)

l13=tk.Label(root, text='Contract')
l13.grid(row=10,column=1)
l14=tk.Label(root, text='PaperlessBilling')
l14.grid(row=10,column=3)
l15=tk.Label(root, text='PaymentMethod')
l15.grid(row=10,column=5)

e16 = Entry(root)
e16.grid(row=10, column=2)
e17= Entry(root)
e17.grid(row=10, column=4)
e18 = Entry(root)
e18.grid(row=10, column=6)

l16=tk.Label(root, text='MonthlyCharges')
l16.grid(row=11,column=1)
l17=tk.Label(root, text='TotalCharges')
l17.grid(row=11,column=3)

e19 = Entry(root)
e19.grid(row=11, column=2)
e20= Entry(root)
e20.grid(row=11, column=4)



def logPrediction():
    #customerId
    var1=e1.get()
    
    #['gender'].replace(['Female','Male'],[1,0],inplace=True)
   # var2=e2.get()
    if e2.get() == "female" :
      var2=1
    elif e2.get()   == "male" :
        var2=0
   # var2=int(var2)  
   
    else :
       print ("please enter valid value")
       
       
    #SeniorCitizen
    var3=e3.get() 
    
    #['Partner'].replace(['Yes','No'],[1,0], inplace=True)
   # var4=e4.get()
    if e4.get() == "yes":
       var4=1
    elif e4.get() == "no"  :
      var4=0
    #var4=int(var4)
   
    else :
       print ("please enter valid value")
       
           
    
    
    #['Dependents'].replace(['Yes','No'],[1,0],inplace=True)
   # var5=e5.get()
    if e5.get()== "yes":
       var5=1
    elif e5.get()  == "no"  :
       var5=0
   # var5=int(var5)
   
    else :
       print ("please enter valid value")
       
           
    #tenure
    var6=e6.get()
    
     # ['PhoneService'].replace(['Yes','No'],[1,0],inplace=True)
  #  var7=e7.get()
    if e7.get()== "yes":
         var7=1
    elif e7.get()== "no" :
         var7=0
   # var7=int(var7)     
   
    else :
       print ("please enter valid value")
       
             
      #['MultipleLines'].replace(['No phone service','No','Yes'],[2,0,1],inplace=True)
    #var8=e8.get()
    if e8.get()== "no phone service":
          var8=2
    elif e8.get()== "yes" :
          var8=1
    elif e8.get()== "no" :
          var8=0
   # var8=int(var8)      
   
    else :
       print ("please enter valid value")
       
                  
      
       #['InternetService'].replace(['DSL','Fiber optic','No'],[2,1,0],inplace=True)
   # var9=e9.get()
    if e9.get()=="dsl":
         var9=2
    elif e9.get() == "fiber optic" :
         var9=1
    elif e9.get()== "no" :
         var9=0
  #  var9=int(var9)     
   
    else :
       print ("please enter valid value")
       
              
       
    # ['OnlineSecurity'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
    #var10=e10.get() 
    if e10.get() == "no internet service":
          var10=2
    elif e10.get() == "yes" :
          var10=1
    elif e10.get() == "no" :
          var10=0
   # var10=int(var10)   
   
    else :
       print ("please enter valid value")
       
              
     #['OnlineBackup'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
    #var11= e11.get()
    if e11.get() == "no internet service":
         var11=2
    elif e11.get() == "yes" :
         var11=1
    elif e11.get() == "no" :
          var11=0
  #  var11=int(var11)
   
    else :
       print ("please enter valid value")
       
           
       #  ['DeviceProtection'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True) 
   # var12=e12.get()
    if e12.get()== "no internet service":
           var12=2
    elif e12.get() == "yes" :
           var12=1
    elif e12.get() == "no" :
          var12=0
   # var12=int(var12)      
   
    else :
       print ("please enter valid value")
       
                   
            
      # ['TechSupport'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
   # var13=e13.get()
    if e13.get() == "no internet service":
         var13=2
    elif e13.get() == "yes" :
          var13=1
    elif e13.get() == "no" :
          var13=0
    #var13 = int(var13 )    
   
    else :
       print ("please enter valid value")
       
              
       #['StreamingTV'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
  #  var14=e14.get()
    if e14.get() == "no internet service":
          var14=2
    elif e14.get()== "yes" :
          var14=1
    elif e14.get() == "no" :
          var14=0
   # var14=int(var14)  
   
    else :
       print ("please enter valid value")
       
            
       #['StreamingMovies'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
    #var15=e15.get()
    if e15.get()== "no internet service":
           var15=2
    elif e15.get()== "yes" :
          var15=1
    elif e15.get()== "no" :
          var15=0
   # var15 = int(var15) 
   
    else :
       print ("please enter valid value")
       
              
       #['Contract'].replace(['Month-to-month','One year','Two year'],[0,1,2],inplace=True)
   # var16 = e16.get()
    if e16.get()== "two year":
          var16 =2
    elif e16.get()== "one year" :
          var16 =1
    elif e16.get()== "month-to-month" :
           var16 =0
    #var16 =int(var16)  
   
    else :
       print ("please enter valid value")
       
              
       
       #['PaperlessBilling'].replace(['Yes','No'],[1,0],inplace=True)
   # var17 = e17.get()
    if e17.get()== "yes":
          var17 =1
    elif e17.get()== "no" :
          var17 =0
    #var17=int(var17)      
   
    else :
       print ("please enter valid value")
       
                  
       
        #['PaymentMethod'].replace(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'],[0,1,2,3],inplace=True)
   # var18= e18.get()
    if e18.get()== "electronic check":
         var18 =0
    elif e18.get()== "mailed check" :
         var18 =1
    elif e18.get()== "bank transfer (automatic)" :
          var18 =2
    elif e18.get()== "credit card (automatic)" :
         var18 =3
   # var18 =int(var18 )  
   
    else :
       print ("please enter valid value")
       
            
    var19 = e19.get()
    
    var20 = e20.get()
    aa=[[var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16,var17,var18,var19,var20]]
    b = np.array(aa,dtype=float)
    df=model.predict(b)
    if df == [1] :
        print ("the prediction value of LOGISITC :yes")
        
    elif df == [0] :
        print ("the prediction value of LOGISTIC :no") 

   
  
   
def svmPrediction():
    #customerId
    var1=e1.get()
    
    #['gender'].replace(['Female','Male'],[1,0],inplace=True)
   # var2=e2.get()
    if e2.get() == "female" :
      var2=1
    elif e2.get()   == "male" :
        var2=0
   # var2=int(var2)    
   
    else :
       print ("please enter valid value")
       
               
    #SeniorCitizen
    var3=e3.get() 
    
    #['Partner'].replace(['Yes','No'],[1,0], inplace=True)
   # var4=e4.get()
    if e4.get() == "yes":
       var4=1
    elif e4.get() == "no"  :
      var4=0
    #var4=int(var4)
   
    else :
       print ("please enter valid value")
       
           
    
    
    #['Dependents'].replace(['Yes','No'],[1,0],inplace=True)
   # var5=e5.get()
    if e5.get()== "yes":
       var5=1
    elif e5.get()  == "no"  :
       var5=0
   # var5=int(var5)
   
    else :
       print ("please enter valid value")
       
           
    #tenure
    var6=e6.get()
    
     # ['PhoneService'].replace(['Yes','No'],[1,0],inplace=True)
  #  var7=e7.get()
    if e7.get()== "yes":
         var7=1
    elif e7.get()== "no" :
         var7=0
   # var7=int(var7)     
   
    else :
       print ("please enter valid value")
       
             
      #['MultipleLines'].replace(['No phone service','No','Yes'],[2,0,1],inplace=True)
    #var8=e8.get()
    if e8.get()== "no phone service":
          var8=2
    elif e8.get()== "yes" :
          var8=1
    elif e8.get()== "no" :
          var8=0
   # var8=int(var8)      
   
    else :
       print ("please enter valid value")
       
                  
      
       #['InternetService'].replace(['DSL','Fiber optic','No'],[2,1,0],inplace=True)
   # var9=e9.get()
    if e9.get()=="dsl":
         var9=2
    elif e9.get() == "fiber optic" :
         var9=1
    elif e9.get()== "no" :
         var9=0
  #  var9=int(var9)     
   
    else :
       print ("please enter valid value")
       
              
       
    # ['OnlineSecurity'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
    #var10=e10.get() 
    if e10.get() == "no internet service":
          var10=2
    elif e10.get() == "yes" :
          var10=1
    elif e10.get() == "no" :
          var10=0
   # var10=int(var10)   
   
    else :
       print ("please enter valid value")
       
              
     #['OnlineBackup'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
    #var11= e11.get()
    if e11.get() == "no internet service":
         var11=2
    elif e11.get() == "yes" :
         var11=1
    elif e11.get() == "no" :
          var11=0
  #  var11=int(var11)
   
    else :
       print ("please enter valid value")
       
           
       #  ['DeviceProtection'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True) 
   # var12=e12.get()
    if e12.get()== "no internet service":
           var12=2
    elif e12.get() == "yes" :
           var12=1
    elif e12.get() == "no" :
          var12=0
   # var12=int(var12)      
    
    else :
       print ("please enter valid value")
       
                  
            
      # ['TechSupport'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
   # var13=e13.get()
    if e13.get() == "no internet service":
         var13=2
    elif e13.get() == "yes" :
          var13=1
    elif e13.get() == "no" :
          var13=0
    #var13 = int(var13 )    
    
    else :
       print ("please enter valid value")
       
             
       #['StreamingTV'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
  #  var14=e14.get()
    if e14.get() == "no internet service":
          var14=2
    elif e14.get()== "yes" :
          var14=1
    elif e14.get() == "no" :
          var14=0
   # var14=int(var14)  
    
    else :
       print ("please enter valid value")
       
           
       #['StreamingMovies'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
    #var15=e15.get()
    if e15.get()== "no internet service":
           var15=2
    elif e15.get()== "yes" :
          var15=1
    elif e15.get()== "no" :
          var15=0
   # var15 = int(var15) 
    
    else :
       print ("please enter valid value")
       
             
       #['Contract'].replace(['Month-to-month','One year','Two year'],[0,1,2],inplace=True)
   # var16 = e16.get()
    if e16.get()== "two year":
          var16 =2
    elif e16.get()== "one year" :
          var16 =1
    elif e16.get()== "month-to-month" :
           var16 =0
    #var16 =int(var16)  
   
    else :
       print ("please enter valid value")
       
              
       
       #['PaperlessBilling'].replace(['Yes','No'],[1,0],inplace=True)
   # var17 = e17.get()
    if e17.get()== "yes":
          var17 =1
    elif e17.get()== "no" :
          var17 =0
    #var17=int(var17)      
   
    else :
       print ("please enter valid value")
       
                  
       
        #['PaymentMethod'].replace(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'],[0,1,2,3],inplace=True)
   # var18= e18.get()
    if e18.get()== "electronic check":
         var18 =0
    elif e18.get()== "mailed check" :
         var18 =1
    elif e18.get()== "bank transfer (automatic)" :
          var18 =2
    elif e18.get()== "credit card (automatic)" :
         var18 =3
   # var18 =int(var18 )  
   
    else :
       print ("please enter valid value")
       
            
    var19 = e19.get()
    
    var20 = e20.get()
   
    df=m.predict([[var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16,var17,var18,var19,var20]])     
    if df == [1] :
        print ("the prediction value of SVM : yes")
        
    elif df == [0] :
        print ("the prediction value of SVM :no") 
      
        
      
def id3Prediction():
    #customerId
    var1=e1.get()
    
    #['gender'].replace(['Female','Male'],[1,0],inplace=True)
   # var2=e2.get()
    if e2.get() == "female" :
      var2=1
    elif e2.get()   == "male" :
        var2=0
   # var2=int(var2)    
   
    else :
       print ("please enter valid value")
       
               
    #SeniorCitizen
    var3=e3.get() 
    
    #['Partner'].replace(['Yes','No'],[1,0], inplace=True)
   # var4=e4.get()
    if e4.get() == "yes":
       var4=1
    elif e4.get() == "no"  :
      var4=0
    #var4=int(var4)
   
    else :
       print ("please enter valid value")
       
           
    
    
    #['Dependents'].replace(['Yes','No'],[1,0],inplace=True)
   # var5=e5.get()
    if e5.get()== "yes":
       var5=1
    elif e5.get()  == "no"  :
       var5=0
   # var5=int(var5)
   
    else :
       print ("please enter valid value")
       
           
    #tenure
    var6=e6.get()
    
     # ['PhoneService'].replace(['Yes','No'],[1,0],inplace=True)
  #  var7=e7.get()
    if e7.get()== "yes":
         var7=1
    elif e7.get()== "no" :
         var7=0
   # var7=int(var7)     
   
    else :
       print ("please enter valid value")
       
             
      #['MultipleLines'].replace(['No phone service','No','Yes'],[2,0,1],inplace=True)
    #var8=e8.get()
    if e8.get()== "no phone service":
          var8=2
    elif e8.get()== "yes" :
          var8=1
    elif e8.get()== "no" :
          var8=0
   # var8=int(var8)      
   
    else :
       print ("please enter valid value")
       
                  
      
       #['InternetService'].replace(['DSL','Fiber optic','No'],[2,1,0],inplace=True)
   # var9=e9.get()
    if e9.get()=="dsl":
         var9=2
    elif e9.get() == "fiber optic" :
         var9=1
    elif e9.get()== "no" :
         var9=0
  #  var9=int(var9)     
   
    else :
       print ("please enter valid value")
       
              
       
    # ['OnlineSecurity'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
    #var10=e10.get() 
    if e10.get() == "no internet service":
          var10=2
    elif e10.get() == "yes" :
          var10=1
    elif e10.get() == "no" :
          var10=0
   # var10=int(var10)   
   
    else :
       print ("please enter valid value")
       
              
     #['OnlineBackup'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
    #var11= e11.get()
    if e11.get() == "no internet service":
         var11=2
    elif e11.get() == "yes" :
         var11=1
    elif e11.get() == "no" :
          var11=0
  #  var11=int(var11)
   
    else :
       print ("please enter valid value")
       
           
       #  ['DeviceProtection'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True) 
   # var12=e12.get()
    if e12.get()== "no internet service":
           var12=2
    elif e12.get() == "yes" :
           var12=1
    elif e12.get() == "no" :
          var12=0
   # var12=int(var12)      
   
    else :
       print ("please enter valid value")
       
                   
            
      # ['TechSupport'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
   # var13=e13.get()
    if e13.get() == "no internet service":
         var13=2
    elif e13.get() == "yes" :
          var13=1
    elif e13.get() == "no" :
          var13=0
    #var13 = int(var13 )    
   
    else :
       print ("please enter valid value")
       
              
       #['StreamingTV'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
  #  var14=e14.get()
    if e14.get() == "no internet service":
          var14=2
    elif e14.get()== "yes" :
          var14=1
    elif e14.get() == "no" :
          var14=0
   # var14=int(var14)  
    
    else :
       print ("please enter valid value")
       
           
       #['StreamingMovies'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
    #var15=e15.get()
    if e15.get()== "no internet service":
           var15=2
    elif e15.get()== "yes" :
          var15=1
    elif e15.get()== "no" :
          var15=0
   # var15 = int(var15) 
    
    else :
       print ("please enter valid value")
       
             
       #['Contract'].replace(['Month-to-month','One year','Two year'],[0,1,2],inplace=True)
   # var16 = e16.get()
    if e16.get()== "two year":
          var16 =2
    elif e16.get()== "one year" :
          var16 =1
    elif e16.get()== "month-to-month" :
           var16 =0
    #var16 =int(var16)  
   
    else :
       print ("please enter valid value")
       
              
       
       #['PaperlessBilling'].replace(['Yes','No'],[1,0],inplace=True)
   # var17 = e17.get()
    if e17.get()== "yes":
          var17 =1
    elif e17.get()== "no" :
          var17 =0
    #var17=int(var17)      
   
    else :
       print ("please enter valid value")
       
                  
       
        #['PaymentMethod'].replace(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'],[0,1,2,3],inplace=True)
   # var18= e18.get()
    if e18.get()== "electronic check":
         var18 =0
    elif e18.get()== "mailed check" :
         var18 =1
    elif e18.get()== "bank transfer (automatic)" :
          var18 =2
    elif e18.get()== "credit card (automatic)" :
         var18 =3
   # var18 =int(var18 )  
   
    else :
       print ("please enter valid value")
       
            
    var19 = e19.get()
    
    var20 = e20.get()
    df=modell.predict([[var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16,var17,var18,var19,var20]])     
    if df == [1] :
        print ("the prediction value of ID3 :yes")
        
    elif df == [0] :
        print ("the prediction value of ID3 : no")    

  

  
def knnPrediction():
    #customerId
    var1=e1.get()
    
    #['gender'].replace(['Female','Male'],[1,0],inplace=True)
   # var2=e2.get()
    if e2.get() == "female" :
      var2=1
    elif e2.get()   == "male" :
        var2=0
   # var2=int(var2)    
   
    else :
       print ("please enter valid value")
       
               
    #SeniorCitizen
    var3=e3.get() 
    
    #['Partner'].replace(['Yes','No'],[1,0], inplace=True)
   # var4=e4.get()
    if e4.get() == "yes":
       var4=1
    elif e4.get() == "no"  :
      var4=0
    #var4=int(var4)
    
    
    else :
       print ("please enter valid value")
       
          
    
    #['Dependents'].replace(['Yes','No'],[1,0],inplace=True)
   # var5=e5.get()
    if e5.get()== "yes":
       var5=1
    elif e5.get()  == "no"  :
       var5=0
   # var5=int(var5)
     
    else :
       print ("please enter valid value")
       
         
    #tenure
    var6=e6.get()
    
     # ['PhoneService'].replace(['Yes','No'],[1,0],inplace=True)
  #  var7=e7.get()
    if e7.get()== "yes":
         var7=1
    elif e7.get()== "no" :
         var7=0
   # var7=int(var7)     
   
    else :
       print ("please enter valid value")
       
             
      #['MultipleLines'].replace(['No phone service','No','Yes'],[2,0,1],inplace=True)
    #var8=e8.get()
    if e8.get()== "no phone service":
          var8=2
    elif e8.get()== "yes" :
          var8=1
    elif e8.get()== "no" :
          var8=0
   # var8=int(var8)      
    
    else :
       print ("please enter valid value")
       
                 
      
       #['InternetService'].replace(['DSL','Fiber optic','No'],[2,1,0],inplace=True)
   # var9=e9.get()
    if e9.get()=="dsl":
         var9=2
    elif e9.get() == "fiber optic" :
         var9=1
    elif e9.get()== "no" :
         var9=0
  #  var9=int(var9)     
       
   
    else :
       print ("please enter valid value")
       
              
    # ['OnlineSecurity'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
    #var10=e10.get() 
    if e10.get() == "no internet service":
          var10=2
    elif e10.get() == "yes" :
          var10=1
    elif e10.get() == "no" :
          var10=0
   # var10=int(var10)   
   
    else :
       print ("please enter valid value")
       
              
     #['OnlineBackup'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
    #var11= e11.get()
    if e11.get() == "no internet service":
         var11=2
    elif e11.get() == "yes" :
         var11=1
    elif e11.get() == "no" :
          var11=0
  #  var11=int(var11)
    
    else :
       print ("please enter valid value")
       
          
       #  ['DeviceProtection'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True) 
   # var12=e12.get()
    if e12.get()== "no internet service":
           var12=2
    elif e12.get() == "yes" :
           var12=1
    elif e12.get() == "no" :
          var12=0
   # var12=int(var12)      
   
    else :
       print ("please enter valid value")
       
                   
            
      # ['TechSupport'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
   # var13=e13.get()
    if e13.get() == "no internet service":
         var13=2
    elif e13.get() == "yes" :
          var13=1
    elif e13.get() == "no" :
          var13=0
    #var13 = int(var13 )    
   
    else :
       print ("please enter valid value")
       
              
       #['StreamingTV'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
  #  var14=e14.get()
    if e14.get() == "no internet service":
          var14=2
    elif e14.get()== "yes" :
          var14=1
    elif e14.get() == "no" :
          var14=0
   # var14=int(var14)  
    
    else :
       print ("please enter valid value")
       
           
       #['StreamingMovies'].replace(['Yes','No','No internet service'],[1,0,2],inplace=True)
    #var15=e15.get()
    if e15.get()== "no internet service":
           var15=2
    elif e15.get()== "yes" :
          var15=1
    elif e15.get()== "no" :
          var15=0
   # var15 = int(var15) 
    
    else :
       print ("please enter valid value")
       
             
       #['Contract'].replace(['Month-to-month','One year','Two year'],[0,1,2],inplace=True)
   # var16 = e16.get()
    if e16.get()== "two year":
          var16 =2
    elif e16.get()== "one year" :
          var16 =1
    elif e16.get()== "month-to-month" :
           var16 =0
    #var16 =int(var16)  
    
    else :
       print ("please enter valid value")
       
             
       
       #['PaperlessBilling'].replace(['Yes','No'],[1,0],inplace=True)
   # var17 = e17.get()
    if e17.get()== "yes":
          var17 =1
    elif e17.get()== "no" :
          var17 =0
    #var17=int(var17)      
           
   
    else :
       print ("please enter valid value")
       
              
        #['PaymentMethod'].replace(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'],[0,1,2,3],inplace=True)
   # var18= e18.get()
    if e18.get()== "electronic check":
         var18 =0
    elif e18.get()== "mailed check" :
         var18 =1
    elif e18.get()== "bank transfer (automatic)" :
          var18 =2
    elif e18.get()== "credit card (automatic)" :
         var18 =3
   # var18 =int(var18 )  
    
    else :
       print ("please enter valid value")
       
           
    var19 = e19.get()
    
    var20 = e20.get()
    c=[[var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16,var17,var18,var19,var20]]
    bb = np.array(c,dtype=float)
    df=model2.predict(bb)  
    if df == [1] :
        print ("the prediction value of KNN :yes")
        
    elif df == [0] :
        print ("the prediction value of KNN : no")  



button3=Button(root,text='Predict using logistic',width=15,command=logPrediction)
button3.grid(column=3,row=13)

button4=Button(root,text='Predict using svm',width=15,command=svmPrediction)
button4.grid(column=4,row=13)

button5=Button(root,text='Predict using id3',width=15,command=id3Prediction)
button5.grid(column=5,row=13)

button5=Button(root,text='Predict using knn',width=15,command=knnPrediction)
button5.grid(column=6,row=13)


#print (df.dtypes)
#print (df.isnull().sum())


root.mainloop()


#data scaling 
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(copy=True,feature_range=(0,1))
df=scaler.fit_transform(df)