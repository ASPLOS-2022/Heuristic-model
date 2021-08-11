import processing
from sklearn.metrics import accuracy_score
import joblib
import os

print('loading the model……')
columnNames2,rows2 = processing.load_data(r'data/test.csv')
columnNames3,rows3 = processing.load_data(r'data/data.csv')

columnNames2,x_test,y_test=processing.separate(columnNames2, rows2,'tag')
columnNames3,x_data,y_data=processing.separate(columnNames3, rows3,'maxval')

#load the model
model = joblib.load('model.txt')

# make predictions for test data
y_predict = model.predict(x_test)


print('\n',"predict :",y_predict)		#output the prediction of tag
print(" real tag:",y_test)		#output the real tag


predictions = [round(value) for value in y_predict]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


###evaluate the performance###
y_rate=y_test
num = len(y_rate)
for i in range(0,num):
   y_rate[i] = y_data[i]/x_data[i][int(y_predict[i])]
# print('\n',y_rate)
print('Time loss:',pow(y_rate.prod(),1/num))

os.system('pause')