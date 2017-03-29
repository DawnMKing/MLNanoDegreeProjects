#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import pandas as pd
import pickle


enron_data = pickle.load(open("C:/Users/Dawn/Desktop/MachineLearningNano/ud120-projects-master/final_project/final_project_dataset.pkl", "r"))

#How many data points (people) are in the dataset?
print('number of people:', len(enron_data))

#For each person, how many features are available?
df = pd.DataFrame(enron_data)#enron_data to Dataframe
print('number of features:', len(df.index))#get number of features

#The “poi” feature records whether the person is a person of interest, 
#according to our definition. How many POIs are there in the E+F dataset?     
count = 0;
for person_name in df:
    if df[person_name]["poi"]==1:
        count = count + 1;
    else:
        count = count;    
print("number of POI's", count)         

#The “poi” feature records whether the person is a person of interest, 
#according to our definition. How many POIs are there in the E+F dataset?
data = open("C:/Users/Dawn/Desktop/MachineLearningNano/ud120-projects-master/final_project/poi_names.txt")
p_data=pd.read_table(data)
print("number of known POI's from web research:", len(p_data.index))#get number of features    

##Queries
#What is the total value of the stock belonging to James Prentice?
q1=enron_data["PRENTICE JAMES"]["total_stock_value"]
print('total stock value J. Prentice', q1)

#How many email messages do we have from Wesley Colwell to persons of interest?
q2=enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print('# emails From Colwell to POI:', q2)

#What’s the value of stock options exercised by Jeffrey K Skilling?
q3=enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
print('Skilling stock options:', q3)

#Of these three individuals (Lay, Skilling and Fastow), who took home the most
#money (largest value of “total_payments” feature)? How much money did that 
#person get?

q4=enron_data["SKILLING JEFFREY K"]["total_payments"]
q5=enron_data["LAY KENNETH L"]["total_payments"]
q6=enron_data["FASTOW ANDREW S"]["total_payments"]

print('Skilling:', q4)
print('Lay:', q5)
print('Fastow:', q6)

#How many folks in this dataset have a quantified salary? What about a known 
#email address?

count_salary = 0
e_count = 0
for key in enron_data.keys():
     if enron_data[key]['salary'] != 'NaN':
         count_salary = count_salary + 1
     if enron_data[key]['email_address'] != 'NaN':
         e_count += 1
print('# of emails:', e_count)
print('# of salaries:', count_salary)
#convert data_frame to array in functions that use numpy
from feature_format import featureFormat 
from feature_format import targetFeatureSplit 
feature_list = ["poi"] 
data_array = featureFormat(enron_data, feature_list, remove_NaN=True)
print(data_array)
label, features = targetFeatureSplit(data_array)
print(label)
#optional
#==============================================================================
#How many people in the E+F dataset (as it currently exists) have “NaN” for 
#their total payments? What percentage of people in the dataset as a whole is this?
tp_count = 0
for key in enron_data.keys():
     if enron_data[key]['total_payments'] == 'NaN':
         tp_count = tp_count + 1
print('# of people with no payment information:', tp_count)
percent = (float(tp_count)/len(enron_data))*100
print('% w/o payment ino:', percent)

#How many POIs in the E+F dataset have “NaN” for their total payments? What 
#percentage of POI’s as a whole is this?
poi_count = 0
for key in enron_data.keys():
     if enron_data[key]['total_payments'] == 'NaN'and enron_data[key]['poi'] == True:
         poi_count = poi_count + 1
print("#number POI's w/o payment info:", poi_count)
