# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:45:55 2019

@author: hp
"""

from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
from flask import request
from datetime import datetime

from flask_cors import CORS, cross_origin
from wtforms import TextField,TextAreaField, SubmitField
from wtforms.validators import Required
 
import sys
import os
import datetime
import calendar

import pickle

import json
# Preparing the Classifier
#monthwise

cur_dir = os.path.dirname('__file__')

#total sales
#monthwise
regressor = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/modelmonths.pkl'), 'rb'))
model_colm = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model_columnsm.pkl'),'rb')) 

#daywise
clf = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/modeldays.pkl'), 'rb'))
model_cold = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model_columnsd.pkl'),'rb')) 

#weekwise
clfweek = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/modelweeks.pkl'), 'rb'))
model_colw = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model_columnsw.pkl'),'rb')) 



#customer visits
#daywise
clfvd = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/modelvdays.pkl'), 'rb'))
model_colvd = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model_columnsvd.pkl'),'rb')) 


#monthwise
clfvm = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/modelvmonths.pkl'), 'rb'))
model_colvm = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model_columnsvm.pkl'),'rb')) 

#weekwise
clfvw = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/modelvweeks.pkl'), 'rb'))
model_colvw = pickle.load(open(os.path.join(cur_dir,
			'pkl_objects/model_columnsvw.pkl'),'rb')) 


#expenses monthwise
regressor_exp = pickle.load(open(os.path.join(cur_dir,
                        'pkl_objects/modelexm.pkl'), 'rb'))
model_colexp = pickle.load(open(os.path.join(cur_dir,
                        'pkl_objects/model_columnsexm.pkl'),'rb'))


# Your API definition
app = Flask(__name__)

#for localhost
cors = CORS(app, resources={r"/": {"origins": "http://localhost:5000"}})

#daywise
@app.route('/day', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])

#for gcp cloud
#cors = CORS(app, resources={r"/": {"origins": "https://jts-board.appspot.com/"}})

#@app.route('/', methods=['POST'])
#@cross_origin(origin='*',headers=['Content- Type','Authorization'])

def predict():
    #print(request)
    if clf:
        try:
            print("LLLLLLLLLLLLLLL")
            jsond = request.json
            queryd = pd.get_dummies(pd.DataFrame(jsond))
            queryd = queryd.reindex(columns=model_cold, fill_value=0)
            
            predictiond = list(clf.predict(queryd).astype("int64"))
            print(predictiond)
            


            print(queryd)
            copy_queryd = queryd.copy(deep=True)
            copy_queryd.columns = ['day','month','year']

            date_time = pd.to_datetime(copy_queryd[['day', 'month', 'year']])
            #date_time.columns = ['timestamp']
 
            date_time_df=pd.DataFrame(date_time, columns=["timestamp"])
            date_time_df['day'] = date_time_df['timestamp'].dt.dayofweek
            dnum=pd.DataFrame(date_time_df['day'])
            dname = dnum['day'].apply(lambda x: calendar.day_abbr[x])
            #df['weekday'] = df['Timestamp'].apply(lambda x: x.weekday())
            
            print("PPPPPPPPPPPPPP")
            #print(copy_queryd)
            #print(date_time_df)
            #print(dname)
            print("JJJJJJJJJJJJJJ")

            #print(queryd['m'])
            #mname = queryd['d'].apply(lambda x: calendar.day_abbr[x])
            #print(mname)
            #print(queryxid1)

            #mname=pd.DataFrame(queryd['d'])
            #mname=pd.DataFrame(mname)
        
            predictiond=pd.DataFrame(predictiond, columns=["sales"])
            predictiond['sales'] = predictiond['sales'].apply(lambda x: x/1000)

            print(predictiond)
            con=pd.concat([dname,predictiond], axis=1)##################
            print(con)
            df=pd.DataFrame(con)###############


            df.set_index('day')['sales'].to_dict()
            print(df)

            count = df.shape[0]
            #print(count)
   
            #print("LLLLLLLLLLOOO") 
            #print(df)
            #print("KKKKKKKKKKKKK")
 

            days = df['day'].tolist()
            sales = df['sales'].tolist()

            #print("LLLLLLLLLLOOO")
            #print(months[1])
            #print(sales[1])
            #print("KKKKKKKKKKKKK")

            list_of_dicts = []
            D={}

            #add a key and setup for a sub-dictionary

            for i in range(count):
                D[i] = {}
                D[i]['x']=days[i]
                D[i]['value']=sales[i]
                list_of_dicts.append(D[i])


            #print(list_of_dicts)

            # convert into JSON:
            json_dict = json.dumps(list_of_dicts)

            # the result is a JSON string:
            print("LLLLLLLLLLOOO")
            print(json_dict)
            print("KKKKKKKKKKKKK")

    









            a = "cheese"        
            return json_dict    
            #return jsonify({'prediction': str(predictiond)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

#monthwise
@app.route('/month', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])


def predmonth():
    #print(request)
    if regressor:
        try:
            json1= request.json
            query1 = pd.get_dummies(pd.DataFrame(json1))
            query1 = query1.reindex(columns=model_colm, fill_value=0)
            #print(query1)
            copy_query1 = query1.copy(deep=True)
            copy_query1['month'] = copy_query1['month'].astype(str) + '月'
  

            print("JJJJJJJJJJJJJJJJJJ")
            print(query1)
            print("SSSSSSSSSSSSSSSSSS")
            
            
            #mname = query1['m'].apply(lambda x: calendar.month_abbr[x])
            mname = copy_query1['month']
            #print(mname)
            #print(query1)
            
            mname=pd.DataFrame(mname)
            print(mname)
             
            prediction1 = (regressor.predict(query1).astype('int64'))
            #print(prediction1)
            #print(prediction)
            prediction1=pd.DataFrame(prediction1, columns=["sales"])
            
            prediction1['sales'] = prediction1['sales'].apply(lambda x: x/1000)

            con=pd.concat([mname,prediction1], axis=1)
            print("IIIIIIIIIII")
            print(con)
            print("NNNNNNNNNN")
            df=pd.DataFrame(con)
            #df.set_index('m')['0'].to_dict()
            df.set_index('month')['sales'].to_dict()

            count = df.shape[0]
            #print(count)
   
            #print("LLLLLLLLLLOOO") 
            #print(df)
            #print("KKKKKKKKKKKKK")
 

            months = df['month'].tolist()
            sales = df['sales'].tolist()

            print("PPPPPPPPPP")
            print(months)
            print(sales)
            print("KKKKKKKKKKKKK")

            list_of_dicts = []
            D={}

            #add a key and setup for a sub-dictionary

            for i in range(count):
                D[i] = {}
                D[i]['x']=months[i]
                D[i]['value']=sales[i]
                list_of_dicts.append(D[i])

            print("BBBBBBBBBBBBBB")
            print(list_of_dicts)
            print("LLLLLLLLLLLLLL")

            # convert into JSON:
            json_dict = json.dumps(list_of_dicts,ensure_ascii=False)

            # the result is a JSON string:
            print("LLLLLLLLLLOOO")
            print(json_dict)
            print("KKKKKKKKKKKKK")


            #dict(zip(df.m, df.s))
            
            #from collections import defaultdict
            #mydict = defaultdict(list)
            #for k, v in zip(df.m.values,df.s.values):
            #    mydict[k].append(v)
            #    mydict[k]="x"
            #    mydict[v]="v"

                
            #print("LLLLLLLLLLOOO") 
            #print(mydict)
            #print(k)
            #print(v)
            #print("LLLLLLLLLL")
            
            #df.groupby('name')[['value1','value2']].apply(lambda g: g.values.tolist()).to_dict()

                
            
            #return jsonify({'prediction': str(prediction1)})
            #t = "cheese"
            #return(t)
            return(json_dict)
            
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

    #weekwise
@app.route('/week', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])


def predweek():
    #print(request)
    if clfweek:
        try:
            print("WWWWWWWWWWWWWWWWWWWWWWWWW")
            jsonw= request.json
            queryw = pd.get_dummies(pd.DataFrame(jsonw))
            queryw = queryw.reindex(columns=model_colw, fill_value=0)


            print(queryw)
            print("OOOOOOOOOOOOOOOOO")
            #print(query1['m'])
            #mname = queryw['w'].apply(lambda x: calendar.day_abbr[x])########
            #print(mname)
            #print(query1)
            mname=pd.DataFrame(queryw['week'])#########
            print(mname)
            print("WWWWWWWWWWWWWWWWWWWWWw")

            
            predictionw = (clfweek.predict(queryw).astype("int64"))
            
            #print("PPPPPPPPPPPPPP")
            #print(predictionw)
            #print("JJJJJJJJJJJJJJ")

            #print(prediction1)
            #print(prediction)
            predictionw=pd.DataFrame(predictionw, columns=["sales"])##########
            print(predictionw)
            predictionw['sales'] = predictionw['sales'].apply(lambda x: x/1000)

            con=pd.concat([mname,predictionw], axis=1)##################
            print(con)
            df=pd.DataFrame(con)################


            df.set_index('week')['sales'].to_dict()
            #print(df)



            count = df.shape[0]
            #print(count)

            #print("LLLLLLLLLLOOO") 
            #print(df)
            #print("KKKKKKKKKKKKK")


            weeks = df['week'].tolist()
            sales = df['sales'].tolist()

            #print("LLLLLLLLLLOOO")
            #print(months[1])
            #print(sales[1])
            #print("KKKKKKKKKKKKK")

            list_of_dicts = []
            D={}

            #add a key and setup for a sub-dictionary

            for i in range(count):
                D[i] = {}
                D[i]['x']=weeks[i]
                D[i]['value']=sales[i]
                list_of_dicts.append(D[i])


            #print(list_of_dicts)

            # convert into JSON:
            json_dict = json.dumps(list_of_dicts)

            # the result is a JSON string:
            print("LLLLLLLLLLOOO")
            print(json_dict)
            print("KKKKKKKKKKKKK")

            return json_dict
            #return jsonify({'prediction weekwise': str(predictionw).splitlines()})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
        
    
  #customer visits  
#daywise
@app.route('/vdays', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])

#for gcp cloud
#cors = CORS(app, resources={r"/": {"origins": "https://jts-board.appspot.com/"}})

#@app.route('/', methods=['POST'])
#@cross_origin(origin='*',headers=['Content- Type','Authorization'])

def predictvdays():
    #print(request)
    if clfvd:
        try:
           # print("LLLLLLLLLLLLLLL")
            jsond = request.json
            queryd = pd.get_dummies(pd.DataFrame(jsond))
            queryd = queryd.reindex(columns=model_colvd, fill_value=0)
            
            predictiond = list(clfvd.predict(queryd))
            #print(predictiond)
            


            #print(queryd)
            copy_queryd = queryd.copy(deep=True)
            copy_queryd.columns = ['day','month','year']

            date_time = pd.to_datetime(copy_queryd[['day', 'month', 'year']])
            #date_time.columns = ['timestamp']
 
            date_time_df=pd.DataFrame(date_time, columns=["timestamp"])
            date_time_df['day'] = date_time_df['timestamp'].dt.dayofweek
            dnum=pd.DataFrame(date_time_df['day'])
            dname = dnum['day'].apply(lambda x: calendar.day_abbr[x])
            #df['weekday'] = df['Timestamp'].apply(lambda x: x.weekday())
            
            #print("PPPPPPPPPPPPPP")
            #print(copy_queryd)
            #print(date_time_df)
            #print(dname)
            #print("JJJJJJJJJJJJJJ")

            #print(queryd['m'])
            #mname = queryd['d'].apply(lambda x: calendar.day_abbr[x])
            #print(mname)
            #print(queryxid1)

            #mname=pd.DataFrame(queryd['d'])
            #mname=pd.DataFrame(mname)
        
            predictiond=pd.DataFrame(predictiond, columns=["visits"])
            #predictiond[''] = predictiond['s'].apply(lambda x: x/1000)

            #print(predictiond)
            con=pd.concat([dname,predictiond], axis=1)##################
            #print(con)
            df=pd.DataFrame(con)###############


            df.set_index('day')['visits'].to_dict()
            #print(df)

            count = df.shape[0]
            #print(count)
   
            #print("LLLLLLLLLLOOO") 
            #print(df)
            #print("KKKKKKKKKKKKK")
 

            days = df['day'].tolist()
            sales = df['visits'].tolist()

            #print("LLLLLLLLLLOOO")
            #print(months[1])
            #print(sales[1])
            #print("KKKKKKKKKKKKK")

            list_of_dicts = []
            D={}

            #add a key and setup for a sub-dictionary

            for i in range(count):
                D[i] = {}
                D[i]['x']=days[i]
                D[i]['value']=sales[i]
                list_of_dicts.append(D[i])


            #print(list_of_dicts)

            # convert into JSON:
            json_dict = json.dumps(list_of_dicts)

            # the result is a JSON string:
            """
            print("LLLLLLLLLLOOO")
            print(json_dict)
            print("KKKKKKKKKKKKK")
            """

    









            a = "cheese"        
            return json_dict    
            #return jsonify({'prediction': str(predictiond)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
    
    
    
#monthwise
        
#monthwise
@app.route('/vmonths', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])


def predvmonths():
    #print(request)
    if clfvm:
        try:
            json1= request.json
            query1 = pd.get_dummies(pd.DataFrame(json1))
            query1 = query1.reindex(columns=model_colvm, fill_value=0)
            #print(query1)
            copy_query1 = query1.copy(deep=True)
            copy_query1['month'] = copy_query1['month'].astype(str) + '月'
            

            """print("JJJJJJJJJJJJJJJJJJ")
            print(query1)
            print("SSSSSSSSSSSSSSSSSS")
            """
            
            #mname = query1['m'].apply(lambda x: calendar.month_abbr[x])
            mname = copy_query1['month']
            #print(mname)
            #print(query1)
            
            mname=pd.DataFrame(mname)
            #print(mname)
             
            prediction1 = (clfvm.predict(query1))
            #print(prediction1)
            #print(prediction)
            prediction1=pd.DataFrame(prediction1, columns=["visits"])
            
            #prediction1['visits'] = prediction1['s'].apply(lambda x: x/1000)

            con=pd.concat([mname,prediction1], axis=1)
            """print("IIIIIIIIIII")
            print(con)
            print("NNNNNNNNNN")"""
            df=pd.DataFrame(con)
            #df.set_index('m')['0'].to_dict()
            df.set_index('month')['visits'].to_dict()

            count = df.shape[0]
            #print(count)
   
            #print("LLLLLLLLLLOOO") 
            #print(df)
            #print("KKKKKKKKKKKKK")
 

            months = df['month'].tolist()
            sales = df['visits'].tolist()

            """print("PPPPPPPPPP")
            print(months)
            print(sales)
            print("KKKKKKKKKKKKK")"""

            list_of_dicts = []
            D={}

            #add a key and setup for a sub-dictionary

            for i in range(count):
                D[i] = {}
                D[i]['x']=months[i]
                D[i]['value']=sales[i]
                list_of_dicts.append(D[i])

            """print("BBBBBBBBBBBBBB")
            print(list_of_dicts)
            print("LLLLLLLLLLLLLL")"""

            # convert into JSON:
            json_dict = json.dumps(list_of_dicts,ensure_ascii=False)

            # the result is a JSON string:
            """print("LLLLLLLLLLOOO")
            print(json_dict)
            print("KKKKKKKKKKKKK")"""


            #dict(zip(df.m, df.s))
            
            #from collections import defaultdict
            #mydict = defaultdict(list)
            #for k, v in zip(df.m.values,df.s.values):
            #    mydict[k].append(v)
            #    mydict[k]="x"
            #    mydict[v]="v"

                
            #print("LLLLLLLLLLOOO") 
            #print(mydict)
            #print(k)
            #print(v)
            #print("LLLLLLLLLL")
            
            #df.groupby('name')[['value1','value2']].apply(lambda g: g.values.tolist()).to_dict()

                
            
            #return jsonify({'prediction': str(prediction1)})
            #t = "cheese"
            #return(t)
            return(json_dict)
            
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')



    #weekwise
@app.route('/vweeks', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])


def predvweeks():
    #print(request)
    if clfvw:
        try:
            print("WWWWWWWWWWWWWWWWWWWWWWWWW")
            jsonw= request.json
            queryw = pd.get_dummies(pd.DataFrame(jsonw))
            queryw = queryw.reindex(columns=model_colvw, fill_value=0)


            print(queryw)
            print("OOOOOOOOOOOOOOOOO")
            #print(query1['m'])
            #mname = queryw['w'].apply(lambda x: calendar.day_abbr[x])########
            #print(mname)
            #print(query1)
            mname=pd.DataFrame(queryw['week'])#########
            print(mname)
            print("WWWWWWWWWWWWWWWWWWWWWw")

            
            predictionw = (clfvw.predict(queryw).astype("int64"))
            
            #print("PPPPPPPPPPPPPP")
            #print(predictionw)
            #print("JJJJJJJJJJJJJJ")

            #print(prediction1)
            #print(prediction)
            predictionw=pd.DataFrame(predictionw, columns=["visits"])##########
            print(predictionw)
            predictionw['visits'] = predictionw['visits'].apply(lambda x: x/1000)

            con=pd.concat([mname,predictionw], axis=1)##################
            print(con)
            df=pd.DataFrame(con)################


            df.set_index('week')['visits'].to_dict()
            #print(df)



            count = df.shape[0]
            #print(count)

            #print("LLLLLLLLLLOOO") 
            #print(df)
            #print("KKKKKKKKKKKKK")


            weeks = df['week'].tolist()
            sales = df['visits'].tolist()

            #print("LLLLLLLLLLOOO")
            #print(months[1])
            #print(sales[1])
            #print("KKKKKKKKKKKKK")

            list_of_dicts = []
            D={}

            #add a key and setup for a sub-dictionary

            for i in range(count):
                D[i] = {}
                D[i]['x']=weeks[i]
                D[i]['value']=sales[i]
                list_of_dicts.append(D[i])


            #print(list_of_dicts)

            # convert into JSON:
            json_dict = json.dumps(list_of_dicts)

            # the result is a JSON string:
            print("LLLLLLLLLLOOO")
            print(json_dict)
            print("KKKKKKKKKKKKK")

            return json_dict
            #return jsonify({'prediction weekwise': str(predictionw).splitlines()})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


#monthwise
@app.route('/monthexp', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])


def predmonthexp():
    #print(request)
    if regressor_exp:
        try:
            print("HELLLO")
            json1= request.json
            length = len(json1)
            
            
            
            query1 = pd.get_dummies(pd.DataFrame(json1))
            query1 = query1.reindex(columns=model_colexp, fill_value=0)
            mname = query1['month'].apply(lambda x: calendar.month_abbr[x])
            
            #print(query1)
            #copy_query1 = query1.copy(deep=True)
            #copy_query1['month'] = copy_query1['month'].astype(str) + '月'
            
            
            #print("JJJJJJJJJJJJJJJJJJ")
            #print(query1)
            #print("SSSSSSSSSSSSSSSSSS")
            
            
            #mname = query1['m'].apply(lambda x: calendar.month_abbr[x])
            #mname = copy_query1['month']
            #print(mname)
            #print(query1)
            
            mname=pd.DataFrame(mname)
            print(mname)
            
            prediction1 = (regressor_exp.predict(query1).astype('int64'))

            print("JJJJJJJJJJ")
            print(prediction1)
            print("KkKKKKKK")
            #print(prediction)
            prediction1=pd.DataFrame(prediction1, columns=["expenses"])
            
            #prediction1['expenses'] = prediction1['expenses'].apply(lambda x: x/1000)
            
            con=pd.concat([mname,prediction1], axis=1)
            print("IIIIIIIIIII")
            print(con)
            print("NNNNNNNNNN")
            df=pd.DataFrame(con)
            #df.set_index('m')['0'].to_dict()
            df.set_index('month')['expenses'].to_dict()
            
            count = df.shape[0]
            #print(count)
            
            #print("LLLLLLLLLLOOO")
            #print(df)
            #print("KKKKKKKKKKKKK")
            
            
            #months = df['month'].tolist()
            expenses = df['expenses'].tolist()
            
            
            ###################################################################
            
            
            
            query2 = pd.get_dummies(pd.DataFrame(json1))
            query2 = query2.reindex(columns=model_colm, fill_value=0)
            
            #print("JJJJJJJJJJJJJJJJJJ")
            #print(query2)
            #print("SSSSSSSSSSSSSSSSSS")
            
            
            mname = query2['month'].apply(lambda x: calendar.month_abbr[x])
            mname=pd.DataFrame(mname)
            
            
            prediction2 = (regressor.predict(query2).astype('int64'))
            prediction2=pd.DataFrame(prediction2, columns=["sales"])
            
            #prediction2['sales'] = prediction2['sales'].apply(lambda x: x/1000)
            
            con=pd.concat([mname,prediction2], axis=1)
            print("IIIIIIIIIII")
            print(con)
            print("NNNNNNNNNN")
            df=pd.DataFrame(con)
            #df.set_index('m')['0'].to_dict()
            df.set_index('month')['sales'].to_dict()
            
            count = df.shape[0]
            #print(count)
            months = df['month'].tolist() ########################
            sales = df['sales'].tolist() ##########################
            
            
            
            
            ################################################################
            
            join = list(zip(months, sales,expenses))  ###################
            join_json_string = json.dumps(join,ensure_ascii=False)
            
    
            
            '''
            print("PPPPPPPPPP")
            print(months)
            print(sales)
            print("KKKKKKKKKKKKK")
            
            list_of_dicts = []
            D={}
            
            #add a key and setup for a sub-dictionary
            
            for i in range(count):
                D[i] = {}
                D[i]['x']=months[i]
                D[i]['value']=sales[i]
                list_of_dicts.append(D[i])
        
            print("BBBBBBBBBBBBBB")
            print(list_of_dicts)
            print("LLLLLLLLLLLLLL")
            
            # convert into JSON:
            json_dict = json.dumps(list_of_dicts,ensure_ascii=False)
            
            # the result is a JSON string:
            print("LLLLLLLLLLOOO")
            print(json_dict)
            print("KKKKKKKKKKKKK")
    
            '''
            #t = "cheese"
            #return(t)
            #return(json_dict)
            return(join_json_string)
        
        except:
            
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

#monthwise
@app.route('/monthv', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])


def predmonv():
    #print(request)
    if clfvm:
        try:
            print("HELLLO")
            json1= request.json
            length = len(json1)
            
            
            
            query1 = pd.get_dummies(pd.DataFrame(json1))
            query1 = query1.reindex(columns=model_colvm, fill_value=0)
            mname = query1['month'].apply(lambda x: calendar.month_abbr[x])
            
            #print(query1)
            #copy_query1 = query1.copy(deep=True)
            #copy_query1['month'] = copy_query1['month'].astype(str) + '月'
            
            
            #print("JJJJJJJJJJJJJJJJJJ")
            #print(query1)
            #print("SSSSSSSSSSSSSSSSSS")
            
            
            #mname = query1['m'].apply(lambda x: calendar.month_abbr[x])
            #mname = copy_query1['month']
            #print(mname)
            #print(query1)
            
            mname=pd.DataFrame(mname)
            print(mname)
            
            prediction1 = (clfvm.predict(query1).astype('int64'))

            print("JJJJJJJJJJ")
            print(prediction1)
            print("KkKKKKKK")
            #print(prediction)
            prediction1=pd.DataFrame(prediction1, columns=["visits"])
            
            #prediction1['expenses'] = prediction1['expenses'].apply(lambda x: x/1000)
            
            con=pd.concat([mname,prediction1], axis=1)
            print("IIIIIIIIIII")
            print(con)
            print("NNNNNNNNNN")
            df=pd.DataFrame(con)
            #df.set_index('m')['0'].to_dict()
            df.set_index('month')['visits'].to_dict()
            
            count = df.shape[0]
            #print(count)
            
            #print("LLLLLLLLLLOOO")
            #print(df)
            #print("KKKKKKKKKKKKK")
            
            
            months = df['month'].tolist()
            visits = df['visits'].tolist()
            
            
            ###################################################################
            
            """
            
            query2 = pd.get_dummies(pd.DataFrame(json1))
            query2 = query2.reindex(columns=model_colm, fill_value=0)
            
            #print("JJJJJJJJJJJJJJJJJJ")
            #print(query2)
            #print("SSSSSSSSSSSSSSSSSS")
            
            
            mname = query2['month'].apply(lambda x: calendar.month_abbr[x])
            mname=pd.DataFrame(mname)
            
            
            prediction2 = (regressor.predict(query2).astype('int64'))
            prediction2=pd.DataFrame(prediction2, columns=["sales"])
            
            #prediction2['sales'] = prediction2['sales'].apply(lambda x: x/1000)
            
            con=pd.concat([mname,prediction2], axis=1)
            print("IIIIIIIIIII")
            print(con)
            print("NNNNNNNNNN")
            df=pd.DataFrame(con)
            #df.set_index('m')['0'].to_dict()
            df.set_index('month')['sales'].to_dict()
            
            count = df.shape[0]
            #print(count)
            months = df['month'].tolist() ########################
            sales = df['sales'].tolist() ##########################
            
            
            
            
            ################################################################
            """
            
            join = list(zip(months, visits))  ###################
            
            join_json_string = json.dumps(join,ensure_ascii=False)
            
    
            
            '''
            print("PPPPPPPPPP")
            print(months)
            print(sales)
            print("KKKKKKKKKKKKK")
            
            list_of_dicts = []
            D={}
            
            #add a key and setup for a sub-dictionary
            
            for i in range(count):
                D[i] = {}
                D[i]['x']=months[i]
                D[i]['value']=sales[i]
                list_of_dicts.append(D[i])
        
            print("BBBBBBBBBBBBBB")
            print(list_of_dicts)
            print("LLLLLLLLLLLLLL")
            
            # convert into JSON:
            json_dict = json.dumps(list_of_dicts,ensure_ascii=False)
            
            # the result is a JSON string:
            print("LLLLLLLLLLOOO")
            print(json_dict)
            print("KKKKKKKKKKKKK")
    
            '''
            #t = "cheese"
            #return(t)
            #return(json_dict)
            return(join_json_string)
        
        except:
            
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    app.run()     


