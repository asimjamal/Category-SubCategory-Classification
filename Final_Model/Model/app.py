import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from flask import Flask, jsonify, request, make_response
import pickle
import json
import nltk
nltk.download('wordnet')

# load model
with open('SubCategoryModel.pkl','rb') as f:
	model = pickle.load(f)

#Load the vectorizer
with open('SubCategoryVectorizer.pkl','rb') as f:
	vectorizer = pickle.load(f)
    
#Load the encoder
with open('SubCategoryEncoder.pkl','rb') as f:
	encoder = pickle.load(f)

#Load the stopwords file
stop_words = pd.read_fwf('stop_words_model_text.txt')
stopWordSet = [x for x in stop_words['stop']]
stopWordSet = list(set(stopWordSet))

#Load the Json File of Model
with open("ModelJson","r") as f:
    dictionary = json.load(f)


#StopWord Function
def stopWords(text):
    text = text.split()
    text = [word for word in text if word not in stopWordSet]
    text = ' '.join(text)
    return(text)

#url-extractor
def url_extractor(text):
    urls = re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?:@#&=%.]+', text)
    text = text.split(' ')
    text = [words for words in text if words not in urls]
    text = ' '.join(text)
    return(text)

#Lemmatizer function
def lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split(' ')
    text = [lemmatizer.lemmatize(term) for term in text]
    text = ' '.join(text)
    return(text)

#Pre-processing models
def preprocessing(text):
    text = str(text).lower()
    text = re.sub("\n",' ',text)
    text = re.sub("\*"," ",text)
    text = re.sub("\[",' ',text)
    text = re.sub("\]",' ',text)
    text = re.sub("\(",' ',text)
    text = re.sub("\)",' ',text)
    urls = url_extractor(text)
    text = re.sub('[^a-zA-Z0-9]+', ' ', text)
    text = stopWords(text)
    text = lemmatizer(text)
    text = lemmatizer(text)
    text_model = re.sub('[0-9]','',text)
    return (text_model,text)
    
#BrandCheck Function for Nutrition Category especially
def BrandCheckNutrition(text):
    text = text.split(' ')
    if len(list(set(text) & set(dictionary['Infant nutrition'])))>0:
        pred = "Infant nutrition"
    elif len(list(set(text) & set(dictionary['Toddler nutrition'])))>0:
        pred = "Toddler nutrition"
    else:
        pred = "Infant nutrition"
    return(pred)
    
#Extracting age_nutriton_function    
def extracting_age(text):
    try:
        age = ['year','yr','yrs','saal','sal']
        for word in age:
            if word in text.lower():
                index = text.find(word)
                rest_string = text[index-5:index]
                number = re.findall(r'\d+', rest_string)
                if (int(number[0])>=2) & ( int(number[0])<4):
                    prediction = 'Toddler nutrition'
                elif (int(number[0])>=4):
                    prediction = 'Big Kid nutrition'
                else:
                    prediction = "Infant nutrition"
                break
            else:
                prediction = BrandCheckNutrition(text)
        return(prediction)
    except:
        prediction = BrandCheckNutrition(text)
        return(prediction)
        
#Campaign post function
def CampaignPost(text):
    text = text.split(" ")
    keys = list(dictionary.keys())
    keys = keys[:-2]
    if len(list(set(text) & set(keys)))>0:
        key = list(set(text) & set(keys))[0]
        prediction = dictionary[key]
    else:
        prediction = None
    return(prediction)

# app
app = Flask(__name__)

# routes

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    return make_response(jsonify({'status':'OK'}),200)

@app.route('/invocations', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)
    text = data["text"]
    keys = list(dictionary.keys())
    keys = keys[:-2]
    result_dict = {'Breastfeeding':0.0, 'Cold & Cough':0.0, 'Diaper':0.0, 'Gear':0.0, 'Immunity':0.0,'Oral Care':0.0, 'Other':0.0, 'Pregnancy':0.00, 'Recipes':0.00, 'Skin':0.00,'WomenHealthSkin':0.0, 'Infant nutrition':0.0,'Toddler nutrition':0.0,'Big Kid nutrition':0.0}
    # predictions
    text_model,text_check = preprocessing(text)
    if len(text_model)<1:
        result_dict["Other"]= 1.0
    
    elif len(list(set(text_model.split(' ')) & set(keys)))>0:
        
        key = list(set(text_model.split(' ')) & set(keys))[0]
        output = dictionary[key]
        result_dict[output] = 1.0
    
    else:
        text_vector = vectorizer.transform([text_model])
        prediction = model.predict(text_vector)
        if int(prediction[0]) == int(11):
            output = extracting_age(text)
            prediction = model.predict_proba(text_vector)[0]
            classes = encoder.classes_
            output_dict = {classes[i]:prediction[i] for i in range(0,len(classes))}
            for keys in output_dict.keys():
                result_dict[keys] = output_dict[keys]
            result_dict[output] = result_dict['nutrition']
            del result_dict['nutrition']
        else:
            result = encoder.inverse_transform(prediction)[0]
            prediction = model.predict_proba(text_vector)[0]
            classes = encoder.classes_
            output_dict = {classes[i]:prediction[i] for i in range(0,len(classes))}
            for keys in output_dict.keys():
                result_dict[keys] = output_dict[keys]
            result_dict['Infant nutrition'] = result_dict['nutrition']
            del result_dict['nutrition']
    return jsonify(result_dict)
                
app.run(port = 8080,threaded=True,debug=True,host='0.0.0.0')    