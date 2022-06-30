import datetime
import time
from datetime import datetime
import facebook_scraper
import pandas as pd
import hebrew_tokenizer as ht
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np


#ML
import sklearn
from sklearn import preprocessing, metrics, pipeline, model_selection, feature_extraction
from sklearn import naive_bayes, linear_model, svm, neural_network, neighbors, tree
from sklearn import decomposition, cluster

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#End - ML


def facebook_scrape():

    #Facebook scraper settings
    Group_ID = [
                "herzliya.apartments", #דירות ללא תיווך הרצליה רמת השרון רעננה
                "550870795608828", #דירות להשכרה בהרצליה בלבד
                "119965198095838", #דירות להשכרה בהרצליה והסביבה,
                "1663070923962851", #דירות להשכרה בהרצליה
                "herzlia",#דירות להשכרה ושותפים בהרצליה
                 "HomeInHerzliyarent", #דירות להשכרה בהרצליה
                "337487666800411", #דירות להשכרה ללא דמי תיווך -הרצליה,רעננה,כפר סבא,רמת השרון
                "secretapartmentssharon",
                "1909379842454834",
                "344940935888684",
                "dirotlehascarabeherzelia",
                ]
    option = {
                "progress": True,  "posts_per_page": 50, "allow_extra_requests": True,
               }
    cookie = "cookies.json"

    #pandas dataframe setting
    columns = ["username", "text","comments_full","comments","likes","image","images","time","post_url","images_lowquality","images_lowquality"]
    info_columns = ["street","area","price","No. of rooms","Porch", "Garden"]

    temp = []

    for id in Group_ID:
        print(id) #Must
        for post in facebook_scraper.get_posts(group=id, pages=10, cookies=cookie, options=option):


            if post["text"] != None:
                # fixed=tokenizer(post["text"])
                # post["comments_full"] = fixed
                temp.append(post)
                time.sleep(1)
    df = pd.DataFrame(temp)
    df.sort_values("time", inplace=True)

    df.to_csv(r"V2\facebook_scrape.csv", index=False, columns=columns, encoding="utf-8-sig")


def tokenizer(str):
    if type(str) is float:
        return "ERROR"
    list = []
    tokens = ht.tokenize(str)
    for grp, token, token_num, (start_index, end_index) in tokens:
        if grp == "HEBREW" or grp == "NUMBER":
            list.append(token)
    return list



def removeempty(str):
    if type(str) is not float:
        lines = str.split("\n")
        non_empty_lines = [line for line in lines if line.strip() != ""]
        string_without_empty_lines = ""

        for line in non_empty_lines:
            string_without_empty_lines += line
        
        return string_without_empty_lines



def text_anal():
    info_columns = ["street","area","price","Rooms","Porch", "Garden"]

    df = pd.read_csv(r"V2\facebook_scrape.csv",encoding="utf-8-sig")
    #df["text"] = df["text"].str.replace("\n", " ") IDK
    df["text"] = df["text"].apply(removeempty)

    #delete unreleavent posts from group managers etc.
    df.drop(df[df["username"] == "פנטהאוזים להשכרה בהרצליה"].index, inplace=True)
    df.drop(df[df["username"] == "Eyal Amir"].index,inplace=True)
    df.drop(df[df["username"] == "Ram Barak"].index, inplace=True)
    df.drop(df[df["username"] == "דירות ללא תיווך הרצליה, רמת השרון, רעננה"].index,inplace=True)
    #df.drop(df[df["time"] > "1/1/2022  10:12:35 AM"].index, inplace=True)
    df.drop_duplicates(subset=['text'], inplace=True)
    df.drop_duplicates(subset=['username'], inplace=True)
    df.dropna(subset="text", inplace=True)
    #df.insert(len(df.columns), column="Rooms", value="")

    for index, row in df.iterrows():
        if datetime.strptime(row["time"],'%Y-%m-%d %H:%M:%S') < datetime(2022,1,1,1,1,1):
            df.drop(index=index)

    #Main for loop
    #for index, row in df.iterrows():
        #df.at[index,"Rooms"] = room_calc(tokenizer(row["text"]))
        #TODO: add filtration to apartament for sale
        #TODO: add filtration to apartments that are from real easte
        #TODO: add apartment size
        #TODO: add price!!!
        #TODO: remove duplicates by username


    df.to_csv(r"V2\facebook_scrape_clean.csv",encoding="utf-8-sig")



def room_calc(liststr):
    if type(liststr) is float:
        return "ERROR"

    for i in range(len(liststr)):
        if liststr[i]=='חדר' or liststr[i]== 'חדרים':

            try:
                Rooms=int(liststr[i - 1])
                if Rooms<15:
                    print(abs(Rooms))
                    return abs(Rooms)
            except:
                return None



def MachineLearn():
    #load CSV
    X_df = pd.read_csv("facebook_scrape_clean.csv",encoding="utf-8-sig")
    V2df = pd.read_csv(r"V2\facebook_scrape_clean.csv",encoding="utf-8-sig")

    X_df.dropna(subset="text", inplace=True)
    V2df.dropna(subset="text", inplace=True)

    #add relevent columns
    #Y_df = pd.DataFrame(columns=[["street","Area","price","Rooms","Class"]], index=range(X_df.shape[0]))




    #Class Classification
    #S - Sale
    #R - Rent
    #Ro - Roomates
    #M - Mediator
    #N - Not relevent
    X_train, X_test, y_train, y_test = train_test_split(X_df['text'], X_df['Class'], test_size=0.2,random_state=42)


    text_clf = Pipeline([('vect', TfidfVectorizer(tokenizer=tokenizer, max_features=100000)),
                         ('CNN', MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15, 5), random_state=1))])

    text_clf.fit(X_train, y_train)

    #df_without["Class"].fillna("")




    prediction = text_clf.predict(X_test)
    #print(pred_train)
    F1 = f1_score(prediction, y_test, average='macro')

    print("acc is", accuracy_score(y_test,prediction))

    V2Pred= text_clf.predict(V2df["text"])

    V2df["Class"] = V2Pred.tolist()

    V2df.to_csv(r"V2\Prediction.csv",encoding="utf-8-sig")

if __name__ == '__main__':
    #facebook_scrape()
    text_anal()
    MachineLearn()