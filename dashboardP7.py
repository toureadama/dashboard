import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import lime
from lime import lime_tabular


def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(
        method='GET', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

path = "C:\\Users\\toure\\Desktop\\OpenClassrooms\\Projet 7\\donnees\\"
num_rows = 10000

data_client = pd.read_csv(path+"donnees_traites.csv", nrows= num_rows)
table_indic = pd.read_csv(path+"HomeCredit_columns_description.csv", encoding= 'unicode_escape')


data_client_0 = data_client[data_client["TARGET"]==0]
data_client_1 = data_client[data_client["TARGET"]==1]


nb_echant = min(50,data_client_1.shape[0])
data_echant = pd.concat([data_client_0[:nb_echant], data_client_1[:nb_echant]]).sort_values(by='SK_ID_CURR')
data_echant.reset_index(drop=True, inplace=True)

data_client = data_echant.drop(['TARGET'], axis=1)
references_test = data_echant.SK_ID_CURR

def main():
    MODELPREDICT_URI = 'http://127.0.0.1:5000/predict'
    
    st.title('Prêt à dépenser: Credit Scoring')
    
    st.sidebar.header("Informations sur le client")
    
    ref_client = st.sidebar.selectbox(
        'La référence du client',
        list(references_test))
    
    st.sidebar.write("Nombre de crédits dans l'échantillon", 
                    data_client.shape[1])
    
    st.sidebar.write("Le montant du crédit", 
                    int(data_client.loc[data_client['SK_ID_CURR']==int(ref_client)]['AMT_CREDIT']))
    
    st.sidebar.write("Son revenu annuel", 
                    int(data_client.loc[data_client['SK_ID_CURR']==int(ref_client)]['AMT_INCOME_TOTAL']))
    
    st.sidebar.write("Le revenu annuel moyen", 
                    int(data_client['AMT_INCOME_TOTAL'].mean()))
    
    st.sidebar.write("AGE", 
                    int(abs(data_client.loc[data_client['SK_ID_CURR']==int(ref_client)]['DAYS_BIRTH']/365)),' ans')
    
    #st.sidebar.write("Situation maritale", 
                    #data_client.loc[data_client['SK_ID_CURR']==int(ref_client)]['NAME_FAMILY_STATUS'])
    
    st.sidebar.write("Nombre d'enfants", 
                    int(data_client.loc[data_client['SK_ID_CURR']==int(ref_client)]['CNT_CHILDREN']))
    
    if int(data_client.loc[data_client['SK_ID_CURR']==int(ref_client)]['CODE_GENDER'])==1:
                       st.sidebar.write("GENRE: ", "F")
    else:
                       st.sidebar.write("GENRE: ", "M")
    
        
    
    predict_btn = st.button('Prédire')
    if predict_btn:
        data = [ref_client]
        pred = None
        pred = request_prediction(MODELPREDICT_URI, data)#[0] * 100000
        st.write(
            'La probabilité de défaut est de {:.2f}'.format(pred))
        
    
    indic = st.checkbox('Plus d\'informations sur les indicateurs?')
    if indic:
        indicateur = st.selectbox(
        'Choisir son indicateur',
        list(table_indic.Row))
        
        st.dataframe(table_indic.loc[table_indic['Row']==indicateur]['Description'])
        
    lime = st.checkbox('Pour plus d\'explications du résultat de la prédiction')
    
    if lime:
        explainer = lime_tabular.LimeTabularExplainer(np.array(data_client), mode="classification",
                                              class_names=['<=50%','>50%'],
                                              feature_names=data_client.columns,
                                             )
        exp = explainer.explain_instance(
            data_row=data_client.iloc[100], 
            predict_fn=loaded_model.predict_proba
            )
        exp.show_in_notebook(show_table=True)
    
    comparatif = st.checkbox('Positionnement dans l\'échantillon')
    
    if comparatif:
        # La pyramide des âges des emprunteurs
        X = round(abs(data_client['DAYS_BIRTH'] / (365)))
        fig2 = plt.figure(figsize =(10, 6))
        sns.histplot(data= data_client, x = X, color = "blue")
        #sns.histplot(data= data_client, x = data_client.loc[data_client['SK_ID_CURR']==int(ref_client)]['DAYS_BIRTH']/ (365), color='green')
        plt.title('Distribution de l\'âge des emprunteurs')
        plt.xlabel('Année')
        st.pyplot(fig2)
        
        # Distribution par sexe
        fig1 = plt.figure(figsize = (10, 5))
        sns.catplot(data = data_client, y = 'CODE_GENDER',kind = 'count')
        plt.xticks(rotation = 60)
        plt.title('Sexe de l\'emprunteur')
        st.pyplot(fig1)
              
        # Les activités professionnelles représentées
        #plt.subplots(figsize =(10, 6))
        #sns.countplot(data = data_client, y = 'OCCUPATION_TYPE')
        #plt.axvspan(data_client.loc[data_client['SK_ID_CURR']==int(ref_client)]['OCCUPATION_TYPE'], facecolor='g', alpha=0.5)
        #plt.title('Métier de l\'emprunteur')
        #st.show()
        
        
        # Distribution du nombre d'années professionnelles
        X = round(abs(data_client['DAYS_EMPLOYED'] / (365)))
        fig3 = plt.subplots(figsize =(10, 6))
        sns.histplot(data= data_client, x = X, color = 'green')
        plt.axvspan(data_client.loc[data_client['SK_ID_CURR']==int(ref_client)]['DAYS_EMPLOYED']/ (365), facecolor='g', alpha=0.5)
        plt.title('Distribution du nombre d\'années professionnelles des emprunteurs')
        plt.xlabel('Années')
        plt.xlim(0, 40)
        st.pyplot(fig3)
    

if __name__ == '__main__':
    main()
