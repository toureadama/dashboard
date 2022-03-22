import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import lime
from lime import lime_tabular
import json
from urllib.request import urlopen
import joblib


@st.cache
def charge_donnees(num_rows, nb_ech):
    path = "C:\\Users\\toure\\Desktop\\OpenClassrooms\\Projet 7\\donnees\\"
    
    data = pd.read_csv(path+"donnees_traites.csv", nrows= num_rows)
    data_client_1 = pd.read_csv(path+"donnees_traites_1.csv")
    colonnes_fr = pd.read_csv(path+"colonnes.csv")
    table_indic = pd.read_csv(path+"HomeCredit_columns_description.csv", encoding= 'unicode_escape')
    
    data = data[list(colonnes_fr.iloc[:,1])]   
    data_client_1 = data_client_1[list(colonnes_fr.iloc[:,1])]
    
    nb_echant = min(nb_ech,data_client_1.shape[0])
    data_echant = pd.concat([data[:nb_echant], data_client_1[:nb_echant]])
    data_echant.reset_index(drop=True, inplace=True)
    
    data_client = data_echant.drop(['TARGET'], axis=1)
    references_test = data_client.SK_ID_CURR
    
    return data, data_client, table_indic, references_test

def main():
    data, data_client, table_indic, references_test = charge_donnees(10000, 50)
    loaded_model = joblib.load('C://Users//toure//Desktop//OpenClassrooms//Projet 7//logreg_housing.joblib')
    
    st.title('Prêt à dépenser: Credit Scoring')
    
    st.sidebar.header("Informations sur le client")
    
    ref_client = st.sidebar.selectbox(
        'La référence du client',
        list(references_test))
    
    st.sidebar.write("Nombre de crédits dans l'échantillon", 
                    len(references_test))
    
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
        # Appel de l'API :
        API_url = "http://127.0.0.1:5000/predict?SK_ID_CURR=" + str(ref_client)
        with st.spinner('Chargement du score client...'):
            json_url = urlopen(API_url)
            prediction = json.loads(json_url.read())
               
        st.write("Risque de défaut client : {:.0f} %".format(round((prediction*100), 2)))
        
        if prediction <= 0.5:
            st.markdown("<h2 style='text-align: center;color: green'>Crédit accordé</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center;color: red'>Crédit réfusé</h2>", unsafe_allow_html=True)
            
    
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
            data_row=data_client.iloc[0], 
            predict_fn=loaded_model.predict_proba
            )
        #exp.show_in_notebook(show_table=True)
        html = exp.as_html()
        components.html(html, height=800)
    
    comparatif = st.checkbox('Positionnement dans l\'échantillon')
    
    if comparatif:
        # La pyramide des âges des emprunteurs
        X = round(abs(data['DAYS_BIRTH'] / (365)))
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
        #plt.axvspan(data_client.loc[data_client['SK_ID_CURR']==int(ref_client)]['DAYS_EMPLOYED']/ (365), facecolor='g', alpha=0.5)
        plt.title('Distribution du nombre d\'années professionnelles des emprunteurs')
        plt.xlabel('Années')
        plt.xlim(0, 40)
        st.pyplot(fig3)
    

if __name__ == '__main__':
    main()