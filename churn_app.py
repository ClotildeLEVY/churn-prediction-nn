import sys
sys.modules["torch.classes"] = None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from data_prep import prepare_data

# Définition du modèle
class ChurnModel(nn.Module):
    def __init__(self, input_dim):
        super(ChurnModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x

# Fonction d'entraînement
def train_model(X_train, y_train, batch_size=64, learning_rate=0.001, epochs=200, patience=10, model_path="churn_model.pth"):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = ChurnModel(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    patience_counter = 0



    if "loss_values" not in st.session_state:
        st.session_state["loss_values"] = []
    if "accuracy_values" not in st.session_state:
        st.session_state["accuracy_values"] = []
    if "plot_fig" not in st.session_state or st.session_state["plot_fig"] is None:
        st.session_state["plot_fig"] = plt.figure(figsize=(8, 5))

    

    loss_values = st.session_state["loss_values"]
    accuracy_values = st.session_state["accuracy_values"]


    plot_placeholder = st.empty()

    
    if st.session_state["plot_fig"] is not None:
        plot_placeholder.pyplot(st.session_state["plot_fig"])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
        
        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total

        loss_values.append(avg_loss)
        accuracy_values.append(accuracy)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(loss_values, label='Loss', color="blue")
        ax.plot(accuracy_values, label="Accuracy", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Evolution des valeurs de loss et accuracy pendant l'entraînement")
        ax.legend()
        
        st.session_state["plot_fig"] = fig
        plot_placeholder.pyplot(st.session_state["plot_fig"])


        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                st.write("Early stopping déclenché")
                break
    
    return model

# Fonction de prédiction
def predict_churn(model_path="churn_model.pth", X_test=None):
    model = ChurnModel(X_test.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()
        predictions = (predictions > 0.5).astype(int)
    
    return np.where(predictions == 1, "Yes", "No")

#Interface Streamlit
st.markdown("""
    <h1 style="text-align: center;">🔍📊 Prédiction de churn sur une clientèle définie</h1>
""", unsafe_allow_html=True)

st.image("./image/churn_telecom_pic.png", caption="Source : Churn Analysis - Analytics Vidhya")

st.markdown("""
<style>
.intro-box {
    background-color: #e6e6fa;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    font-size: 16px;
    line-height: 1.6;
    margin-bottom: 35px;
}
</style>

<div class="intro-box">
    <p>Ce projet vise à anticiper le départ des clients d'une entreprise en analysant leurs comportements ainsi que leurs données transactionnelles. Il permet l'identification de clients à risque de départ, de déterminer les facteurs influençant leur fidélisation grâce à l'analyse de corrélation des variables mais aussi d'aider les entreprises à prendre des décisions stratégiques afin de réduire et/ou prévenir le taux de résiliation.</p>  
    <p>Les prédictions sont effectuées à partir d'un fichier CSV.<br>Notre modèle utilise un réseau de neurones et a obtenu un score maximal de <b>0.76</b> d'accuracy sur nos données de test.</p>  
</div>
""", unsafe_allow_html=True)


train_file = st.file_uploader("Chargez le fichier d'entraînement (Votre dataset annoté au format CSV)", type=["csv"])
if train_file is not None:
    df_train = pd.read_csv(train_file)

    if st.checkbox("Afficher un aperçu du dataset"):
        st.write(df_train.head())

    df_train = prepare_data(df_train)
    X_train = df_train.drop(columns=['Churn', 'customerID'])
    y_train = df_train['Churn']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=['float64', 'int64']))

    if "correlation_fig" in st.session_state:
        st.subheader("Matrice de corrélation entre Churn et autres catégories")
        st.pyplot(st.session_state["correlation_fig"])

    # Bouton d'entraînement du modèle
    if st.button("Entraîner le modèle"):
        with st.spinner("Entraînement en cours..."):
            model = train_model(X_train_scaled, y_train)
        st.success("Modèle entraîné avec succès !")

        # Génération de la heatmap et stockage dans session_state
        if train_file is not None:
            df_corr = df_train.select_dtypes(include=['float64', 'int64'])
            df_corr["Churn"] = y_train.replace({"Yes": 1, "No": 0})

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.session_state["correlation_fig"] = fig 
            st.subheader("Matrice de corrélation entre Churn et autres catégories")
            st.pyplot(fig)
    

# Chargement des données de test
st.subheader("Faire des prédictions sur vos données non annotées")
test_file = st.file_uploader("Chargez le fichier sur lequel faire des prédictions (CSV)", type=["csv"], key="test")

if test_file is not None:
    df_test = pd.read_csv(test_file)
    df_test.set_index("customerID", inplace=True)
    X_test = df_test.select_dtypes(include=['float64', 'int64'])
    X_test_scaled = scaler.transform(X_test)

    # Bouton pour lancer la prédiction
    if st.button("Lancer les prédictions"):
        predictions = predict_churn(X_test=X_test_scaled)
        df_test["Churn_Predicted"] = predictions
        st.session_state["df_predictions"] = df_test

    # Affichage des prédictions si elles existent
    if "df_predictions" in st.session_state:
        st.subheader("Prédictions enregistrées")
        st.dataframe(st.session_state["df_predictions"])

        # Bouton de téléchargement
        csv = st.session_state["df_predictions"].to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger le fichier avec les prédictions", csv, "predictions.csv", "text/csv")

        # Recherche d'un client spécifique
        st.subheader("Rechercher un client spécifique")
        client_id = st.text_input("Entrez l'ID du client")

        if client_id:
            df_predictions = st.session_state["df_predictions"]
            if client_id in df_predictions.index:
                st.write("Données du client :")
                st.dataframe(df_predictions.loc[[client_id]])

                churn_status = df_predictions.loc[client_id, "Churn_Predicted"]
                st.write(f"Prédiction :")
                if churn_status == "Yes":
                    st.warning("**Yes** - ce client est un **churner**.")
                else:
                    st.success("**No** - ce client est un **non-churner**.")

                st.markdown("""
                ### ℹ️ **Interprétation des résultats**
                - Si le statut est **'No'**, le client est **non-churner** (il reste abonné).
                - Si le statut est **'Yes'**, le client est **churner** (il risque de partir).  
                            
                Pour avoir plus de détails sur les motivations des clients, référez-vous au **graphique de corrélation** en haut de page.  
                
                **Lecture de la corrélation :**            
                Plus une valeur est proche de **1** ou **-1**, plus la corrélation est forte.  
                exemple : si à l'intersection de 'Churn' et 'tenure', la valeur est de -0.41, la corrélation est négative et plutôt forte. Cela signifie que plus un client est fidèle, moins il y a de churn.
                """)
            else:
                st.error("❌ Client introuvable.")
