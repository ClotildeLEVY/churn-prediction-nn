import os
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset


from data_prep import prepare_data

# 1. Charger et préparer les données
def load_and_prepare_data(csv_file, is_train=True, scaler=None):
    df = pd.read_csv(csv_file)

    if is_train:
        df = prepare_data(df)
        X = df.drop(columns=['Churn'])
        y = df['Churn']
    else:
        X = df
        y = None

    # Normalisation
    if scaler:
        X_scaled = scaler.transform(X.select_dtypes(include=['float64', 'int64']))
    else:
        X_scaled = X.select_dtypes(include=['float64', 'int64']).values

    return X, X_scaled, y


# 2. Définition du modèle
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


# 3. Entraînement du modèle
def train_model(X_train, y_train, batch_size=64, learning_rate=0.001, epochs=200, patience=10, model_path="churn_model.pth"):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ChurnModel(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []
    accuracy_history = []
    best_loss = float('inf')
    patience_counter = 0

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

            # Calcul de l'accuracy
            predictions = (outputs > 0.5).float()
            correct += (predictions == target).sum().item()
            total += target.size(0)

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total

        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4%}")

        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label="Training Loss")
    plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, label="Training Accuracy", linestyle="dashed")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss & Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


# 4. Fonction de prédiction avec un dataset
def predict_churn(model_path="churn_model.pth", scaler=None):
    test_csv = input("\n👉 Entrez le chemin du fichier test : ").strip()

    if not os.path.exists(test_csv):
        print("❌ Fichier introuvable. Vérifiez le chemin.")
        return

    X_test, X_test_scaled, _ = load_and_prepare_data(test_csv, is_train=False, scaler=scaler)

    model = ChurnModel(X_test_scaled.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    # Prédictions
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()
        predictions = (predictions > 0.5).astype(int)

    # Sauvegarde des prédictions
    output_df = X_test.copy()
    output_df["Churn_Predicted"] = predictions
    output_csv = "predictions.csv"
    output_df.to_csv(output_csv, index=False)

    print(f"\n✅ Prédictions sauvegardées dans : {output_csv}")


# 5. Fonction d'évaluation des prédictions
def evaluate_predictions(predictions_csv, actual_csv):
    pred_df = pd.read_csv(predictions_csv)
    actual_df = pd.read_csv(actual_csv)

    if len(pred_df) != len(actual_df):
        print("❌ Erreur : Les fichiers n'ont pas le même nombre de lignes.")
        return

    # Convertir 'Yes' / 'No' en 1 / 0
    actual_df["Churn"] = actual_df["Churn"].map({"Yes": 1, "No": 0})

    # Comparaison des prédictions
    y_pred = pred_df["Churn_Predicted"].values
    y_actual = actual_df["Churn"].values

    # Calcul de l'accuracy
    accuracy = accuracy_score(y_actual, y_pred)
    print(f"\n🎯 Accuracy : {accuracy:.4%}")

    # Matrice de confusion
    cm = confusion_matrix(y_actual, y_pred)
    print("\n📊 Matrice de confusion :\n", cm)

    # Rapport de classification
    print("\n📜 Rapport de classification :\n", classification_report(y_actual, y_pred))

    # Affichage heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    plt.show()


# 6. Script principal
if __name__ == "__main__":
    # Entraînement du modèle
    train_csv = "./data/Telco-Customer-Churn_train.csv"
    X_train, X_train_scaled, y_train = load_and_prepare_data(train_csv, is_train=True)

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=['float64', 'int64']))

    print("\n🚀 Entraînement du modèle en cours...")
    trained_model = train_model(X_train_scaled, y_train)

    # Demande du dataset test et prédiction
    predict_churn(scaler=scaler)
    # Évaluation du modèle après prédiction (comparaison prédictions/valeurs réelles)
    evaluate_predictions("predictions.csv", "data/Telco-Customer-Churn_test_actual.csv")

