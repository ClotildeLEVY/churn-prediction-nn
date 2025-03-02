# Projet réseaux de neurones - M2 TAL  
Marie Delporte-Landat & Clotilde Lévy

## 1. Objectifs du projet  
Ce projet a pour objectif de prédire le départ des clients d'une entreprise en analysant leurs données transactionnelles et comportementales.  
L'intérêt est donc de pouvoir identifier les clients suceptibles de résilier le service, de déterminer les facteurs qui causent le départ _(cf. analyse des corrélations)_ et, par la suite, de pouvoir prévenir ces risques de départ.  

Notre projet est basé sur un modèle de réseau de neurones qui est entraîné sur un ensemble de données clients et intégré à une interface interactive Streamlit. 

## 2. Dataset
Pour réaliser ce projet, nous avons utilisé le dataset [**Teleco Customer Churn**](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), disponible publiquement sur la plateforme Kaggle.

Il est au format CSV, et chaque ligne correspond à un client et contient diverses informations sur ce dernier _(21 caractéristiques au total)_, notamment :  
- les **informations générales du client** : identifiant, sexe, âge, présence d'un partenaire ou de personnes à charge  
- les **données d'abonnement** : type de contrat, durée d'engagement, mode de facturation et de paiement  
- les **services souscrits** : type de connexion internet, accès aux services additionnels (téléphone, streaming, protection des appareils, support technique)  
- la **facturation** : coût de l'abonnement, montant total facturé depuis l'inscription  
- le **churn** qui constitue notre variable à prédire indiquant si le client va quitter ou non l'entreprise  

### Preprocessing du dataset
Avant d'entraîner notre modèle, nous avons procédé à un rééquilibrage des classes **Churn = Yes** et **Churn = No**, puisque notre dataset initial présentait un déséquilibre entre les clients ayant résilié leur abonnement et ceux l'ayant conservé.  
Pour corriger cela, nous avons appliqué une méthode de **sur-échantillonage** de la classe minoritaire afin d'obtenir une répartition équilibrée de nos données et ainsi éviter que le modèle ne soit biaisé en faveur de la classe majoritaire.  


## 3. Méthodologie
Voici la manière dont nous avons procédé pour construire notre projet :  
- sélection d'un dataset spécialement constitué pour de l'analyse et prédiction de churn
- observation des données
- test de méthodes de rééquilibrage des classes afin d'obtenir des résultats équilibrés
- test de modèles de réseaux de neurones
- implémentation du meilleur modèle
- implémentation de l'interface streamlit
- lier le modèle neuronal à l'interface streamlit  

Nous avons travaillé simultanément sur les différentes étapes du projet dans le but d'obtenir la meilleure méthode de préparation des données et meilleur modèle.  
Nous n'avons pas rencontré de difficulté particulière sur l'implémentation du modèle.  
Les difficultés sont arrivées lorsque nous souhaitions lier le modèle à l'interface streamlit.  
De plus streamlit étant une interface dynamique, dès que l'utilisateur interagit avec l'application, tous les graphiques et tableaux disparaissent. Il a donc fallu beaucoup jouer avec l'outil ```st.session_state``` pour résoudre ces problèmes.  



## 4. Implémentation
Notre implémentation du projet s'est réparti en plusieurs parties :  
- observation du dataset dans un notebook (non fourni) pour comprendre les données, la répartition des deux classes à prédire, des tests de différentes méthodes de rééquilibrage des classes
- plusieurs implémentations de modèles neuronaux testés pour obtenir la meilleure accuracy
- implémentation d'un script python exécutable en ligne de commande ```churn_model_with_testpred.py```
- implémentation d'une interface web avec streamlit ```churn_app.py```


## 5. Utilisation des scripts et données
Voici ce qu'il faut savoir pour utiliser nos scripts et l'interface streamlit.  

Deux options sont possibles :  

- utiliser le script ```churn_model_with_testpred.py``` pour tester le modèle en ligne de commande.  
Ce script sert **uniquement** à tester l'efficacité et les résultats de notre modèle en faisant l'entrainement sur notre dataset de train, puis de le tester sur notre dataset de test. **Pour lancer les prédictions sur vos propres données, référez-vous à la partie sur l'interface streamlit.** Une fois les prédictions terminées, le fichier de prédictions généré est directement comparé aux valeurs réelles. Ce script permet de suivre l'entrainement, faire les prédictions sur un fichier de test non annoté, sauvegarder les prédictions, afficher les résultats du modèle en comparant les prédictions aux valeurs réelles.  
Pour lancer ce script : 
> python3 churn_model_with_testpred.py   

Le fichier d'entrainement et de test avec les valeurs réelles sont directement dans le script. Il faudra seulement mentionner le fichier de test sur lequel faire les prédictions ici :  

>  👉 Entrez le chemin du fichier test : data/Telco-Customer-Churn_test.csv  

- utiliser le script ```churn_app.py``` avec l'interface streamlit.  
Ici, tout se passe dans l'interface web.  
Au début, vous devrez déposer votre fichier d'entrainement au format CSV (```fichier ./data/Telco-Customer-Churn_train.csv pour nous```) puis laissez-vous guider par les différents boutons de l'interface.  
Il sera possible d'afficher un aperçu de votre dataset, lancer l'entrainement du modèle. Ensuite une matrice de corrélation s'affiche, mais il faudra s'y référer après les prédictions faites (*tout est expliqué sur l'interface*).  

Ensuite, vous devrez déposer votre fichier de test au format CSV **non annoté** (```fichier ./data/Telco-Customer-Churn_test.csv pour nous```) puis cliquez sur **_Lancer les prédictions_**.  
Ensuite, vous pourrez télécharger les prédictions au format CSV et faire des recherches sur un client spécifique en rentrant sont *customerID*.  
Une fois un ID rentré, les données du client s'affichent puis la prédiction qui lui est associée, ainsi que des explication permettant de comprendre les résultats.  


## 6. Résultats & discussion
Notre modèle de prédiction de churn sur des données clients obtient un meilleur résultat de **0.74 d'accuracy** par rapport à notre dataset d'entrainement et de test.  
Pour utiliser notre interface, il est nécessaire d'avoir un dataset déjà formé avec une partie annotée et une non annotée. Pour l'utilisateur, l'intérêt est d'avoir une quantité limitée de données annotées et à travers notre modèle, de faire des prédictions sur d'autres clients. L'avantage est qu'il n'aura pas besoin d'avoir le même dataset que celui sur lequel nous avons travaillé avec exactement les mêmes colonnes, ce qui rend notre application généralisable.  
Concernant les perspectives d'amélioration, nous souhaiterions trouver un moyen d'afficher des explications plus précises qu'une analyse de corrélation sur pourquoi le client souhaite continuer de souscrire aux services ou partir. Nous avons fait des tests avec la library *Shap* qui permet d'expliquer les sorties d'un modèle de Machine Learning ou Réseaux de Neurones, mais nous ne sommes pas parvenues à l'implémenter correctement pour avoir des résultats cohérent.  