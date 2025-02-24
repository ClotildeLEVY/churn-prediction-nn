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


## 4. Implémentation

## 5. Résultats & discussion
