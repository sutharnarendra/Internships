#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

#Loading the dataset
df=pd.read_csv("/content/netflix_titles.csv")

#Display basic information
print("Initial Dataset Info: ")
df.info()

#Displaying first 5 rows
print("\nFirst 5 Records: ")
print(df.head())

#Check for missing values
print("\nMissing Values per Columns: ")
print(df.isnull().sum())

#Heatmap of null values
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

#Check for duplicate records
print("\nNumber of Duplicate Records: ",df.duplicated().sum())

#Drop duplicate records
df = df.drop_duplicates()

#Handling Missing values: 
#Filling 'rating' missing values with the mode
if df['rating'].isnull().sum() > 0:
  df['rating'].fillna(df['rating'].mode()[0], inplace=True)

#Fillig 'country' missing values with the mode
if df['country'].isnull().sum() > 0:
  df['country'].fillna(df['country'].mode()[0], inplace=True)

#Fill 'cast' and 'dicrector' missing values with 'Not Available'
df['cast'].fillna("Not Available", inplace=True)
df['director'].fillna("Not Available", inplace=True)

#Fill 'date_added' missing values with a placeholder
df['date_added'].fillna("Unknown", inplace=True)

#Ensuring correct data types
#Converting 'date_added' to datetime where possible
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

#Re-checking data types and missing values after transformation
print("\nDataset Info After Cleaning: ")
df.info()

#Using NumPy to Perform a transformation: count title length
#Create a new column 'tital_length' for the number of characters in the title
df['title_length'] = np.vectorize(len)(df['title'])

#Correlation matrix
plt.figure(figsize=(6,4))
sns.heatmap(df[['title_length']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#Filtering : Showing Only TV shows released after 2015
filtered_df = df[(df['type'] == 'TV Show') & (df['release_year']>2015)]
print("\nFiltered TV Shows After 2015: ")
print(filtered_df[['title','release_year']].head())

#Sort data by release year and show ID
df_sorted = df.sort_values(by=['release_year','show_id'])

#Group by types and show count
type_counts = df['type'].value_counts()
print("\nContent Type Distribution: ")
print(type_counts)

#Summary Statistics
print("\nSummary Statistics: ")
print(df.describe(include='all'))

#Label encode 'type', 'rating' and 'country'
label_enc = LabelEncoder()
df['type_encoded'] = label_enc.fit_transform(df['type'])
df['rating_encoded'] = label_enc.fit_transform(df['rating'])
df['country_encoded'] = label_enc.fit_transform(df['country'])

#Saving the cleaned dataset to a new CSV
df.to_csv("netflix_title_cleaned.csv",index=False)
print(pd.read_csv("netflix_title_cleaned.csv"))
