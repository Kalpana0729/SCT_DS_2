#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:\\Users\\kalpa\\Downloads\\titanic\\gender_submission.csv")   # adjust filename if inside the zip
df.head()


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:\\Users\\kalpa\\Downloads\\titanic\\test.csv")   # adjust filename if inside the zip
df.head()


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:\\Users\\kalpa\\Downloads\\titanic\\train.csv")   # adjust filename if inside the zip
df.head()


# In[5]:


df.isnull().sum()


# In[20]:


print(df.columns)


# In[25]:


# Fill Age if exists
if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked if exists
if 'Embarked' in df.columns:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Drop Cabin if exists
if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)


# In[26]:


df['Survived'].value_counts(normalize=True).plot(kind='bar', color=['red', 'green'])
plt.title("Survival Rate (0=Dead, 1=Survived)")
plt.show()


# In[45]:


if 'Age' in df.columns and df['Age'].notnull().sum() > 0:
    df['Age'].fillna(df['Age'].median(), inplace=True)


# In[46]:


if 'Embarked' in df.columns and df['Embarked'].notnull().sum() > 0:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


# In[47]:


# Fill numeric NaNs with median if possible
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    if df[col].notnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)


# In[48]:


print(df.isnull().sum())


# In[49]:


import pandas as pd

# Load dataset (adjust filename if needed)
df = pd.read_csv("C:\\Users\\kalpa\\Downloads\\titanic\\test.csv")

# ---------------------------
# SAFER CLEANING PIPELINE
# ---------------------------

# Handle numeric columns (Age, Fare, etc.)
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    if df[col].isnull().sum() > 0 and df[col].notnull().sum() > 0:
        # Fill with median if not completely empty
        df[col].fillna(df[col].median(), inplace=True)

# Handle categorical columns (Embarked, Sex, etc.)
for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0 and df[col].notnull().sum() > 0:
        # Fill with mode if not completely empty
        df[col].fillna(df[col].mode()[0], inplace=True)

# Drop high-missing or unnecessary columns if present
drop_cols = ['Cabin', 'Ticket', 'Name', 'PassengerId']
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# Encode Sex if exists
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Encode Embarked if exists
if 'Embarked' in df.columns:
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Final check
print("Remaining missing values:\n", df.isnull().sum())
print("\nFinal columns:\n", df.columns)


# In[50]:


import pandas as pd

# Load dataset (adjust filename if needed)
df = pd.read_csv("C:\\Users\\kalpa\\Downloads\\titanic\\train.csv")

# ---------------------------
# SAFER CLEANING PIPELINE
# ---------------------------

# Handle numeric columns (Age, Fare, etc.)
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    if df[col].isnull().sum() > 0 and df[col].notnull().sum() > 0:
        # Fill with median if not completely empty
        df[col].fillna(df[col].median(), inplace=True)

# Handle categorical columns (Embarked, Sex, etc.)
for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0 and df[col].notnull().sum() > 0:
        # Fill with mode if not completely empty
        df[col].fillna(df[col].mode()[0], inplace=True)

# Drop high-missing or unnecessary columns if present
drop_cols = ['Cabin', 'Ticket', 'Name', 'PassengerId']
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# Encode Sex if exists
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Encode Embarked if exists
if 'Embarked' in df.columns:
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Final check
print("Remaining missing values:\n", df.isnull().sum())
print("\nFinal columns:\n", df.columns)


# In[51]:


df['Survived'].value_counts(normalize=True).plot(kind='bar', color=['red', 'green'])
plt.title("Survival Rate (0=Dead, 1=Survived)")
plt.show()


# In[52]:


sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()


# In[53]:


sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()


# In[54]:


sns.histplot(df[df['Survived']==1]['Age'], bins=20, color='green', label='Survived', kde=True)
sns.histplot(df[df['Survived']==0]['Age'], bins=20, color='red', label='Did Not Survive', kde=True)
plt.legend()
plt.title("Age Distribution by Survival")
plt.show()


# In[55]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[56]:


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Survival Distribution
sns.countplot(x='Survived', data=df, palette='coolwarm')
plt.title("Overall Survival Distribution (0 = Died, 1 = Survived)")
plt.show()

# 2. Survival by Gender
if 'Sex' in df.columns:
    sns.barplot(x='Sex', y='Survived', data=df, palette='viridis')
    plt.title("Survival Rate by Gender")
    plt.show()

# 3. Survival by Passenger Class
if 'Pclass' in df.columns:
    sns.barplot(x='Pclass', y='Survived', data=df, palette='mako')
    plt.title("Survival Rate by Passenger Class")
    plt.show()

# 4. Age Distribution
if 'Age' in df.columns:
    sns.histplot(df['Age'], bins=20, kde=True, color='blue')
    plt.title("Age Distribution of Passengers")
    plt.show()

    # Age vs Survival
    sns.histplot(data=df, x='Age', hue='Survived', bins=20, kde=True, palette=['red','green'])
    plt.title("Age Distribution by Survival")
    plt.show()

# 5. Fare vs Survival
if 'Fare' in df.columns:
    sns.boxplot(x='Survived', y='Fare', data=df, palette='Set2')
    plt.title("Fare Paid vs Survival")
    plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




