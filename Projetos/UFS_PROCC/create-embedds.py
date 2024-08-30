#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import chromadb
from sentence_transformers import SentenceTransformer


# ## Importing Data

# In[2]:


def map_age(x, context):
    if x == 1:
        return context+" fifty to sixty years "
    elif x == 2:
        return context+" sixty-five to eighty years "

def map_health(x, context):
    if x == -1:
        return context+" refused answer "
    elif x == 1:
        return context+" is excellent "
    elif x == 2:
        return context+" is very good "
    elif x == 3:
        return context+" is good "
    elif x == 4:
        return context+" is fair "
    elif x == 5:
        return context+" is poor "


def map_employment(x, context):
     if x == -1:
        return context+" refused answer "
     elif x == 1:
        return context+" Working full-time "
     elif x == 2:
        return context+" Working part-time "
     elif x == 3:
        return context+" Retired "
     elif x == 4:
        return context+" Not working at this time "


def map_sleeping(x, context):
     if x == -0:
        return " No "+context
     elif x == 1:
        return context

def map_medication_sleep(x):
    if x == -1:
        return "The patient refused answer if use Prescription Sleep Medication ."
    elif x == 1:
        return "The patient Use regularly Prescription Sleep Medication ."
    elif x == 2:
        return "The patient Use occasionally Prescription Sleep Medication ."
    elif x == 3:
        return "Thepatient  Do not use Prescription Sleep Medication ."
    
def map_racial(x):
    if x == -1:
        return "The patient refused answer its ethnic racial ."
    elif x == 1:
        return " The patient's racial is white ."
    elif x == 2:
        return " The patient's racial is black ."
    elif x == 3:
        return " The patient's racial other ."
    elif x == 4:
        return " The patient's racial Hispanic ."
    elif x == 5:
        return " The patient's racial mixed ."
    elif x == -2:
        return "The patient's was  not asked. "


def map_gender(x):
     if x == -2:
        return " The gender was not asked "
     elif x == -1:
        return "The patient refused answer its gender"
     elif x == 1:
        return " The patient gender is Male "
     elif x == 2:
        return " The Female "

def map_class(x, context ):
    if x == 1:
        return context+" zero to one doctors "
    elif x == 2:
        return context+" two to three doctors "
    elif x == 3:
        return context+" four or more doctors "
    


# In[3]:


# Get the current working directory
current_dir = os.getcwd()

# Append the relative path to the file
file_path = os.path.join(current_dir, 'npha-doctor-visits.csv')

# Use the file_path in your code
df = pd.read_csv(file_path)


df.head()


# In[4]:


df['Age'] = df['Age'].apply(map_age, context=" This patient has ")
df['Phyiscal Health'] = df['Phyiscal Health'].apply(map_health, context=" and a Phyiscal Health that ")
df['Mental Health'] = df['Mental Health'].apply(map_health,  context=" and a Mental Health that ")
df['Dental Health'] = df['Dental Health'].apply(map_health,  context=" and a Dental Health that ")
df['Employment'] = df['Employment'].apply(map_employment, context="The patient employment status is ")
df['Stress Keeps Patient from Sleeping'] = df['Stress Keeps Patient from Sleeping'].apply(map_sleeping, context="Stress Keeps Patient from Sleeping, ")
df['Medication Keeps Patient from Sleeping'] = df['Medication Keeps Patient from Sleeping'].apply(map_sleeping, context=" Medication Keeps Patient from Sleeping ,")
df['Pain Keeps Patient from Sleeping'] = df['Pain Keeps Patient from Sleeping'].apply(map_sleeping, context=' Pain Keeps Patient from Sleeping , ')
df['Bathroom Needs Keeps Patient from Sleeping'] = df['Bathroom Needs Keeps Patient from Sleeping'].apply(map_sleeping, context=" Bathroom Needs Keeps Patient from Sleeping and")
df['Uknown Keeps Patient from Sleeping'] = df['Uknown Keeps Patient from Sleeping'].apply(map_sleeping, context=" Uknown things Keeps Patient from Sleeping and")
df['Trouble Sleeping'] = df['Trouble Sleeping'].apply(map_sleeping,context='Trouble Sleeping is present' )
df['Prescription Sleep Medication'] = df['Prescription Sleep Medication'].apply(map_medication_sleep)
df['Race'] = df['Race'].apply(map_racial)
df['Gender'] = df['Gender'].apply(map_gender)
df['Number of Doctors Visited'] = df['Number of Doctors Visited'].apply(map_class, context="The total count of different doctors the patient has seen was")


# In[5]:


name_labels = df.columns.values
sentences= df.to_string(header=False,
                  index=False,
                  index_names=False).split('\n')


# ### Creating Embeddings https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# In[7]:


transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = transformer.encode(sentences[0:50])
print(embeddings)


# ## Saving in ChromaDb

# In[8]:


import chromadb.utils.embedding_functions as embedding_functions
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="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",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# In[9]:


from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="chroma_file/")
collection = client.get_or_create_collection(name="collection03", embedding_function=huggingface_ef)


# In[10]:


text = sentences[0:50]
collection.add(
    documents=text,
    ids=[f"id{i}" for i in range(len(text))]
)



# In[11]:


result = collection.query(
    query_texts=["Patients that sleep well"],
    n_results=1
)
# result = collection.query(
#     query_embeddings=embedding_query,
#     n_results=1
# )
print(result)


# In[ ]:




