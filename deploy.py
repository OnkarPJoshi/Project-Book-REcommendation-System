import pandas as pd
import streamlit as st
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

st.title('Model Deployment : Book Recommendation')

master_data = pickle.load(open(r'C:\Users\HP\Desktop\courses\Data_Science\DS_Project_Book_Recommendation_System\uncleaned_master_data.pkl','rb'))
famous_books = pickle.load(open(r'C:\Users\HP\Desktop\courses\Data_Science\DS_Project_Book_Recommendation_System\most_famous_books.pkl','rb'))
model_data = pickle.load(open(r'C:\Users\HP\Desktop\courses\Data_Science\DS_Project_Book_Recommendation_System\model_data.pkl','rb'))

st.sidebar.header('User Input Parameters')

reader_userid = st.sidebar.number_input('enter your user_id',min_value=1,step=1)
reader_age = model_data[model_data['User-ID']==reader_userid]['Age'].values[0]
book_read = master_data[master_data['User-ID']==reader_userid]['Book-Title'].values[0]
data = pd.DataFrame({'reader_age':reader_age,
                         'reader_userid ':reader_userid,
                         'book_read':book_read},index=[0])
    
st.subheader('User Input parameters')
st.write(data)


rating_based_similarity= model_data[model_data['Book-Rating']!=0]
rating_based_similarity_model=rating_based_similarity[rating_based_similarity['num_of_books_read']>50].head(50000)
df1 = rating_based_similarity_model.pivot_table(columns='User-ID',index='Book-Title',values='Book-Rating',aggfunc='mean')
df1.fillna(0,inplace=True)
df1=df1.T
df1_rating_based_simi = 1-pairwise_distances(df1.values,metric='cosine')
np.fill_diagonal(df1_rating_based_simi,0)
df1_rating_based_model=pd.DataFrame(df1_rating_based_simi)
df1_rating_based_model.index=rating_based_similarity_model['User-ID'].unique()
df1_rating_based_model.columns=rating_based_similarity_model['User-ID'].unique()
def rec_book_rating_based(user_id):
    sim_users = list(df1_rating_based_model.sort_values(user_id,ascending=False).head(5).index)
    sim_books = [master_data[master_data['User-ID']==user]['Book-Title'].values for user in sim_users]
    sim_books = np.concatenate(sim_books)
    user_books = master_data[master_data['User-ID']==user_id]['Book-Title'].values
    return list(set(sim_books)-set(user_books))[0:50]



model_data.loc[model_data['avg_rating']<=2,'book_status']='worst'
model_data.loc[(model_data['avg_rating']>2) & (model_data['avg_rating']<=4),'book_status']='bad'
model_data.loc[(model_data['avg_rating']>4) & (model_data['avg_rating']<=6),'book_status']='good'
model_data.loc[(model_data['avg_rating']>6) & (model_data['avg_rating']<=8),'book_status']='better'
model_data.loc[model_data['avg_rating']>8,'book_status']='best'


user_collab_data = model_data[['avg_rating','num_times_book_rated','num_of_books_read','book_status']].head(50000)
user_collab_data=user_collab_data[(user_collab_data['avg_rating']!=0) & (user_collab_data['num_of_books_read']>50)]
user_collab_data=pd.get_dummies(user_collab_data)
from sklearn.preprocessing import StandardScaler
std_sca = StandardScaler()
std_model_data = std_sca.fit_transform(user_collab_data)
from sklearn.neighbors import NearestNeighbors
n_neighbors=6
model = NearestNeighbors(n_neighbors=n_neighbors,algorithm='ball_tree')
model.fit(std_model_data)
distance,user_id = model.kneighbors(std_model_data)
def recommend_book_read(book_title):
    book_list=[]
    book_id=model_data[model_data['Book-Title']==book_title].index
    book_id=book_id[0]
    for i in user_id[book_id][1:n_neighbors]:
        book_list.append(model_data.loc[i,'Book-Title'])
    return book_list

age_model_data=model_data[['User-ID','Book-Title','Age','num_of_books_read']].head(200000)
age_based_model_data=age_model_data[age_model_data['num_of_books_read']>50]
age_based_model_data.drop_duplicates(inplace=True)
teen_age = age_based_model_data[(age_based_model_data['Age']<21)]
young_age = age_based_model_data[(age_based_model_data['Age']>20) & (age_based_model_data['Age']<31)]
adult_age = age_based_model_data[(age_based_model_data['Age']>30) & (age_based_model_data['Age']<41)]
mature_age = age_based_model_data[(age_based_model_data['Age']>40) & (age_based_model_data['Age']<61)]
seniors_age = age_based_model_data[(age_based_model_data['Age']>60)]

teen_df = teen_age.pivot_table(columns='User-ID',index='Book-Title',values='Age',aggfunc='mean',fill_value=0)
teen_df=teen_df.T
teen_df_sim = 1-pairwise_distances(teen_df.values,metric='cosine')
np.fill_diagonal(teen_df_sim,0)
teen_sim_model = pd.DataFrame(teen_df_sim)
teen_sim_model.index = teen_age['User-ID'].unique()
teen_sim_model.columns = teen_age['User-ID'].unique()
def rec_book_teen_age(user_id):
    sim_users_teen = list(teen_sim_model.sort_values(user_id,ascending=False).head(5).index)
    sim_book_teen = [master_data[master_data['User-ID']==user]['Book-Title'].values for user in sim_users_teen]
    sim_book_teen = np.concatenate(sim_book_teen)
    user_book_teen = master_data[master_data['User-ID']==user_id]['Book-Title'].values
    return list(set(sim_book_teen)-set(user_book_teen))[0:50]

young_df = young_age.pivot_table(columns='User-ID',index='Book-Title',values='Age',aggfunc='mean',fill_value=0)
young_df=young_df.T
young_df_sim = 1-pairwise_distances(young_df.values,metric='cosine')
np.fill_diagonal(young_df_sim,0)
young_sim_model = pd.DataFrame(young_df_sim)
young_sim_model.index = young_age['User-ID'].unique()
young_sim_model.columns = young_age['User-ID'].unique()
def rec_book_young_age(user_id):
    sim_users_young = list(young_sim_model.sort_values(user_id,ascending=False).head(5).index)
    sim_book_young = [master_data[master_data['User-ID']==user]['Book-Title'].values for user in sim_users_young]
    sim_book_young = np.concatenate(sim_book_young)
    user_book_young = master_data[master_data['User-ID']==user_id]['Book-Title'].values
    return list(set(sim_book_young)-set(user_book_young))[0:50]

adult_df = adult_age.pivot_table(columns='User-ID',index='Book-Title',values='Age',aggfunc='mean',fill_value=0)
adult_df=adult_df.T
adult_df_sim = 1-pairwise_distances(adult_df.values,metric='cosine')
np.fill_diagonal(adult_df_sim,0)
adult_sim_model = pd.DataFrame(adult_df_sim)
adult_sim_model.index = adult_age['User-ID'].unique()
adult_sim_model.columns = adult_age['User-ID'].unique()
def rec_book_adult_age(user_id):
    sim_users_adult = list(adult_sim_model.sort_values(user_id,ascending=False).head(5).index)
    sim_book_adult = [master_data[master_data['User-ID']==user]['Book-Title'].values for user in sim_users_adult]
    sim_book_adult = np.concatenate(sim_book_adult)
    user_book_adult = master_data[master_data['User-ID']==user_id]['Book-Title'].values
    return list(set(sim_book_adult)-set(user_book_adult))[0:50]


mature_df = mature_age.pivot_table(columns='User-ID',index='Book-Title',values='Age',aggfunc='mean',fill_value=0)
mature_df=mature_df.T
mature_df_sim = 1-pairwise_distances(mature_df.values,metric='cosine')
np.fill_diagonal(mature_df_sim,0)
mature_sim_model = pd.DataFrame(mature_df_sim)
mature_sim_model.index = mature_age['User-ID'].unique()
mature_sim_model.columns = mature_age['User-ID'].unique()
def rec_book_mature_age(user_id):
    sim_users_mature = list(mature_sim_model.sort_values(user_id,ascending=False).head(5).index)
    sim_book_mature = [master_data[master_data['User-ID']==user]['Book-Title'].values for user in sim_users_mature]
    sim_book_mature = np.concatenate(sim_book_mature)
    user_book_mature = master_data[master_data['User-ID']==user_id]['Book-Title'].values
    return list(set(sim_book_mature)-set(user_book_mature))[0:50]

senior_df = seniors_age.pivot_table(columns='User-ID',index='Book-Title',values='Age',aggfunc='mean',fill_value=0)
senior_df=senior_df.T
senior_df_sim = 1-pairwise_distances(senior_df.values,metric='cosine')
np.fill_diagonal(senior_df_sim,0)
senior_sim_model = pd.DataFrame(senior_df_sim)
senior_sim_model.index = seniors_age['User-ID'].unique()
senior_sim_model.columns = seniors_age['User-ID'].unique()
def rec_book_senior_age(user_id):
    sim_users_senior = list(senior_sim_model.sort_values(user_id,ascending=False).head(5).index)
    sim_book_senior = [master_data[master_data['User-ID']==user]['Book-Title'].values for user in sim_users_senior]
    sim_book_senior = np.concatenate(sim_book_senior)
    user_book_senior = master_data[master_data['User-ID']==user_id]['Book-Title'].values
    return list(set(sim_book_senior)-set(user_book_senior))[0:50]
           
def new_user_rec(user_id):
     user_books = set(master_data[master_data['User-ID']==user_id]['Book-Title'])
     return list(set(famous_books)-user_books) 
 
selection = st.selectbox('Select Recommendation Model',['Popular Books for Early Stage Readers',
                                           'Books Based on Ratings','Books Based on Age',
                                           'Books based on Previously Read Book'])

if selection=='Popular Books for Early Stage Readers':
    st.title('Famous Books')      
    st.write(new_user_rec(reader_userid))
elif selection=='Books Based on Ratings':
    st.write(rec_book_rating_based(reader_userid))
elif selection=='Books based on Previously Read Book':
    st.write(recommend_book_read(book_read))
elif selection=='Books Based on Age':
    if reader_age<21:
        st.write(rec_book_teen_age(reader_userid))
    elif reader_age<31 and reader_age>20:
        st.write(rec_book_young_age(reader_userid))
    elif reader_age<41 and reader_age>30:
        st.write(rec_book_adult_age(reader_userid))
    elif reader_age<61 and reader_age>40:
        st.write(rec_book_mature_age(reader_userid))
    else:
        st.write(rec_book_senior_age(reader_userid))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    