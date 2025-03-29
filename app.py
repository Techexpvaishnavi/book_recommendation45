import pickle
import streamlit as st 
import numpy as np 

st.header("Books Recommender System using Machine Learning")

model = pickle.load(open('artifacts/model.pkl','rb'))
book_name = pickle.load(open('artifacts/book_name.pkl', 'rb'))
pivot_table = pickle.load(open('artifacts/book_pivot.pkl',"rb"))
final_rating = pickle.load(open('artifacts/final_rating.pkl',"rb"))




def fetch_poster(suggestion):
    book_name = []
    book_idx = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(pivot_table.index[book_id])

    for name in book_name[0]:
        ids = np.where(final_rating['title'] == name)[0][0]
        book_idx.append(ids)

    for idx in book_idx:
        url = final_rating.iloc[idx]["img_url"]
        poster_url.append(url)

    return poster_url


def recommendation_book(book_name):
    book_list = []
    book_id = np.where(pivot_table.index == book_name)[0][0]
    distance,suggestion = model.kneighbors(pivot_table.iloc[book_id,:].values.reshape(1,-1),n_neighbors=6)

    poster_url = fetch_poster(suggestion)

    for i in range(len(suggestion)):
        book = pivot_table.index[suggestion[i]]
        for j in book:
            book_list.append(j)

    return book_list,poster_url

    


selected_book = st.selectbox(
    "Type or select a book",
    book_name
)

if st.button('Show Recommendation'):
    Recommendation_book,poster_url = recommendation_book(selected_book)
    col1,col2,col3,col4,col5 = st.columns(5)

    with col1:
        st.text(Recommendation_book[1])
        st.image(poster_url[1])

    with col2:
        st.text(Recommendation_book[2])
        st.image(poster_url[2])
    
    with col3:
        st.text(Recommendation_book[3])
        st.image(poster_url[3])

    with col4:
        st.text(Recommendation_book[4])
        st.image(poster_url[4])

    with col5:
        st.text(Recommendation_book[5])
        st.image(poster_url[5])