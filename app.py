import streamlit as st
from rag import RAGBot

st.title("Cafe FAQ Bot")


@st.cache_resource
def load_bot():
    return RAGBot()


bot = load_bot()

if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

if prompt := st.chat_input("Ask about our cafe..."):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.write(prompt)
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            answer = bot.answer(prompt)
        st.write(answer)
        st.session_state.messages.append({'role': 'assistant', 'content': answer})
