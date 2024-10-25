import streamlit as st
from model import LLM_Chat

class App:
    def __init__(self):
        st.set_page_config(page_title="MIMIC IV")
        st.title("Medical Information Mart for Intensive Care")
        st.markdown(f"""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500&display=swap');
                * {{font-family: 'Vazirmatn', sans-serif;
                    }}
                .stChatMessage {{ text-align: right; direction: rtl; }} 
                .stChatInput {{ text-align: right; direction: rtl; }} 
            </style>
        """, unsafe_allow_html=True)

    def display_sidebar(self):
        api_key = st.sidebar.text_input("Enter Cohere API Key", type="password")
        if not api_key:
            st.sidebar.warning('Please Enter Cohere API Key to Continue')
            st.sidebar.info('You Can Get Your API Key From [Cohere Web Page](https://dashboard.cohere.com/welcome/register)')
            st.stop()
        return api_key

    def display_chat(self, chat_history):

        if len(chat_history) == 0 or st.sidebar.button("Reset chat history"):
            return True 

        for msg in chat_history:
            if msg.type == 'AIMessageChunk':
                msg.type = 'ai'
            st.chat_message(msg.type).write(msg.content)
        return False

    def get_user_input(self):
        return st.chat_input(placeholder= 'سوال خود را درباره‌ی پایگاه‌داده MIMIC-IV بپرسید.')

    def display_message(self, message_type, content):
        st.chat_message(message_type).write(content)

    def display_app(self):
        api_key = self.display_sidebar()
        backend = LLM_Chat(api_key)

        if self.display_chat(backend.get_chat_history()):
            backend.reset_chat()

        prompt = self.get_user_input()
        if prompt:
            self.display_message("human", prompt)
            response = backend.process_input(prompt)
            self.display_message("ai", response)

if __name__ == "__main__":
    app = App()
    app.display_app()