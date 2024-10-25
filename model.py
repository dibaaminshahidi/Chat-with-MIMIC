from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import StreamlitChatMessageHistory
from langchain_cohere import ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings


class LLM_Chat:
    def __init__(self, api_key):

        self.api_key = api_key

        self.embeddings = CohereEmbeddings(
            cohere_api_key = api_key,
            model = "embed-multilingual-light-v3.0",
            max_retries = 5,
            request_timeout = 20
        )

        self.vector_store = Chroma(
        collection_name = "mimic",
        embedding_function = self.embeddings,
        persist_directory = "./collections",
        )   
    
        self.llm = ChatCohere(cohere_api_key=self.api_key, temperature= 0)
        self.chat_history = StreamlitChatMessageHistory(key="special_app_key")
        self.system_prompt = (
            "از محتوای بافتاری زیر برای پاسخ به سوال استفاده کن"
                        "اگر پاسخ سوال را نمی‌دانی، بگو نمی‌دانم. "
                        "جواب رو در نهایت به فارسی بدی حتما"
                        "بافتار: {context}")
        
        self.prompt_template = ChatPromptTemplate.from_messages(
                            [("system", self.system_prompt),
                             ("placeholder", "{chat_history}"),
                             ("human", "{input}"),
                            ]
                        )


        self.chain_with_message_history = RunnableWithMessageHistory(
            create_stuff_documents_chain(self.llm, self.prompt_template),
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        self.retriever = self.vector_store.as_retriever(search_type = "mmr",
                                      search_kwargs = {"fetch_k": 20, "k": 10})

        # self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
        self.chain = create_retrieval_chain(self.retriever, self.chain_with_message_history)

        

    def reset_chat(self):
        self.chat_history.clear()
        # self.chat_history.add_ai_message("Ask about MIMIC IV tables")

    def get_chat_history(self):
        return self.chat_history.messages

    def process_input(self, prompt):
        config = {"configurable": {"session_id": "any"}}
        try:
            response = self.chain.invoke({"input": prompt},  config)

            print(response)
            return response['answer']
        except Exception as error:
            # print(error)
            return str(error)