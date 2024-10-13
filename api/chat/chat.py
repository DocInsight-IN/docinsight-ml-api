import json

from langchain import hub
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from .prompt import SYSTEM_PROMPT, CONDENSE_QUESTION_PROMPT
from .utils import get_retreiver, reciprocal_rank_fusion
from ml_deploy.models import Conversation, Message

class ChatService:
    def __init__(self, pdf_file_path: str):
        self.conversations = {}

        self.retreiver = get_retreiver()
        self.openai_handler = OpenAICallbackHandler()
        self.llm = OpenAI(temperature=0, callbacks=[self.openai_handler])

        self.system_prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
        self.condense_question_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT)

        self.fusion_prompt = hub.pull("langchain-ai/rag-fusion-query-generation")
        self.generate_queries = (
            self.fusion_prompt | self.llm | StrOutputParser() | (lambda x: x.split("\n"))
        )
        self.fusion_chain = self.generate_queries | self.retreiver.map() | reciprocal_rank_fusion
        self.rag_chain = load_qa_chain(self.llm, chain_type='stuff', prompt=self.system_prompt)
    
    def start_conversation(self, conversation_id):
        self.conversations[conversation_id] = {'messages': []}

    def add_ai_message(self, conversation_id, message):
        ai_msg = {'sender': 'Bot', 'message': message}
        self.conversations[conversation_id]['messages'].append(ai_msg)

    def add_user_message(self, conversation_id, message):
        user_msg = {'sender': 'User', 'message': message}
        self.conversations[conversation_id]['messages'].append(user_msg)

    def get_conversation_buffer_string(self, conversation_id):
        if conversation_id in self.conversations:
            messages = self.conversations[conversation_id]['messages']
            conversation_str = ""
            for message in messages:
                sender = message["sender"]
                content = message['message']
                conversation_str += f"{sender}: {content}\n"
            return conversation_str
        return ""

    def get_response(self, conversation_id, message:str) -> str:
        if message:
            _inputs = RunnableParallel(
                standalone_question=RunnablePassthrough.assign(
                    chat_history=lambda x: self.get_conversation_buffer_string(conversation_id)
                )
                | self.condense_question_prompt
                | self.llm
                | StrOutputParser(),
            )

            similar_docs = self.fusion_chain.invoke({"original_query": message})
            response = self.rag_chain.invoke({"question": message}, input_documents=similar_docs)
            return response
        return "You can ask anything."