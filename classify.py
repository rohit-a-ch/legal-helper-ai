from langchain_core.prompts.prompt import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
class Query:
    def __init__(self):
        # self.llm = ChatGoogleGenerativeAI(
        #         model="gemini-1.5-pro",
        #         temperature=0.7,
        #         max_tokens=None,
        #         timeout=None,
        #         api_key=os.getenv('GEMINI_API_KEY')
        #     )
        self.llm=ChatGroq(
            model="llama3-8b-8192",
            temperature=0,
            max_tokens=None,
            timeout=None,
            api_key=os.getenv('GROQ_API_KEY')
        )
    def classify_question(self, question: str) -> str:  
        """
        Uses an LLM to classify the question as either 'conversational' or 'legal'.
        """
        prompt = (
            "Classify the following question as 'conversational_query' or 'legal_query':\n"
            f"Question: {question}\n\n"
            "Answer only with 'conversational_query' or 'legal_query'."
        )
        classify_prompt = PromptTemplate(template=prompt, input_variables=["question"])

        try:
            chain = classify_prompt | self.llm

            response = chain.invoke({ 
                "question": question
            })
            
            classification = response.content.strip().lower()
            if classification not in {"conversational_query", "legal_query"}:
                raise ValueError("Unexpected classification result.")
            return classification

        except Exception as e:
            raise Exception(f"Error Classifying {str(e)}")


    def call_conversational_tool(self, question: str) -> str: 
        """Handle conversational questions using a conversational tool."""
        prompt_template = PromptTemplate(
                                        input_variables=["question"],
                                        template=(
                                            "System: You are a conversational assistant who engages in natural, friendly, and meaningful dialogue. You are limited to casual and conversational topics and will not provide detailed technical, academic, or specialized information. "
                                            "If a topic goes beyond casual conversation (like coding, technical details, academic queries, etc.), politely decline and steer the conversation back to general, friendly topics. "
                                            "Always respond in a conversational, engaging, and professional tone suitable for a friendly chat. "
                                            "Do not explain your features, capabilities, or the system you are running on unless explicitly asked.\n\n"
                                            f"Human Question: '{question}'\n\n"
                                            "Assistant:"
                                        )
                                    )
        try:

            chain = prompt_template | self.llm

            response = chain.invoke({  
                "question": question
            })

            conversational_result =response.content.strip()
            

            return {'result': conversational_result}
        
        except Exception as e:
            raise Exception(f"Failed to handle conversational question {str(e)}")