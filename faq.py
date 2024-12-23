from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine,desc,func
from sqlalchemy.orm import sessionmaker
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from models import Base,FAQ
import os
from dotenv import load_dotenv
from langchain_core.prompts.prompt import PromptTemplate
import uuid

load_dotenv()
class LegalFAQProcessor:
    def __init__(self, chroma_client, db_path="sqlite:///legal_faqs.db"):
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2') 
        self.chroma_client = chroma_client  # Persistent Chroma client
        self.collection_name = "legal_faqs"  # ChromaDB collection name

        # SQLite setup
        self.engine = create_engine(db_path)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db_session = self.Session()

        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

    def create_collection_for_faqs(self):
        """
        Create a collection for FAQs in ChromaDB if it doesn't already exist.
        """
        # Initialize the collection
        self.chroma_client.get_or_create_collection(self.collection_name)
        return "success"

    def process_user_question(self, user_question):
        """
        Process the user question:
        1. Check for similarity with existing FAQs in ChromaDB.
        2. If not found, add it to ChromaDB and SQLite.
        3. If found, update its frequency in SQLite.
        """
        try:
            # Correct grammar of the user's question
            user_question = self.correct_grammar(user_question)
            
            # Generate embedding for the question
            user_embedding = self.embeddings.embed_query(user_question)
            
            # Retrieve the collection
            collection = self.chroma_client.get_or_create_collection(self.collection_name)
            
            # Perform similarity search
            search_results = collection.query(
                query_embeddings=[user_embedding],
                n_results=1,  # Get top 1 result
                include=["distances", "metadatas"]  # Include distances and metadata
            )
            print(search_results)
            threshold = 0.2  # Similarity threshold
            # Check if there are any results
            if search_results and search_results["distances"] and search_results["distances"][0]:
                # Similar question found
                if search_results["distances"][0][0] < threshold:
                    similar_question_metadata = search_results["metadatas"][0][0]
                    similar_question_text = similar_question_metadata["question"]

                    # Check if it's already in SQLite
                    faq_row = self.db_session.query(FAQ).filter(FAQ.question == similar_question_text).first()
                    if faq_row:
                        # Increment the frequency count
                        faq_row.cnt_of_freq += 1
                        self.db_session.merge(faq_row)
                        self.db_session.commit()
                    else:
                        # Add the question to SQLite if not found
                        max_id = self.db_session.query(func.max(FAQ.id)).scalar() or 0
                        new_row = FAQ(
                            id=max_id+1,
                            question=similar_question_text,
                            cnt_of_freq=1
                        )
                        self.db_session.add(new_row)
                        self.db_session.commit()
                else:
                    # No similar question found; add it to ChromaDB
                    collection.add(
                        embeddings=[user_embedding],
                        documents=[user_question],
                        metadatas=[{"question": user_question}],
                        ids=[str(uuid.uuid4())]
                    )

                    # Maintain the 1000-limit in SQLite
                    row_count = self.db_session.query(FAQ).count()
                    if row_count >= 1000:
                        # Delete the oldest entry to make room
                        oldest_row = self.db_session.query(FAQ).order_by(FAQ.cnt_of_freq.asc()).first()
                        if oldest_row:
                            self.db_session.delete(oldest_row)
                            self.db_session.commit()

                    max_id = self.db_session.query(func.max(FAQ.id)).scalar() or 0
                    new_row = FAQ(
                        id=max_id+1,
                        question=user_question,
                        cnt_of_freq=1
                    )
                    self.db_session.add(new_row)
                    self.db_session.commit()
            else:
                # No similar question found; add it to ChromaDB
                collection.add(
                    embeddings=[user_embedding],
                    documents=[user_question],
                    metadatas=[{"question": user_question}],
                    ids=[str(uuid.uuid4())]
                )

                # Maintain the 1000-limit in SQLite
                row_count = self.db_session.query(FAQ).count()
                if row_count >= 1000:
                    # Delete the oldest entry to make room
                    oldest_row = self.db_session.query(FAQ).order_by(FAQ.cnt_of_freq.asc()).first()
                    if oldest_row:
                        self.db_session.delete(oldest_row)
                        self.db_session.commit()

                max_id = self.db_session.query(func.max(FAQ.id)).scalar() or 0
                new_row = FAQ(
                    id=max_id+1,
                    question=user_question,
                    cnt_of_freq=1
                )
                self.db_session.add(new_row)
                self.db_session.commit()

            return "success", "FAQ updated or added successfully"
        
        except Exception as e:
            self.db_session.rollback()
            return "error", f"An error occurred while processing the question: {str(e)}"


        
    
    def correct_grammar(self, sentence):
        """
        Corrects grammar for a given sentence using OpenAI or any other LLM.
        """
        prompt = (
            f"Just correct the grammar of the following question, do not provide additional text, do not expand the question: '{sentence}'. Give only the correct question as the output."
        )
        grammer_prompt = PromptTemplate(template=prompt, input_variables=["sentence"])
        llm = ChatGroq(
                model="llama3-8b-8192",
                temperature=0,
                max_tokens=None,
                timeout=None,
                api_key=os.getenv('GROQ_API_KEY')
                )
        grammer_chain= grammer_prompt | llm
        response= grammer_chain.invoke({"sentence":sentence})
        # Extract and return the corrected sentence
        return response.content.strip()
    
    def fetch_top_faqs(self,limit=4):
        try:
            results = self.db_session.query(FAQ).order_by(desc(FAQ.cnt_of_freq)).limit(limit).all()
            questions = [row.question for row in results]
            return questions
        except Exception as e:
            raise Exception(f"An error occurred while fetching FAQs: {str(e)}")