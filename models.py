from sqlalchemy import Column, BigInteger,VARCHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

Base = declarative_base()
db_path="sqlite:///legal_faqs.db"
engine = create_engine(db_path)
class FAQ(Base):
    __tablename__ = 'legal_faqs'

    id = Column(BigInteger, autoincrement=True,primary_key=True)  
    question = Column(VARCHAR(5000), nullable=False)            
    cnt_of_freq = Column(BigInteger, default=0, nullable=False)

Base.metadata.create_all(engine)