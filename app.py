from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone_text.sparse import BM25Encoder
from bot import CocktailsBot
import gradio as gr
import os

INDEX_NAME = 'cocktails'
MEMORY_INDEX_NAME = 'chatmemory'
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

bm25 = BM25Encoder().load("bm25.json")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=reranker, top_n=10)

llm = ChatGroq(
    temperature=0.1, 
    model_name="llama3-8b-8192", 
    groq_api_key=GROQ_API_KEY,
    max_tokens=8192
)

hybrid_vector_store = PineconeHybridSearchRetriever(
    index=index,
    embeddings=embeddings,
    sparse_encoder=bm25,
    top_k=30,
    alpha=0.7
)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=hybrid_vector_store
)

memory_vector_store = PineconeVectorStore.from_documents(
    documents=[],
    embedding=embeddings,
    index_name=MEMORY_INDEX_NAME
)

memory_vector_store.add_documents(documents=[Document(
    page_content="I like lemons",
    metadata={},
)])
document_contents = 'History of users messages. \
Please provide a structured response with the following keys: \
- `query`: The parsed question from the user. \
- `filter`: Any relevant filters based on the metadata fields. \
If no filters are applicable, set `filter` to "NO_FILTER".'
memory_retriever = SelfQueryRetriever.from_llm(
    llm, 
    memory_vector_store, 
    document_contents=document_contents, 
    metadata_field_info=[],
    verbose=False
)

bot = CocktailsBot(
    llm=llm, 
    retriever=retriever, 
    memory_vector_store=memory_vector_store, 
    memory_retriever=memory_retriever
)
    
    
def message_respond(message, history):
    if len(message) > 100:
        return "Your message exceeds the 100 character limit. Please try shortening your request."
    answer = bot.get_answer(message)
    return answer


gr.ChatInterface(
    fn=message_respond,
    type="messages",
    title="Cocktail Advisor Chat",
    description="Here you can ask any questions in the context of the 'Cocktails' dataset.",
    theme=gr.themes.Soft(),
    examples=["What are the 5 cocktails containing lemon?", 
              "What are the 5 non-alcoholic cocktails containing sugar?", 
              "What are my favourite ingredients?", 
              "Recommend 5 cocktails that contain my favourite ingredients",
              "Recommend a cocktail similar to “Hot Creamy Bush”"
              ],
    cache_examples=False,
).launch()