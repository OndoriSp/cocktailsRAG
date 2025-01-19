from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBED_DIM = 768
INDEX_NAME = 'cocktails'
MEMORY_INDEX_NAME = 'chatmemory'

"""
    Data preparing
"""
df = pd.read_csv('final_cocktails.csv')
#texts = df['text'].tolist()
df['full'] = '\nCocktail: ' + df['name'] + '\nAlcoholic: ' + df['alcoholic'] \
+ '\nCategory: ' + df['category'] + '\nGlass Type: ' + df['glassType'] \
+ '\nInstructions: ' + df['instructions'] + '\nIngredients: ' + df['ingredients'] \
+ '\nIngredient Measures: ' + df['ingredientMeasures'] + '\nDescription: ' + df['text']
texts = df['full'].tolist()
ids = [str(i) for i in df['id'].tolist()]
metadatas = df.drop(columns=['id', 'drinkThumbnail', 'text', 'full'], axis=1).to_dict(orient='records')

"""
    PINECONE
"""
pc = Pinecone(api_key=PINECONE_API_KEY)

bm25_encoder = BM25Encoder()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

"""
    Creating Pinecone index for storing cocktails data
    Adding data to the vector DB
"""
try:
    if INDEX_NAME in pc.list_indexes().names():
        pc.delete_index(INDEX_NAME)

    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric='dotproduct',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print(f'INFO: Pinecone index "{INDEX_NAME}" has been created')

    # Connecting to the Pinecone index
    index = pc.Index(INDEX_NAME)
    print(f'INFO: Connected to the "{INDEX_NAME}" Pinecone index')
    print(index.describe_index_stats())

    bm25_encoder.fit(texts)
    bm25_encoder.dump("bm25.json")

    retriever = PineconeHybridSearchRetriever(
        index=index,
        embeddings=embeddings,
        sparse_encoder=bm25_encoder
    )   

    retriever.add_texts(
        texts=texts,
        ids=ids,
        metadatas=metadatas
    )
    print(f'INFO: Data stored successfully into "{INDEX_NAME}" Pinecone index')
    print(index.describe_index_stats())
except Exception as e:
    print(f'!!! EXCEPTION: {e}')

"""
    Creating Pinecone index for storing user messages
"""
try:
    if MEMORY_INDEX_NAME in pc.list_indexes().names():
        pc.delete_index(MEMORY_INDEX_NAME)

    pc.create_index(
        name=MEMORY_INDEX_NAME,
        dimension=EMBED_DIM,
        metric='dotproduct',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print(f'INFO: Pinecone index "{MEMORY_INDEX_NAME}" has been created')

    # Connecting to the Pinecone index
    memory_index = pc.Index(MEMORY_INDEX_NAME)
    print(f'INFO: Connected to the "{MEMORY_INDEX_NAME}" Pinecone index')
    print(memory_index.describe_index_stats())
except Exception as e:
    print(f'!!! EXCEPTION: {e}')
