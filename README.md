# Project overview
**Amount of time it took to complete the assessment:** I spent approximately 14 hours completing this project.

**Brief summary:**
- Cocktails data stored in vector DB (Pinecone).
- User preferences for cocktails/ingredients are also stored in vector DB (Pinecone).
- RAG system uses a hybrid retriever (BM25+Dense+Reranker) for existing dataset and PineconeVectorStore + SelfQueryRetriever to work with the user's chat history.
- Memory summarization is used.
- Detection of desired user messages (preferences in cocktails and ingredients) is implemented using prompt engineering. 

**Project tree:**
- ``final_cocktails.csv`` - dataset ([https://www.kaggle.com/datasets/aadyasingh55/cocktails](https://www.kaggle.com/datasets/aadyasingh55/cocktails))
- ``bot.py`` - chatbot class implementation
- ``setup.py`` - Pinecone setup and data preparation
- ``app.py`` - main file which starts web application


# Setting up
## Create virtual enviroment
```bash
python -m venv venv
```
```bash
venv\Scripts\activate
```
## Pip Install Requirements.txt
```python
pip install -r requirements.txt
```
## Declare environment variables
([Generate your Pinecone API Key](https://app.pinecone.io))
([Generate your GROQ API Key](https://console.groq.com/playground))
```bash
$env:PINECONE_API_KEY="pinecone_api_key"  
```
```bash
$env:GROQ_API_KEY="groq_api_key"  
```

*You can use my API keys, but I'll remove them from this README next week.
```bash
$env:PINECONE_API_KEY="pcsk_7NLps7_FwQUMcEUFze3wA3nKyHM1fFV3H6sH51pRvQu9E2aHiEq1698HYL89yi1chmcojL"  
```
```bash
$env:GROQ_API_KEY="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"
```
# Usage
Firstly start ``setup.py``. This file creates Pinecone indexes. After that you can use application via starting ``app.py``.


