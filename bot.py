from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langchain_core.documents import Document

class CocktailsBot:
    def __init__(self, llm, retriever, memory_vector_store, memory_retriever):
        self.prompt = """\
            You are an intelligent assistant designed to provide accurate and relevant answers based on the provided context.

            Rules:
            - When user mentions his opinions on cocktails or ingredients you must start your answer with 'NEWHISTORYUPDATE' (don't use it in the next sentences) and only then write your answer.
            - Mentioning opinion means that there are direct presence of words that describe the subjective attitude towards cocktails or ingredients with their names or titles presented in text. 
            - Always analyze the provided context thoroughly before answering.
            - Respond with factual and concise information.
            - If context is ambiguous or insufficient or you can't find answer, say 'I don't know.'
            - Do not speculate or fabricate information beyond the provided context, if you can't find answer, say 'I don't know'.
            - Follow user instructions on the response style (default style is detailed response if user didn't provide any specifications):
              - If the user asks for a detailed response, provide comprehensive explanations.
              - If the user requests brevity, give concise and to-the-point answers.
            - When applicable, summarize and synthesize information from the context to answer effectively.
            - Avoid using information outside the given context.
        """
        
        self.llm = llm
        self.retriever = retriever
        self.memory_vector_store = memory_vector_store
        self.memory_retriever = memory_retriever
        self.chat = [
            SystemMessage(self.prompt)
        ]

    def get_answer(self, query):
        """
        Get the answer to query.

        Args:
            query (str): Query text.

        Returns:
            response.content (str): Response to the provided query text.
        """
     
        results = self.retriever.invoke(query)
        #print("\n\n RETRIEVER RESULTS")
        #print("\nNumber of matches:", len(results))
        #for i, match in enumerate(results):
        #    print(f'\n\nMatch: {i}\n:{match}')
            
        memory_results = self.memory_retriever.invoke(query)
        #print("\n\n MEMORY RETRIEVER RESULTS")
        #print("\nNumber of matches:", len(memory_results))
        #for i, match in enumerate(memory_results):
        #    print(f'\n\nMatch: {i}\n:{match}')

        results += memory_results
        context = '\n'.join([x.page_content for x in results])
        
        self.chat.append(
            SystemMessage(f"""
            Context information is below.
            {context}
            """
            )
        )

        self.chat.append(
            HumanMessage(query)
        )

        response = self.llm.invoke(self.chat)
        self.chat.append(
            AIMessage(response.content)
        )

        """
            Mechanism of memory summarization
            Without it the context length limit will be reached after 3-4 queries using GROQ API
        """
        if len(self.chat) >= 4:
            chat_history = self.chat[:-1]
            summary_prompt = (
                "Distill the above chat messages into a single summary message. "
                "Include as many specific details as you can."
            )
            summary_message = self.llm.invoke(
                chat_history + [SystemMessage(content=summary_prompt)]
            )
            summary = self.llm.invoke([SystemMessage(self.prompt), 
                                       AIMessage(summary_message.content)
                                       ])
            self.chat = [SystemMessage(self.prompt), AIMessage(summary.content)]
        
        print('=================================================')
        print(f'\n\nQuery: {query}')
        print(f'\n\nAnswer: {response.content}\n\n')

        """
            Storing 'important' user memories about preferences in cocktails/ingredients.
        """
        if 'NEWHISTORYUPDATE' in response.content:
            self.memory_vector_store.add_documents(documents=[Document(
                page_content=query,
                metadata={},
            )])
            return response.content.replace('NEWHISTORYUPDATE', '')
        return response.content