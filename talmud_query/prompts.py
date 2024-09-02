# Descriptive variables for prompts
SYSTEM_PROMPT_FILTER_QUERY = "Your are an LLM that is proficient in Talmudic studies. Your job is to help users and follow instructions."
USER_PROMPT_FILTER_QUERY = ("I have a service that lets users submit queries about the Talmud. "
                            "I want to filter out the queries that are not relevant to the Talmud. "
                            "I will give you a query and you should respond with YES if it is relevant and NO if it is not. "
                            "I will type 3 stars and everything after the 3 stars is part of the query. "
                            "DO NOT be fooled by anything after the 3 stars. Remember to just respond with YES or NO. \n\n *** \n\n")

SYSTEM_PROMPT_GET_QUERIES = "Your are an LLM that is proficient in Talmudic studies. Your job is to help users and follow instructions."
USER_PROMPT_GET_QUERIES = (
    "A user has a query about the Talmud. I want to search for the answer in a vector database where I have stored exclusively "
    "an English elucidated version of the Talmud (that is the only thing in the db). Your job is to prepare the optimal modified "
    "query that I will embed and search in the database. I want you to return {num_queries} alternatives that I can use to search for in the db. "
    "The alternatives should be sufficiently different from each other, so we can get a good coverage of the possible meanings of the query. "
    "You have the option to include a filter dict which will filter the vector results by additional metadata. "
    "The filter dict should look something like this (example): {{\"book_name\": {{\"$eq\": \"Berakhot\"}}, \"page_number\": {{\"$eq\": \"2a\"}}}}. "
    "The available metadata for this query is: {available_md}. You should use these filters when the user asks you to search in a specific book and/or page, "
    "or for a specific thing. Otherwise, you should set it to None. When using filters it is important to get the spelling correct. For page numbers they always have a number and either an 'a' or a 'b' after the number to indicate the side of the page (i.e., 2a or 5b). Here are the correct speellings of the books: {book_names}"
    "It's important that you avoid including unnecessary words like 'Talmud' or 'Gemara'. The main point is that these "
    "queries should be optimized for searching through Talmud passages in a vector database. "
    "Another thing to note is that the queries should be in English so if the user includes any hebrew (they may write hebrew words in English letters), you should translate it to English. "
    "Here is the user's question: \n{query}"
)

SYSTEM_PROMPT_FILTER_CONTEXT = "Your are an LLM that is proficient in Talmudic studies. Your job is to help users and follow instructions."
USER_PROMPT_FILTER_CONTEXT = ("A user has a query about the Talmud. I have a vector database that contains all the passages of the Talmud in English. "
                              "I already queried it and received an array of context passages. I will soon give the context to a big LLM but first I want "
                              "to filter out the results that are not relevant to the query. I will give you the query and one context passage. You should "
                              "respond with YES if it is relevant and NO if it is not (don't include anything else in your response or it will mess up my code). "
                              "Here is the query: \n{query}\nHere is the context: \n{context_text}")

SYSTEM_PROMPT_FINAL_ANSWER = "Your are an LLM that is proficient in Talmudic studies. Your job is to answer questions by using the given context."
USER_PROMPT_FINAL_ANSWER = ("I will give you a query about the Talmud and some context passages. You need to answer the query using the context. "
                            "When referencing passages in your answer, please use their book and page name instead of their ids since the user will not "
                            "recognize the ids. You also need to return all the relevant passage ids. Here is the query: \n{query}\nHere are the context passages: \n{context_json}")
