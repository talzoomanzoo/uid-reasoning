def get_gpqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"What is the energy range of pp III neutrinos?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up details about pp III neutrinos.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_math_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do you compute the integral of e^(x^2) dx?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up techniques for integrating e^(x^2).\n\n"
        "Assistant:\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_code_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Find the minimum number of vertices in a Steiner tree that includes all specified vertices in a given tree.\"\n"
        "Assistant thinking steps:\n"
        "- I need to understand what a Steiner tree is and how to compute the minimum number of vertices required to include all specified vertices in a given tree.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Minimum Steiner Tree problem in trees<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_medical_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. If feel like you are uncertain of your answer, you are encouraged to perform a web search to help you answer the question. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Find the most effective treatments for drug-resistant tuberculosis.\"\n"
        "Assistant thinking steps:\n"
        "- I need to understand what drug-resistant tuberculosis (TB) is and what treatments are effective for it.\n\n"
        "- I should find the latest guidelines or studies on drug-resistant TB management.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Latest treatment guidelines for drug-resistant tuberculosis<|end_search_query|>\n\n"
        "(System returns processed information from relevant medical sources,ssuch as WHO guidelines or recent research articles)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


def get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

    
def get_singleqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention several URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/first_nobel_physics.html<|end_url|>\n\n"
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_multiqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>voice actor of Lara Croft<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results provide names of voice actors for Lara Croft. I need to confirm if Alice David is one of them.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention the company that developed the video game featuring Alice David as Lara Croft.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/lara_croft_voice_actor.html, http://example.com/game_developer.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_gpqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"What is the energy range of pp III neutrinos?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up details about pp III neutrinos.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/ppIII_neutrino.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_math_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s math-related question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do you compute the integral of e^(x^2) dx?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up techniques for integrating e^(x^2).\n\n"
        "Assistant:\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/integration_e_x_squared.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_code_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s programming-related question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do I implement a binary search algorithm in Python?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up the implementation details of binary search in Python.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>binary search algorithm implementation in Python<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/python_binary_search.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_naive_rag_instruction(question, documents):
    return (
        "You are a knowledgeable assistant that uses the provided documents to answer the user's question.\n\n"
        "Question:\n"
        f"{question}\n"
        "Documents:\n"
        f"{documents}\n"
    )


def get_task_instruction_medical(question, options, model_name=None): ##should change the prompt 
    # if model_name == 'qwq': #qwq is the alias for LRM right now
    user_prompt = (
            'Analyze the medical case based on the question and options provided.'
            f'Question:\n{question}\nOptions:\nA:{options[0]}\nB:{options[1]}\nC:{options[2]}\nD:{options[3]}\n\n'
            'Which option is the correct answer to the question? Along with the option index, provide your reasoning for the answer. ALWAYS provide the option index of the final answer AT THE END in boxed format EXACTLY like this: \\boxed{OPTION_INDEX}.\n\n'
            'DO NOT add any other text or any other variations.'
        )
    # else:
    #     user_prompt = (
    #         'Please answer the following question. You should think step by step to solve it.\n\n'
    #         'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
    #         f'Question:\n{question}\n\n'
    #     )
    return user_prompt

def get_task_instruction_medical_tuned(question, options, model_name=None):
    # if model_name == 'qwq':
    user_prompt = (
        'Analyze the medical case based on the question and options provided.'
        f'Question:\n{question}\nOptions:\nA:{options[0]}\nB:{options[1]}\nC:{options[2]}\nD:{options[3]}\n\n'
        'INSTRUCTIONS:\n'
            '1.Select the option that is the correct answer to the question.\n'
            '2.Plan your reasoning process based on the 4-STEP GUIDELINE FOR EXPERT MEDICAL REASONING.\n'
            '3.Then, generate your reasoning response before </think>, always starting each reasoning step with triple hashtags(###), EXACTLY like this: ###Step1. [Cue Acquisition]...\n\t###Step2. [Hypothesis Generation]...\n\t###Step3. [Cue Interpretation with Analogical Reasoning]...\n\t###Step4. [Hypothesis Evaluation]...\n</think>\n Any response that does not follow the above process should be regenerated. \n'
            '4.ALWAYS provide the option index of the final answer AT THE END with TRIPLE BACKTICKS among A, B, C, D, EXACTLY like this: The answer is ```X```. DO NOT add any other text or variations.\n'
            '4-STEP GUIDELINE FOR EXPERT MEDICAL REASONING:\n\n###Step1. [Cue Acquisition]: Identify key information or evidence from the given context.\n###Step2. [Hypothesis Generation]: Based on the cues, propose possible explanations or solutions.\n###Step3. [Cue Interpretation with Analogical Reasoning]:\n\t###Step3.1. [Find Analagous Case] Identify a familiar case that shares structural similarities with the current problem.\n\t###Step3.2. [Compare Structural Patterns] Compare key features of the familiar case and the current problem to assess relevant patterns.\n\t###Step3.3. [Apply Insights]: Apply lessons from the familiar case to interpret the gathered information and refine potential explanations.\n\t###Step4. [Hypothesis Evaluation] Compare hypotheses against the interpreted cues to determine the most valid conclusion.\n'
        )
    # else:
    #     user_prompt = (
    #         'Please answer the following question. You should think step by step to solve it.\n\n   '
    #         'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
    #         f'Question:\n{question}\n\n'
    #     )
    return user_prompt
    

def get_task_instruction_openqa(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_math(question, model_name=None):
    # if model_name == 'qwq':
    #     user_prompt = (
    #         'Please answer the following math question. '
    #         'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
    #         f'Question:\n{question}\n\n'
    #     )
    # else:
    #     user_prompt = (
    #         'Please answer the following math question. You should think step by step to solve it.\n\n'
    #         'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
    #         f'Question:\n{question}\n\n'
    #     )
    user_prompt = (
            'Please answer the following math question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_multi_choice(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following multiple-choice question. '
            'Your choice should ONLY be one of the letters A, B, C, or D. DO NOT include any answer content.\n\n'
            'Always should provide your choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'llama':
        user_prompt = (
            'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
            'Your choice should be one of the letters A, B, C, or D, DO NOT include any answer content.\n\n'
            'Provide your choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following multiple-choice question.\n\n'
            'Your choice should be one of the letters A, B, C, or D. DO NOT include any answer content.\n\n'
            'Always provide your choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_code(question, question_title=None, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Generate a correct Python program that passes all tests for the given problem. '
            'You should provide your final code within a Python code block using triple backticks (```python\n'
            'YOUR_CODE\n'
            '```).\n\n'
            f'Problem Title: {question_title}\n\n'
            f'Problem Statement:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. '
            f'You should think step by step to solve it.\n\nQuestion:\n{question}\n\n'
            'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n\n'
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return user_prompt


def get_step_splitting_instruction(problem, solution, model_name=None):
    if model_name == "Qwen/Qwen2.5-0.5B":
        user_prompt = (
            'You will be given a solution to a problem. Your task is to improve its readability by reformatting it into well-structured reasoning steps **without altering any of the original content**. Follow these guidelines:\n\n'
            '1. **Do NOT alter the content**: Preserve the original wording, mathematical expressions, and logic. Especially, don\'t try to solve the problem, FOCUS ON REFORMATTING IT!\n\n'
            '2. **Do NOT add additional information**: Do not add additional information such as explanations or revision of the given solution.\n\n'
            '3. **Do NOT remove existing information**: Do not get rid of parts of the given solution regardless of how useful it is.\n\n'
            '4. **Insert special characters to split into reasoning steps**: Use `[SPLIT]` to separate distinct reasoning steps and `[END]` to indicate that you are done. Examples of reasoning steps include:\n'
            '   - Introductory analysis or setup.\n'
            '   - Making assumptions.\n'
            '   - Case discussions.\n'
            '   - Checking whether the logic mentioned in the previous reasoning step is valid.\n'
            '   - Formula derivations or simplifications.\n'
            '   - Conclusions or final results.\n\n'
            '5. The reformated solution would look like:\n'
            '```\n'
            '**(reasoning step 1)**[SPLIT]**(reasoning step 2)**[SPLIT]**(reasoning step 3)**[SPLIT] ... [SPLIT]**(reasoning step N)**[END]\n'
            '```\n\n'
            '6. **Avoid overly granular reasoning steps**: Ensure that each reasoning step contains enough information to be meaningful and self-contained. Avoid splitting the solution into excessively small or trivial steps, as this can make the reasoning fragmented and harder to follow. Instead, aim to group related ideas, calculations, or explanations into cohesive blocks that provide sufficient context and logical progression.\n'
            '    - Mathematical expressions: Keep mathematical expressions, derivations, or expansions within the same reasoning step as the surrounding text to maintain cohesion. For example, if expanding an equation to solve for a variable, include the entire expansion process in a single reasoning step rather than breaking it into multiple steps.\n'
            '    - Logical flow: Group steps that are part of the same logical flow or task. For instance, if solving a problem involves multiple related calculations or transformations, combine them into a single reasoning step rather than separating each minor operation.\n'
            '    - Sweet spot: Strive for a balance where reasoning steps are neither too long (overwhelming) nor too short (trivial). Each step should be substantial enough to stand on its own while contributing to the overall solution.\n'
            '    - Meaningful chunks: Focus on creating reasoning steps that represent distinct, meaningful phases of the solution. For example, in a multi-step problem, each reasoning step could correspond to a major task (e.g., setting up the problem, performing calculations, interpreting results).\n\n'
            '7. **Use key phrases to split into reasoning steps**: If the solution includes certain phrases such as "First", "Second", "Lastly", "(A)", "(B)", "1)", "2)", "Next", "Finally", "Since", and "If", try to split the solution into separate reasoning steps using these phrases as identifiers. For example, you could insert "[SPLIT]" before those keyphrases. However, try not to rely on "\\n" as an identifier when splitting into reasoning steps.\n\n'
            '8. **Focus on readability**: Ensure the reformatted solution is easy to follow while retaining the original structure and content.\n\n'
            '9. **Segment code**: When a code solution is provided, segment it into meaningful, cohesive blocks based on logical functionality or purpose. Avoid splitting the code line by line unless absolutely necessary. Instead, group related lines of code into distinct units that represent specific tasks, phases, or logical steps.\n'
            '    - Function definitions: If the code defines a function, separate the function definition (e.g., signature and docstring) from its implementation or usage.\n'
            '    - Logical phases: If the code involves multiple logical phases (e.g., data preprocessing, model training, and evaluation), segment the code into blocks corresponding to each phase.\n'
            '    - Specific tasks: If a block of code performs a specific task (e.g., a loop, conditional block, or data transformation), treat it as a single reasoning step.\n'
            '    - Comments: Comments should NEVER be treated as separate reasoning steps. Always include comments with their corresponding code to maintain context and clarity.\n'
            '    - Natural grouping: Focus on grouping the code into natural, logical segments that improve readability and understanding, without considering the executability of the code.\n\n'
            '--------------------------------------------------\n\n'
            'Start writing after the "[Reformatted Solution]" identifier without generating any opening, closing, and explanations. Do NOT rewrite or modify the original content. Don\'t repeat the "[Reformatted Solution]" identifier as well.\n\n'
            'Here is the problem, and the solution that needs to be reformatted:\n\n'
            f'Problem: {problem}\n\n'
            f'Solution: {solution}\n\n'
        )
    else:
        user_prompt = (
            'You will be given a solution to a problem. Your task is to improve its readability by reformatting it into well-structured reasoning steps **without altering any of the original content**. Follow these guidelines:\n\n'
            '1. **Do NOT alter the content**: Preserve the original wording, mathematical expressions, and logic. Especially, don\'t try to solve the problem, FOCUS ON REFORMATTING IT!\n\n'
            '2. **Do NOT add additional information**: Do not add additional information such as explanations or revision of the given solution.\n\n'
            '3. **Do NOT remove existing information**: Do not get rid of parts of the given solution regardless of how useful it is.\n\n'
            '4. **Insert special characters to split into reasoning steps**: Use `[SPLIT]` to separate distinct reasoning steps and `[END]` to indicate that you are done. Examples of reasoning steps include:\n'
            '   - Introductory analysis or setup.\n'
            '   - Making assumptions.\n'
            '   - Case discussions.\n'
            '   - Checking whether the logic mentioned in the previous reasoning step is valid.\n'
            '   - Formula derivations or simplifications.\n'
            '   - Conclusions or final results.\n\n'
            '5. The reformated solution would look like:\n'
            '```\n'
            '**(reasoning step 1)**[SPLIT]**(reasoning step 2)**[SPLIT]**(reasoning step 3)**[SPLIT] ... [SPLIT]**(reasoning step N)**[END]\n'
            '```\n\n'
            '6. **Avoid overly granular reasoning steps**: Ensure that each reasoning step contains enough information to be meaningful and self-contained. Avoid splitting the solution into excessively small or trivial steps, as this can make the reasoning fragmented and harder to follow. Instead, aim to group related ideas, calculations, or explanations into cohesive blocks that provide sufficient context and logical progression.\n'
            '    - Mathematical expressions: Keep mathematical expressions, derivations, or expansions within the same reasoning step as the surrounding text to maintain cohesion.\n'
            '    - Logical flow: Group steps that are part of the same logical flow or task.\n'
            '    - Sweet spot: Strive for a balance where reasoning steps are neither too long nor too short.\n'
            '    - Meaningful chunks: Focus on creating reasoning steps that represent distinct, meaningful phases of the solution.\n\n'
            '7. **Use key phrases to split into reasoning steps** where applicable.\n\n'
            '8. **Focus on readability** while retaining the original structure and content.\n\n'
            '9. **Segment code** into meaningful blocks if the solution is code.\n\n'
            '--------------------------------------------------\n\n'
            'Start writing after the "[Reformatted Solution]" identifier without generating any opening, closing, and explanations.\n\n'
            f'Problem: {problem}\n\n'
            f'Solution: {solution}\n\n'
        )
    return user_prompt