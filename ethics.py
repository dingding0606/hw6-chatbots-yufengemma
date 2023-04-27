"""
Please answer the following ethics and reflection questions. 

We are expecting at least three complete sentences for each question for full credit. 

Each question is worth 2.5 points. 
"""

######################################################################################
"""
QUESTION 1 - Anthropomorphizing

Is there potential for users of your chatbot to possibly anthropomorphize 
(attribute human characteristics to an object) it? 
What are some possible ramifications of anthropomorphizing chatbot systems? 
Can you think of any ways that chatbot designers could ensure that users can easily 
distinguish the chatbot responses from those of a human?
"""

Q1_your_answer = """
It is unlikely that a user would anthropomorphize our chatbot. We added helpful (debugging) information
into our chatbot's responses that make it clear it is simply looking to clarify a user's sentiment
and the movie title they are dicussing.

If a chatbot too closely mimics human speech, users might give up personal information that they would
otherwise only entrust to a friend. Additionally, they might put too much weight in the recommendations
of the chatbot without critically analyzing it as they would any other algorithm or program.

Chatbot designers might include some information, as we did, which clearly reflects the chatbot's
algorithmic goals (e.g. 'extracting title, sentiment clear') so that users cannot mistake a text-matching
algorithm with conscious understanding. To that same end, designers might write responses that are very clearly not-human (e.g. 'Title:' instead of 'Omg, so what did you think of that movie????').
"""

######################################################################################

"""
QUESTION 2 - Privacy Leaks

One of the potential harms for chatbots is collecting and then subsequently leaking 
(advertly or inadvertently) private information. Does your chatbot have risk of doing so? 
Can you think of ways that designers of the chatbot can help to mitigate this risk? 
"""

Q2_your_answer = """
There is nothing stopping users from inputting highly sensitive private information. However, our chatbot
only holds two pieces of state directly related to their input: a simple sentiment rating, and a list of
potential title indices. We do not save the user input past the initial processing stage. Designers of task-specific chatbots should not hold onto any state that is unnecessary for the completion of their task.
"""

######################################################################################

"""
QUESTION 3 - Effects on Labor

Advances in dialogue systems, and the language technologies based on them, could lead to the automation of 
tasks that are currently done by paid human workers, such as responding to customer-service queries, 
translating documents or writing computer code. These could displace workers and lead to widespread unemployment. 
What do you think different stakeholders -- elected government officials, employees at technology companies, 
citizens -- should do in anticipation of these risks and/or in response to these real-world harms? 
"""

Q3_your_answer = """

We're not experts in thinking about this, but here are some thoughts:

1. For the government, it is definitely crucial to keep in mind the consequences of incorporating automated systems. For example, there could be regulations that grants a longer period of notice before laying off workers whose jobs are replaced by AI systems.

2. Job re-training for newer opportunities opened by these AI systems or opporunities in other industries.

3. Universal basic income-- Andrew Yang famously argued for this on a national scale, but smaller case studies like Seattle's implementation in the 60's demonstrate that UBI could be effective in buffering people from the scarier short-term risks of unemployment (like losing healthcare and income). 
"""

"""
QUESTION 4 - Refelection 

You just built a frame-based dialogue system using a combination of rule-based and machine learning 
approaches. Congratulations! What are the advantages and disadvantages of this paradigm 
compared to an end-to-end deep learning approach, e.g. ChatGPT? 
"""

Q4_your_answer = """
There are several advantages to using a frame-based approach, including high interpretability, less susceptibility to information leakage, less susceptibility to anthropomorphization, and the fact that it does not require any training, which saves energy. However, there are also several disadvantages to using a frame-based approach, such as its poor generalizability, its complicated modular implementation that is less maintainable and difficult to organize (e.g., the process() function), and its requirement for user input in a particular format. We prefer to fill out a web form rather than communicate with a chatbot.
"""