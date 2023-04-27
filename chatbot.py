import numpy as np
import argparse
import joblib
import re  
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import nltk 
from collections import defaultdict, Counter
from typing import List, Dict, Union, Tuple

import util

class Chatbot:
    """Class that implements the chatbot for HW 6."""

    def __init__(self):
        # The chatbot's default name is `moviebot`.
        self.name = 'MOOviebot' # TODO: Give your chatbot a new name.

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, self.ratings = util.load_ratings('data/ratings.txt')
        
        # Load sentiment words 
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        # Train the classifier
        self.train_logreg_sentiment_classifier()
        
        #### possible ways of holding onto user input: ####
        self.data_points = defaultdict(int)
        self.num_data_points = 0
        self.current_title = []
        self.current_sentiment = 0
        self.num_title_clarifications = 0
        self.num_sentiment_clarifications = 0

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        
        return """
        Hi! I'm MOOviebot.
        I'm going to recommend a movie to you. 
        First I will ask you about your taste in movies. 
        Tell me about a movie that you have seen. 
        To exit: write ":quit" (or press Ctrl-C to force the exit).
        """
    

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "Hey, how can I help you?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Have a nice day :)"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    def debug(self, line):
        """
        Returns debug information as a string for the line string from the REPL

        No need to modify this function. 
        """
        return str(line)

    ############################################################################
    # 2. Extracting and transforming                                           #
    ############################################################################

    def initialize_variables(self):
        self.num_data_points = 0
        self.current_title = []
        self.current_sentiment = 0
        self.num_title_clarifications = 0
        self.num_sentiment_clarifications = 0
        
    def get_num_data_points(self):
        return len(self.data_points)
    
    def get_indices_from_line(self, line: str) -> list:
        titles = self.extract_titles(line)
        if len(titles) == 0:
            return []
        else:
            return self.find_movies_idx_by_title(titles[0])
        
    def get_titles_from_indices(self, indices: list) -> list:
        titles = []
        for index in indices:
            titles.append(self.titles[index][0])
        return titles
    
    def get_title_clarification(self) -> str:
        self.num_title_clarifications += 1
        response = "[title unclear] We have a couple titles matching that movie in our database. Which one did you see?\n"
        for title in self.get_titles_from_indices(self.current_title):
            response += str(title) + "\n"
        
        return response
    
    def get_wrapup_response(self) -> str:
        response = ""
        if self.get_num_data_points() >= 5:
            # make a movie recommendation
            recommendations = self.recommend_movies(self.data_points, 5)
            response = "[sentiment and title confirmed and 5 data points] We recommend: \n" + str(recommendations)
        else:
            response = "[sentiment and title confirmed] Thank you. Tell me about another movie you have seen."
        return response
    
    def get_response_given_one_title(self) -> str:
        response = ""
        if self.current_sentiment != 0:
            # update data point dictionary
            self.data_points[self.current_title[0]] = self.current_sentiment
                    
            # reinitialize current variables
            self.initialize_variables()
            # check data points -- do we have 5?
            response = self.get_wrapup_response()
                     
        else: # current sentiment = 0 , i.e., unclear, we ask for sentiment.
            self.num_sentiment_clarifications += 1
            response = "[title confirmed and sentiment unclear] Tell me more about what you thought of that movie."
        
        return response
    
    def process(self, line: str) -> str:
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this script.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'
        
        Arguments: 
            - line (str): a user-supplied line of text
        
        Returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        
        response = ""
        
        # NO TITLES GIVEN YET
        if len(self.current_title) == 0:
            # extract sentiment from input
            self.current_sentiment = self.predict_sentiment_rule_based(line)
            
            # extract title from input
            possible_title_indices = self.get_indices_from_line(line)
            self.current_title = possible_title_indices
            
            # Multiple possible titles
            if len(self.current_title) > 1:
                response = self.get_title_clarification()
            
            # Only 1 possible title :)
            elif len(self.current_title) == 1:
                response = self.get_response_given_one_title()
                
            # No possible title. Ask again.
            else:
                response = "[sentiment and title unclear] I'm sorry, I'm just a frame-based chatbot and I don't understand. Can you tell me about an actual movie you've seen? Please put the movie title in quotation marks."
        
        # MORE THAN 1 CANDIDATES FOR TITLE
        elif len(self.current_title) > 1:
            narrowed_titles = self.disambiguate_candidates(line.strip(), self.current_title)
            
            if len(narrowed_titles) == 1:
                self.current_title = narrowed_titles      
                response = self.get_response_given_one_title()
                
            elif len(narrowed_titles) > 1:
                self.current_title = narrowed_titles
                response = self.get_title_clarification()
                
            else: 
                response = self.get_title_clarification()
                
        # WE HAVE EXACTLY 1 CANDIDATE TITLE
        else:
            if self.current_sentiment == 0:
                self.current_sentiment = self.predict_sentiment_rule_based(line)
            response = self.get_response_given_one_title()
            
        # ----- DONE WITH CASES, BEFORE WE RETURN ----- #
        # Check that we have not exceeded either 5 title or 5 sentiment clarifications
        if (self.num_title_clarifications > 5) or (self.num_sentiment_clarifications > 5):
            response = "[exceeded num clarifications] Hm. I can't seem to figure out what exactly to make of your response. Would you mind telling me about a different movie you've seen?" 
            self.initialize_variables()
        
        return response
    
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def extract_titles(self, user_input: str) -> list:
        """Extract potential movie titles from the user input.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example 1:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I do not like any movies'))
          print(potential_titles) // prints []

        Example 2:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        Example 3: 
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'There are "Two" different "Movies" here'))
          print(potential_titles) // prints ["Two", "Movies"]                              
    
        Arguments:     
            - user_input (str) : a user-supplied line of text

        Returns: 
            - (list) movie titles that are potentially in the text

        Hints: 
            - What regular expressions would be helpful here? 
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                   
        pattern = r"\"(.*?)\""
        
        match = re.findall(pattern, user_input)
        
        if len(match) == 0:
            return [user_input]
            
        return match
            

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def remove_non_alphanumeric_tokens(self, input:str) -> list:
        tokens = self.tokenizer(input)
        cleaned = []
        for token in tokens:
            alphanumerics = ''.join(c for c in token if c.isalnum())
            if alphanumerics != '':
                cleaned.append(alphanumerics)
        
        return cleaned
        
    def find_movies_idx_by_title(self, title:str) -> list:
        """ Given a movie title, return a list of indices of matching movies
        The indices correspond to those in data/movies.txt.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list that contains the index of that matching movie.

        Example 1:
          ids = chatbot.find_movies_idx_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        Example 2:
          ids = chatbot.find_movies_idx_by_title('Twelve Monkeys')
          print(ids) // prints [31]

        Arguments:
            - title (str): the movie title 

        Returns: 
            - a list of indices of matching movies

        Hints: 
            - You should use self.titles somewhere in this function.
              It might be helpful to explore self.titles in scratch.ipynb
            - You might find one or more of the following helpful: 
              re.search, re.findall, re.match, re.escape, re.compile
            - Our solution only takes about 7 lines. If you're using much more than that try to think 
              of a more concise approach 
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                   
        indices = []
        
        for i, movie in enumerate(self.titles):
            full_title = movie[0]
            
            # use lower() so that matching is case-insensitive
            if title.lower() in full_title.lower():
                indices.append(i)
            else:
                total_matches = 0
                tokenized_input = self.remove_non_alphanumeric_tokens(title.lower())
                tokenized_potential_title = self.remove_non_alphanumeric_tokens(full_title.lower())
                
                for word in tokenized_input:
                    if word in tokenized_potential_title:
                        total_matches += 1
                
                if (total_matches/len(tokenized_potential_title)) >= 0.6:
                    indices.append(i)
        
        return indices
        
        
#         regex = r"(\d*)%.*?(?:" + title + r").*?%"
#         print(regex)
#         indices = re.findall(regex, movie_file, flags=re.IGNORECASE)
#         return indices

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    def disambiguate_candidates(self, clarification:str, candidates:list) -> list: 
        """Given a list of candidate movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (e.g. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)


        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If the clarification does not uniquely identify one of the movies, this 
        should return multiple elements in the list which the clarification could 
        be referring to. 

        Example 1 :
          chatbot.disambiguate_candidates("1997", [1359, 2716]) // should return [1359]

          Used in the middle of this sample dialogue 
              moviebot> 'Tell me one movie you liked.'
              user> '"Titanic"''
              moviebot> 'Which movie did you mean:  "Titanic (1997)" or "Titanic (1953)"?'
              user> "1997"
              movieboth> 'Ok. You meant "Titanic (1997)"'

        Example 2 :
          chatbot.disambiguate_candidates("1994", [274, 275, 276]) // should return [274, 276]

          Used in the middle of this sample dialogue
              moviebot> 'Tell me one movie you liked.'
              user> '"Three Colors"''
              moviebot> 'Which movie did you mean:  "Three Colors: Red (Trois couleurs: Rouge) (1994)"
                 or "Three Colors: Blue (Trois couleurs: Bleu) (1993)" 
                 or "Three Colors: White (Trzy kolory: Bialy) (1994)"?'
              user> "1994"
              movieboth> 'I'm sorry, I still don't understand.
                            Did you mean "Three Colors: Red (Trois couleurs: Rouge) (1994)" or
                            "Three Colors: White (Trzy kolory: Bialy) (1994)" '
    
        Arguments: 
            - clarification (str): user input intended to disambiguate between the given movies
            - candidates (list) : a list of movie indices

        Returns: 
            - a list of indices corresponding to the movies identified by the clarification

        Hints: 
            - You should use self.titles somewhere in this function 
            - You might find one or more of the following helpful: 
              re.search, re.findall, re.match, re.escape, re.compile
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                   
        possible_movies = []
        
        for index in candidates:
            if clarification.lower() in self.titles[index][0].lower():
                possible_movies.append(index)
        
        return possible_movies
        
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    ############################################################################
    # 3. Sentiment                                                             #
    ########################################################################### 

    def predict_sentiment_rule_based(self, user_input: str) -> int:
        """Predict the sentiment class given a user_input

        In this function you will use a simple rule-based approach to 
        predict sentiment. 

        Use the sentiment words from data/sentiment.txt which we have already loaded for you in self.sentiment. 
        Then count the number of tokens that are in the positive sentiment category (pos_tok_count) 
        and negative sentiment category (neg_tok_count)

        This function should return 
        -1 (negative sentiment): if neg_tok_count > pos_tok_count
        0 (neural): if neg_tok_count is equal to pos_tok_count
        +1 (postive sentiment): if neg_tok_count < pos_tok_count

        Example:
          sentiment = chatbot.predict_sentiment_rule_based('I LOVE "The Titanic"'))
          print(sentiment) // prints 1
        
        Arguments: 
            - user_input (str) : a user-supplied line of text
        Returns: 
            - (int) a numerical value (-1, 0 or 1) for the sentiment of the text

        Hints: 
            - Take a look at self.sentiment (e.g. in scratch.ipynb)
            - Remember we want the count of *tokens* not *types*
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                   
        sentiment = 0
        
        for word in self.tokenizer(user_input):
            if word.lower() in self.sentiment.keys():
                if self.sentiment[word.lower()] == "pos":
                    sentiment += 1
                else:
                    sentiment -= 1
        
        if sentiment > 0:
            return 1
        elif sentiment < 0:
            return -1
        else:
            return 0
        
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
    
    def tokenizer(self, text: str) -> List[str]:
        '''
        Citation: This tokenizer function & regex rule is borrowed from Katie's tokenizer regex demo at:
        https://www.cs.williams.edu/~kkeith/teaching/s23/cs375/attach/tokenization_regex_demo.html

        This helper function takes a string and returns a list of tokenized strings.
        '''
        regex = r"[A-Za-z]+|\$[\d\.]+|\S+" 
        return nltk.regexp_tokenize(text, regex)
    
    def train_logreg_sentiment_classifier(self):
        """
        Trains a bag-of-words Logistic Regression classifier on the Rotten Tomatoes dataset 

        You'll have to transform the class labels (y) such that: 
            -1 inputed into sklearn corresponds to "rotten" in the dataset 
            +1 inputed into sklearn correspond to "fresh" in the dataset 
        
        To run call on the command line: 
            python3 chatbot.py --train_logreg_sentiment

        Hints: 
            - Review how we used CountVectorizer from sklearn in this code
                https://github.com/cs375williams/hw3-logistic-regression/blob/main/util.py#L193
            - You'll want to lowercase the texts
            - Review how you used sklearn to train a logistic regression classifier for HW 5.
            - Our solution uses less than about 10 lines of code. Your solution might be a bit too complicated.
            - We achieve greater than accuracy 0.7 on the training dataset. 
        """ 
        
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################  
        
        #load training data  
        
        texts, y = util.load_rotten_tomatoes_dataset()
        y = [1 if label=='Fresh' else -1 for label in y]
        texts = [text.lower() for text in texts]
        
        # TODO consider grid search for tuning hyperparameters
        
        self.model = linear_model.LogisticRegression() #variable name that will eventually be the sklearn Logistic Regression classifier you train
        
        self.count_vectorizer = CountVectorizer(min_df=20,
                                                stop_words='english',
                                                max_features=1000)
        
        # train-dev-test split
        X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.33, random_state=42)
        X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        
        # Tokenization
        X_train = self.count_vectorizer.fit_transform(X_train).toarray()
        y_train = np.array(y_train)
        
        X_dev = self.count_vectorizer.transform(X_dev).toarray()
        y_dev = np.array(y_dev)
        
        X_test = self.count_vectorizer.transform(X_test).toarray()
        y_test = np.array(y_test)
        
        x_columns_as_words = self.count_vectorizer.get_feature_names_out()
        
        # Training
        self.model.fit(X_train, y_train)
        
        # Deving
        print("Accuracy on the dev set: ", self.model.score(X_dev, y_dev))

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    def predict_sentiment_statistical(self, user_input: str) -> int: 
        """ Uses a trained bag-of-words Logistic Regression classifier to classifier the sentiment

        In this function you'll also uses sklearn's CountVectorizer that has been 
        fit on the training data to get bag-of-words representation.

        Example 1:
            sentiment = chatbot.predict_sentiment_statistical('This is great!')
            print(sentiment) // prints 1 

        Example 2:
            sentiment = chatbot.predict_sentiment_statistical('This movie is the worst')
            print(sentiment) // prints -1

        Example 3:
            sentiment = chatbot.predict_sentiment_statistical('blah')
            print(sentiment) // prints 0

        Arguments: 
            - user_input (str) : a user-supplied line of text
        Returns: int 
            -1 if the trained classifier predicts -1 
            1 if the trained classifier predicts 1 
            0 if the input has no words in the vocabulary of CountVectorizer (a row of 0's)

        Hints: 
            - Be sure to lower-case the user input 
            - Don't forget about a case for the 0 class! 
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                             
        X = self.count_vectorizer.transform([user_input.lower()]).toarray()
        
        if (X == ([0] * len(X))).all():
            return 0
        
        y = self.model.predict(X)        
        return y
        
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    ############################################################################
    # 4. Movie Recommendation                                                  #
    ############################################################################

    def recommend_movies(self, user_ratings: dict, num_return: int = 3) -> List[str]:
        """
        This function takes user_ratings and returns a list of strings of the 
        recommended movie titles. 

        Be sure to call util.recommend() which has implemented collaborative 
        filtering for you. Collaborative filtering takes ratings from other users
        and makes a recommendation based on the small number of movies the current user has rated.  

        This function must have at least 5 ratings to make a recommendation. 

        Arguments: 
            - user_ratings (dict): 
                - keys are indices of movies 
                  (corresponding to rows in both data/movies.txt and data/ratings.txt) 
                - values are 1, 0, and -1 corresponding to positive, neutral, and 
                  negative sentiment respectively
            - num_return (optional, int): The number of movies to recommend

        Example: 
            bot_recommends = chatbot.recommend_movie({100: 1, 202: -1, 303: 1, 404:1, 505: 1})
            print(bot_recommends) // prints ['Trick or Treat (1986)', 'Dunston Checks In (1996)', 
            'Problem Child (1990)']

        Hints: 
            - You should be using self.ratings somewhere in this function 
            - It may be helpful to play around with util.recommend() in scratch.ipynb
            to make sure you know what this function is doing. 
        """ 
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                   
        # binarize user ratings and get ratings matrix
        sparse_data_array = [0] *len(self.titles)

        for key in user_ratings.keys():
            sparse_data_array[key] = user_ratings[key]
        
        return  self.get_titles_from_indices(util.recommend(sparse_data_array, self.ratings, num_return))
    
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    ############################################################################
    # 5. Open-ended                                                            #
    ############################################################################

    def function1(self, title:str) -> list:
        """
        Matches movie title even though it is not in quotation marks. Case insensitive.
        
        Option 1: Matches potential title if a certain percent of tokens in that title is 
        matched with the user input. Such percent depends on the length of the
        potential title. 

        E.g. when the title is "Chisum (1970)" (length == 2), 50% of the tokens must be matched.

        Option 5: This matching scheme is based on percentage of tokens matched in a title, so
        it also fulfills Option 5. "American in Paris, An (1951)" is of length 5, which means
        60% of its tokens, or three words, must be present in the user input. So, a user could write
        either "I liked "American in Paris"" or "I liked "An American in Paris" and the chatbot would
        recognize it.

        Arguments:
        - title: the user input
        
        Example:
        function1(self, "I like finding nemo") 
        output: [4610]
        """
        
        indices = []
        
        for i, movie in enumerate(self.titles):
            full_title = movie[0]
            
            # use lower() so that matching is case-insensitive
            if title.lower() in full_title.lower():
                indices.append(i)
            else:
                total_matches = 0
                tokenized_input = self.remove_non_alphanumeric_tokens(title.lower())
                tokenized_potential_title = self.remove_non_alphanumeric_tokens(full_title.lower())
                
                for word in tokenized_input:
                    if word in tokenized_potential_title:
                        total_matches += 1
                
                percent = 0
                
                if len(tokenized_potential_title) <= 2:
                    percent = 0.5
                elif len(tokenized_potential_title) <= 3:
                    percent = 0.6
                elif len(tokenized_potential_title) <= 4:
                    percent = 0.5
                elif len(tokenized_potential_title) <= 5:
                    percent = 0.6
                else:
                    percent = 0.5
                
                if (total_matches/len(tokenized_potential_title)) >= percent: 
                    indices.append(i)
        
        return indices

    def function2():
        """
        See function1()-- we've implemented Options 1 and 5 in a single title-matching function.
        function1() would replace our find_movies_idx_by_title() function
        """  
        pass

    def function3(): 
        """
        Any additional functions beyond two count towards extra credit  
        """
        pass 


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')



