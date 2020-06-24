class Timer():

    '''A timer used to record how long a process takes.
    After instaniating, a .start() and .stop() can be used 
    before and after a process in respective order.'''## def init
    def __init__(self,format_="%m/%d/%y - %I:%M %p"):

        import tzlocal
        self.tz = tzlocal.get_localzone()
        self.fmt = format_
        
        self.created_at = self.get_time()# get time
         
    ## def get time method
    def get_time(self):
        import datetime as dt
        return dt.datetime.now(self.tz)

    ## def start
    def start(self):
        time = self.get_time()
        self.start = time
        print(f"[i] Timer started at{self.start.strftime(self.fmt)}")
        
        ## def stop
    def stop(self):
        time = self.get_time()
        self.end = time
        print(f"[i] Timer ended at {self.end.strftime(self.fmt)}")
        print(f"- Total time = {self.end-self.start}")
timer = Timer()
print(timer.created_at)
timer.start()
timer.stop()


def process_comment(text):
    '''A pre-processing function that cleans text of stopwords, punctuation and capitalization, tokenizes
    then finds the most frequently used 100 words

    text - the text to be cleaned in string format'''

    # Get all the stop words in the English language
    stopwords_list = stopwords.words('english')

    #remove punctuation
    stopwords_list += list(string.punctuation)
    ##adding adhoc all strings that don't appear to contribute, added 'article, page and wikipedia' iteratively as 
    ##these are parts of most comment strings
    stopwords_list += ("''","``", "'s", "\\n\\n" , '...', 'i\\','\\n',
                       '•', "i", 'the', "'m", 'i\\', "'ve", "don\\'t",
                      "'re", "\\n\\ni", "it\\", "'ll", 'you\\', "'d", "n't",
                      '’', 'article', 'page', 'wikipedia') 
    
    from nltk import word_tokenize
    tokens = word_tokenize(text)
    stopped_tokens = [w.lower() for w in tokens if w.lower() not in stopwords_list]
    freqdist = FreqDist(stopped_tokens)
    most_common_stopped = freqdist.most_common(100)
    return most_common_stopped
 
def clean_up(freq_tox): 
    '''Takes the most frequently used and highly offensive words and replaces them with 
    edited versions

    freq_tox - takes a dictionary
    '''
    
    
    ## creating a dictionary of the most offensive words
    replace = {'fuck': 'f$%!', 'nigger' : "n*###%" ,'nigga':'n#5#*', 'fucking' : 'f*@%!ng',
               'faggot':'f@&&*#', 'cunt' : 'c&#^' , 'fag' : 'f@$',
               "'fuck" : "'f$%!'", 'faggots':'f@&&*!$'}

    #using the 'replace' dictionary above, 

    new_dict = {}
    for k, v in dict(freq_tox).items():
        if k in replace:
            key = replace[k]
        else:
            key = k

        new_dict[key] = v
        
    cleaned_list = [ [k,v] for k, v in new_dict.items() ]
    
    return cleaned_list

def replace_all(lst, dic):
    '''another version of cleaning a list of words using a dictionary
    this function is more flexible than clean_up as the list and 
    dictionary can be used on the fly

    lst - a list of words separated by commas
    dictionary - dictionary with key value pairs
    '''

    new_lst = []
    for st in lst:
        for i, j in dic.items():
            st = st.replace(i, j)
            new_lst.append(st)
    return new_lst


def wrd_cld(toks):
    '''Function to visualize word frequency using tokens and is particular to findings of the 
    corpus used in the toxic words challenge from Kaggle.

    toks - tokens rendered from tokinization'''
    import string
    import nltk
    from nltk.corpus import stopwords
    from nltk import word_tokenize
    import matplotlib.pyplot as plt

    # Get all the stop words in the English language
    stopwords_list = stopwords.words('english')

    #remove punctuation
    stopwords_list += list(string.punctuation)
    ##adding adhoc all strings that don't appear to contribute, added 'article, page and wikipedia' iteratively as 
    ##these are parts of most comment strings
    stopwords_list += ("''","``", "'s", "\\n\\n" , '...', 'i\\','\\n',
                       '•', "i", 'the', "'m", 'i\\', "'ve", "don\\'t",
                      "'re", "\\n\\ni", "it\\", "'ll", 'you\\', "'d", "n't",
                      '’', 'article', 'page', 'wikipedia')
    import wordcloud
    from wordcloud import WordCloud
    wordcloud = WordCloud(stopwords=stopwords_list,collocations=False)
    wordcloud.generate(','.join(toks))
    plt.figure(figsize = (12, 12), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis('off')


def plot_acc_loss(history):
    '''plotting function to visualize performance history of deep learning that 
    compares training data with validation data - metrics are accuracy and loss.

    history - the output of a the .fit() function in predictive modeling.'''
    import matplotlib.pyplot as plt
    # %matplotlib inline 

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) +1)
    plt.plot(epochs, acc, label='Training accuracy')
    plt.plot(epochs, val_acc,color='g', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, color='g' , label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def tok_text(text):
    '''Function to tokenize text from a string

    text - text in a string format
    '''
    import string
    import nltk
    from nltk.corpus import stopwords
    from nltk import word_tokenize

    # Get all the stop words in the English language
    stopwords_list = stopwords.words('english')

    #remove punctuation
    stopwords_list += list(string.punctuation)
    ##adding adhoc all strings that don't appear to contribute, added 'article, page and wikipedia' iteratively as 
    ##these are parts of most comment strings
    stopwords_list += ("''","``", "'s", "\\n\\n" , '...', 'i\\','\\n',
                       '•', "i", 'the', "'m", 'i\\', "'ve", "don\\'t",
                      "'re", "\\n\\ni", "it\\", "'ll", 'you\\', "'d", "n't",
                      '’', 'article', 'page', 'wikipedia')
    
    tokens = word_tokenize(text)
    stopped_tokens = [w.lower() for w in tokens if w.lower() not in stopwords_list]
    return(stopped_tokens)


def freq_dist(tokens, n=100):
   '''Function to discover the most frequently used words
   in a corpus.  Default is 100.

   tokens - takes tokenized words as input

   returns a list of tuples displaying the words and associated count'''
   from nltk import FreqDist
   freqdist = FreqDist(tokens)
   most_common_stopped = freqdist.most_common(n)
   return most_common_stopped





def class_report_model(y_train,y_test, y_preds):
    '''creates a confusion matrix and classification model
    using training testing and predictions from a RNN classification model

    y_train - training data in tokenized form
    y_test - testing data in tokeinzed form
    y_preds - multinomial classification predictions in DataFrame form'''

    from sklearn.metrics import classification_report, confusion_matrix

    for i in range(0,y_train.shape[1]):
        y_i_hat_trnn = y_preds.iloc[:,i]
        y_tst = y_test.iloc[:,i]
    print(y_train.columns[i])
    print(confusion_matrix(y_tst, y_i_hat_trnn, normalize='true'))
    print()
    print(classification_report(y_tst, y_i_hat_trnn))       