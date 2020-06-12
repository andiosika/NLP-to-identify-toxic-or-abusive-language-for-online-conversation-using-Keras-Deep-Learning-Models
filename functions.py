
class Timer():
    '''A timer used to record how long a process takes.
    After instaniating, a .start() and .stop() can be used 
    before and after a process in respective order.'''




    ## def init
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
    from nltk import word_tokenize
    
    tokens = word_tokenize(text)
    stopped_tokens = [w.lower() for w in tokens if w.lower() not in stopwords_list]
    return stopped_tokens
 



def class_report_model(y_train, y_test, y_preds):
	for i in range(0,y_train.shape[1]):
    y_i_hat_trnn = y_preds.iloc[:,i]
    y_i_trnn = y_test.iloc[:,i]
    print(y_train.columns[i])
    print(classification_report(y_i_trnn, y_i_hat_trnn))