#Imports...Start here
#%%
# #######################################################
# IMPORTS
#######################################################
# NECESSARY IMPORTS
import streamlit as st
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
from nltk.corpus.reader import aligned
import pandas as pd
import numpy as np
import datetime as dt
import time

# IMPORTS FOR WEB SCRAPING
from pmaw import PushshiftAPI
api = PushshiftAPI()

# IMPORT TO EXTRACT SUBREDDITS
import json, requests, pprint

# IMPORT FOR WEB APPLICATION
import streamlit.components.v1 as components
import math
import json
import requests
import itertools

#from datetime import datetime, timedelta
# IMPORTS FOR DATA ANALYTICS
import pandas as pd
#pd.set_option("display.max_colwidth", 200)
import numpy as np

# IMPORTS FOR NATURAL LANGUAGE PROCESSING(NLP)
import re
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
nlp = spacy.load("en_core_web_lg")
from spacy.lang.en.stop_words import STOP_WORDS
##nltk.download('stopwords') # run this one time
from nltk.corpus import stopwords

#import neuralcoref

# IMPORTS FOR TOPIC MODELLING
import gensim
from gensim import corpora
from gensim.test.utils import common_texts
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import datetime as dt
# IMPORTS FOR DATA VISUALIZATION
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
from wordcloud import WordCloud
from PIL import Image
from textblob import TextBlob
import pyLDAvis
import pyLDAvis.gensim
#%matplotlib inline+

#IMPORTS FOR SENTIMENT ANALYTICS
from vaderSentiment import vaderSentiment
#from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans

# IMPORTS FOR KNOWLEDGE GRAPH
import bs4
import requests
from spacy import displacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 
import networkx as nx
from tqdm import tqdm
from pyvis.network import Network

from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
#Imports...ends here



#Global variables and config...starts here
sent_analyser = SentimentIntensityAnalyzer()

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use','a','about', 'above', 'across','take','give','be','say','would','want','come'])
st1= ['also','although','always','am','among','amongst','amoungst','amount','an','and','another','any','anyhow','anyone','anything','anyway','anywhere','are','around','as','at','back','be','became','because','become','becomes','becoming','been','be','before','beforehand','behind','being','below','beside','besides','between','beyond','bill','both','bottom','but','by','call','can','cannot','cant','co','con','could','couldnt','cry','de','describe','detail','do','done','down','due','during','each','eg','eight','either','eleven','else','elsewhere','empty','enough','etc','even','ever','every','everyone','everything','everywhere','except','few','fifteen','fifty','fill','find','fire','first','five','for','former','formerly','forty','found','four','from','front','full','further','get','give','go','had','has','hasnt','have','he','hence','her','here','hereafter','hereby','herein','hereupon','hers','herself','him','himself','his','how','however','hundred','i','ie','if','in','inc','indeed','interest','into','is','it','its','itself','keep','know','last','latter','latterly','least','less','ltd','made','many','may','me','meanwhile','might','mill','mine','more','moreover','most','mostly','move','much','must','my','myself','name','namely','neither','never','nevertheless','next','nine','no','nobody','none','noone','nor','not','nothing','now','nowhere','of','off','often','on','once','one','only','onto','or','other','others','otherwise','our','ours','ourselves','out','over','own','part','per','perhaps','please','put','rather','re','same','see','seem','seemed','seeming','seems','serious','several','she','should','show','side','since','sincere','six','sixty','so','some','somehow','someone','something','sometime','sometimes','somewhere','still','such','system','take','ten','than','that','the','their','them','themselves','then','thence','there','thereafter','thereby','therefore','therein','thereupon','these','they','thick','thin','think','third','this','those','though','three','through','throughout','thru','thus','to','together','too','top','toward','towards','twelve','twenty','two','un','under','until','up','upon','us','very','via','was','we','well','were','what','whatever','when','whence','whenever','where','whereafter','whereas','whereby','wherein','whereupon','wherever','whether','which','while','whither','who','whoever','whole','whom','whose','why','will','with','within','without','would','yet','you','your','yours','yourself','yourselves']
stop_words.extend(st1)

st.title("""
SOCIAL ANALYTICS APP
Social media monitoring and analytics help you find relevant healthcare conversations on various social media handles.
""")
today = dt.date.today()
tomorrow = today + dt.timedelta(days=1)


#Global variables and config...ends here


# Session states...start here
if 'has_initiated' not in st.session_state:
    st.session_state['has_initiated'] = False
    print(f"Session state is {st.session_state['has_initiated'] }")

if 'has_dataProcessed' not in st.session_state:
    st.session_state['has_dataProcessed'] = False
    print(f"Session state is {st.session_state['has_dataProcessed'] }")

if "df_KGCleaned" not in st.session_state:
    st.session_state['df_KGCleaned'] = pd.DataFrame()
    print(f"df_KGCleaned state is {st.session_state['df_KGCleaned'] }")

if "scraped_data" not in st.session_state:
    st.session_state['scraped_data'] = pd.DataFrame()
    print(f"scraped_data state is {st.session_state['scraped_data'] }")


if "fig" not in st.session_state:
    st.session_state['fig'] = None
    print(f"fig state is {st.session_state['fig'] }")
    
if "fig1" not in st.session_state:
    st.session_state['fig1'] = None
    print(f"fig1 state is {st.session_state['fig1'] }")
    
if "fig2" not in st.session_state:
    st.session_state['fig2'] = None
    print(f"fig2 state is {st.session_state['fig2'] }")

if "scatter_plot" not in st.session_state:
    st.session_state['scatter_plot'] = None
    print(f"scatter_plot state is {st.session_state['scatter_plot'] }")


if "html_string" not in st.session_state:
    st.session_state['html_string'] = None
    print(f"html_string state is {st.session_state['html_string'] }")


if "perplexity_str" not in st.session_state:
    st.session_state['perplexity_str'] = None
    print(f"perplexity_str state is {st.session_state['perplexity_str'] }")


if "df_lemmatized" not in st.session_state:
    st.session_state['df_lemmatized'] = pd.DataFrame()
    print(f"df_lemmatized state is {st.session_state['df_lemmatized'] }")


if "wordcloud" not in st.session_state:
    st.session_state['wordcloud'] = None
    print(f"wordcloud state is {st.session_state['wordcloud'] }")

#Sessions states...ends here

    
#%%
#######################################################
## SECTION 2 : DATA GATHERING (REDDIT WEB SCRAPING)
#######################################################


# Form ...starts here
with st.form('Form1'):
        #q = st.text_input("Enter search keyword", "...")
        subreddit = st.text_input("Enter Subreddit Topic", "...")
        limit = st.slider(label='Select intensity', min_value=0, max_value=10000, key=4)
        st.write("""
        #### Select Date range
        """)
        after = st.date_input('From date', today)
        before = st.date_input('To date', tomorrow)       
        
        if before > after:
            st.success('Subreddit `%s`\n\nMax Value `%s`\n\nStart date: `%s`\n\nEnd date:`%s. `' % (subreddit, limit, before, after))    
        else:
            st.error('Error: End date must fall after start date.')
        submitted1 = st.form_submit_button('Submit')

if submitted1:
    if st.session_state['has_initiated'] == False:
        before = before.strftime('%d-%m-%Y')
        after = after.strftime('%d-%m-%Y')
        url = f'http://127.0.0.1:5000/fetch/{subreddit}/{limit}/{before}/{after}'
        print("calling" + url)
        r = requests.get(url)
        j = r.json()
        st.session_state['scraped_data'] = pd.DataFrame.from_dict(j)
        st.session_state['has_initiated'] = True  

if st.button("Reset"):
    st.session_state['has_initiated'] = False
    st.session_state['has_dataProcessed'] = False
    st.session_state['df_KGCleaned'] = pd.DataFrame()
    st.session_state['scraped_data'] = pd.DataFrame()
    st.session_state['fig'] = None
    st.session_state['fig1'] = None
    st.session_state['fig2'] = None
    st.session_state['scatter_plot'] = None
    st.session_state['html_string'] = None
    st.session_state['perplexity_str'] = None
    st.session_state['df_lemmatized'] = pd.DataFrame()
    st.session_state['wordcloud'] = None
#Form ...ends here



#Functions...start here
def freq_words(x, terms = 100):
    print(f"x IS {x}")
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    
    #print(f"all_words IS {all_words}")
    print(f"all_words type is {type(all_words)}")
    
    fdist = FreqDist(all_words)
    
    #print(f"fdist IS {fdist}")
    print(f"fdist type is {type(fdist)}")
    
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
    # selecting top 100 most frequent words
    d = words_df.nlargest(columns="count", n = terms)
    data = [go.Bar(
            x = d['word'],
            y = d['count'],
            marker= dict(colorscale='Jet',
                        color = d['count'].values),
            #text='Word counts'
    )]
    layout = go.Layout(height=700)
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(yaxis_title="Word Count")
    st.subheader('WORD FREQUENCY BAR GRAPH')
    st.plotly_chart(fig, use_container_width=True)
            #py.iplot(fig, filename='Word frequency Bar plot')


def get_lemmatized_text(corpus):
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]



def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def sentiment(text):
    return (sent_analyser.polarity_scores(text)["compound"])


# function to calculate subjectivity
def getSubjectivity(review):
    try:
        return TextBlob(review).sentiment.subjectivity
    except:
        return None
    # function to calculate polarity
def getPolarity(review):
    try:
        return TextBlob(review).sentiment.polarity
    except:
        return None

# function to analyze the reviews
def analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


def get_entities(sent):
        ## chunk 1
            ent1 = ""
            ent2 = ""

            prv_tok_dep = ""    # dependency tag of previous token in the sentence
            prv_tok_text = ""   # previous token in the sentence

            prefix = ""
            modifier = ""

            #############################################################
        
            for tok in nlp(sent):
            ## chunk 2
            # if token is a punctuation mark then move on to the next token
                if tok.dep_ != "punct":
            # check: token is a compound word or not
                    if tok.dep_ == "compound":
                        prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                        if prv_tok_dep == "compound":
                            prefix = prv_tok_text + " "+ tok.text
            
            # check: token is a modifier or not
                    if tok.dep_.endswith("mod") == True:
                        modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                        if prv_tok_dep == "compound":
                            modifier = prv_tok_text + " "+ tok.text
            
            ## chunk 3
                    if tok.dep_.find("subj") == True:
                        ent1 = modifier +" "+ prefix + " "+ tok.text
                        prefix = ""
                        modifier = ""
                        prv_tok_dep = ""
                        prv_tok_text = ""      

            ## chunk 4
                    if tok.dep_.find("obj") == True:
                        ent2 = modifier +" "+ prefix +" "+ tok.text
                
            ## chunk 5  
            # update variables
                    prv_tok_dep = tok.dep_
                    prv_tok_text = tok.text
        #############################################################

            return [ent1.strip(), ent2.strip()]


def get_relation(sent):
    doc = nlp(sent)

    #Matcher class object 
    matcher = Matcher(nlp.vocab)

    #define the pattern 
    pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},
            {'POS':'ADJ','OP':"?"}] 

    #matcher.add("matching_1", None, pattern)
    matcher.add("matching_1", [pattern], on_match=None)

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]] 
    return(span.text)


#Functions...end here





scraped_data = st.session_state['scraped_data']
if isinstance(scraped_data, pd.DataFrame) and not scraped_data.empty and st.session_state['has_dataProcessed'] == False:
    
    #######################################################
    scraped_data.head()
    scraped_data.info()
    #%%
    #######################################################
    ## SECTION 3 : DATA PRE-PROCESSING
    ## DATA WRANGLING
    #######################################################
    # DROP NA
    if not scraped_data.empty:
        
        df = scraped_data.dropna(subset=['body'])
        print('Is null : ',df['body'].isnull().sum())
        # DROP DUPLICATES
        df = df.drop_duplicates(subset = ['body'])
        print('Is unique : ',df['body'].is_unique)
        #DROP UNNECESSARY COLUMNS
        df = df[df.body != "[removed]"]
        #CONVERT UTC DATE TO DATE TIMESTAMP
        df['creation_date'] = pd.to_datetime(df['created_utc'], dayfirst=True, unit='s')
        #df = df.filter(['author', 'body','permalink','score','subreddit','creation_date'])
        df.head()
        #######################################################
        ## SECTION 4 : TEXT PRE-PROCESSING
        ## NATURAL LANGUAGE PROCESSING(NLP)
        #######################################################
        # Lets plot to see the most frequently used terms in the subreddit comments
        #%%
        df['body_SplChar'] = df['body'].str.replace("\r", " ")
        df['body_SplChar'] = df['body_SplChar'].str.replace("\n", " ")
        df['body_SplChar'] = df['body_SplChar'].str.replace("    ", " ")

        # Filter out " when quoting text
        df['body_SplChar'] = df['body_SplChar'].str.replace('"', '')

        #LOWER CASING OF ALL TEXT FOR UNIFORMITY
        df['body_LwerCasing'] = df['body_SplChar'].str.lower()

        #FILTER THE UNNECESSARY PUNCTUATION SIGNS
        punctuation_signs = list("?:!.,;’`','’")

        for punctuation in punctuation_signs:
            df['body_LwerCasing'] = df['body_LwerCasing'].str.replace(punctuation, '')

        symbols = "[/(/)\-/*/,\!\?]|[^ -~]"
        df['body_Parsed'] = df['body_LwerCasing'].apply(lambda series: re.sub(symbols, " ", series))

        #FILTERING POSSESSIVE PRONOUN- "'s"
        #df['body_Parsed_1'] = df['body_Parsed_1'].str.replace("'s", "")

        #STEMMING AND LEMMATIZATION
        # # Downloading punkt and wordnet from NLTK
        # nltk.download('punkt')
        # print("------------------------------------------------------------")
        # nltk.download('wordnet')

        #Saving the lemmatizer into an object
        wordnet_lemmatizer = WordNetLemmatizer()

        ######## DON'T RUN#######
        #df['body_lemmatized'] = df.body_Parsed.apply(lambda series: ' '.join([word.lemma_ for word in nlp(series)]))
        #df['contains_pron'] = df.body_lemmatized.apply(lambda series: 1 if series.__contains__('-PRON-') else 0 )
        #df['verb_count'] = df.body_lemmatized.apply(lambda series: len([token for token in nlp(series) if token.pos_ == 'VERB']))



        df['lemmatized_comment'] = get_lemmatized_text(df['body_Parsed'])

        #STOPWORD REMOVAL
        for stop_word in stop_words:

            regex_stopword = r"\b" + stop_word + r"\b"
            df['body_StopWordRem'] = df['lemmatized_comment'].str.replace(regex_stopword, '')

        # Convert to list
        df_parsed = df.body_StopWordRem.values.tolist()
        df_parsed = [re.sub('\S*@\S*\s?', '', str(sent)) for sent in df_parsed]

        # Remove new line characters
        df_parsed = [re.sub('\s+', ' ', sent) for sent in df_parsed]

        # Remove distracting single quotes
        df_parsed = [re.sub("\'", "", sent) for sent in df_parsed]
        df_parsed = [re.sub("-", " ", sent) for sent in df_parsed]
        df_parsed = [re.sub(":", "", sent) for sent in df_parsed]



        df_parsed_words = list(sent_to_words(df_parsed))

        #CREATING N-GRAMS MODELS
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(df_parsed_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[df_parsed_words], threshold=100)  

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)



        # Remove Stop Words
        data_covwords_nostops = remove_stopwords(df_parsed_words)

        # Form Bigrams
        data_words_bigrams = make_bigrams(data_covwords_nostops)
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        for words in data_lemmatized:
            for word in words:
                if len(word) < 3:
                    words.remove(word)

        df_lemmatized = pd.DataFrame({"data_lemmatized": data_lemmatized})
        #%%
        symbols = "['',\[\]\!\?]|[^ -~]"
        df_lemmatized['data_lemmaStr'] = df_lemmatized['data_lemmatized'].astype(str).apply(lambda series: re.sub(symbols, " ", series))
        #df_lemmatized
        #######################################################
        ## SECTION 5 : EXPLORATORY DATA ANALYSIS
        ## 
        #######################################################
        #%%
        #100 MOST FREQUENTLY USED WORDS
        st.session_state['df_lemmatized'] = df_lemmatized.copy()
        freq_words(df_lemmatized['data_lemmaStr'])
        #%%
        #WORDCLOUD
        list=[]
        for i in df_lemmatized['data_lemmaStr'] :
            list.append(i)
        slist = str(list)

        st.session_state['wordcloud']  = WordCloud(width=1100, height=500).generate(slist)

        
        
        # plt.imshow(st.session_state['wordcloud'] , interpolation="bilinear")
        # plt.axis("off")
        # plt.margins(x=0, y=0)
        
        # st.subheader('WORD FREQUENCY WORDCLOUD')
        # st.pyplot(use_container_width=True)
        #%%
        #TIMESERIES WITH REGRESSION ANALYSIS
        freq='D' # or M or Y
        df_1 = df.groupby(['subreddit', pd.Grouper(key='creation_date', freq=freq)])['subreddit'].agg(['count']).reset_index()
        df_1 = df_1.sort_values(by=['creation_date', 'count']).reset_index(drop=True)
        # group the dataframe
        group = df_1.groupby('subreddit')
        # create a blank canvas
        fig = go.Figure()
        # each group iteration returns a tuple
        # (group name, dataframe)
        for group_name, df_1 in group:
            fig.add_trace(
                go.Scatter(
                    x=df_1['creation_date']
                    , y=df_1['count']
                    , fill='tozeroy'
                    , name=group_name
                ))
            # generate a regression line with px
            help_fig = px.scatter(df_1, x=df_1['creation_date'], y=df_1['count']
                                , trendline="lowess")
            # extract points as plain x and y
            x_trend = help_fig["data"][1]['x']
            y_trend = help_fig["data"][1]['y']
            # add the x,y data as a scatter graph object
            fig.add_trace(
                go.Scatter(x=x_trend, y=y_trend
                        , name=str('trend ' + group_name)
                        , line = dict(width=4, dash='dash')))

            transparent = 'rgba(0,0,0,0)'
            fig.update_layout(
                hovermode='x',
                showlegend=True
                , paper_bgcolor=transparent
                , plot_bgcolor=transparent
            )
        st.session_state['fig'] = fig
        #Session-State Fig
        #st.subheader('MONTHLY TIME SERIES WITH REGRESSION')
        #st.plotly_chart(fig, use_container_width=True, height = 600)
        
        #%%
        #######################################################
        ## SECTION 6 : TOPIC MODELLING
        ## TOPIC MODELLING USING LDA
        #######################################################
        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        [[(id2word[id], freq) for id, freq in cp] for cp in corpus[11:12]]

        # Building the LDA model for Covid dataset
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=10, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
        doc_lda = lda_model[corpus]
        print('TOPICS:')
        print(lda_model.print_topics())
        #%%
        panel = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='tsne')
        html_string = pyLDAvis.prepared_data_to_html(panel)
        #from streamlit import components
        st.session_state['html_string'] = html_string
        #Session-State html_string
        #st.subheader('MOST WIDELY DISCUSSED TOPICS')
        #components.html(html_string, width=1300, height=800)
        
        #%%
        # Compute Perplexity
        try:
            print('\nPerplexity: ', lda_model.log_perplexity(corpus))
            Perplexity=lda_model.log_perplexity(corpus)
            perplexity_str = "PERPLEXITY SCORE OF TOPICS : "+str(Perplexity)
            #Session-State perplexity_str
            #st.subheader(perplexity_str)
            st.session_state['perplexity_str'] = perplexity_str
            print('THIS IS A TEST TO CHECK WE COMPLETED THIS')
        except ValueError:
            print(ValueError)
        #%%
        #######################################################
        ## SECTION 7 : SENTIMENT ANALYTICS
        ## USING VADER SENTIMENT ANALYZER
        #######################################################


        df_lemmatized['Subjectivity'] = df_lemmatized['data_lemmaStr'].apply(getSubjectivity) 
        df_lemmatized['Polarity'] = df_lemmatized['data_lemmaStr'].apply(getPolarity) 
        df_lemmatized['Analysis'] = df_lemmatized['Polarity'].apply(analysis)
        df_lemmatized.head()
        #%%
        # Sentiment Bar Chart
        
        sentiment_plot = pd.DataFrame(df_lemmatized['Analysis'].value_counts().sort_values(ascending=False)[:3]).T
        colors = ['lightslategray',] * 3
        colors[0] = 'green'
        colors[1] = 'red'


        fig1 = go.Figure(data=[go.Bar(x=sentiment_plot.columns,
                                        y=[sentiment_plot[i][0] for i in sentiment_plot],
                                        marker_color=colors)])
            #fig.update_layout(xlabel='Count')
        fig1.update_layout(yaxis=dict(title='Count'))
        st.session_state['fig1'] = fig1
        
        fig2 = px.pie(sentiment_plot.columns, values=[sentiment_plot[i][0] for i in sentiment_plot], names=sentiment_plot.columns,
                        color_discrete_sequence=px.colors.sequential.RdBu)
        fig2.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20,
                        marker=dict(colors=colors, line=dict(color='#000000', width=2)))
        st.session_state['fig2'] = fig2

        # col1, col2 = st.columns(2)
        # with col1:
        #     #Session-state fig1
        #     st.plotly_chart(fig1)
        
        # # Sentiment Pie Chart
        # #col2.header('SENTIMENT POLARITY : PIE CHART')
        # with col2:
        #     colors = ['crimson', 'seagreen', 'lightslategray']
        #     #Session-state fig2
        #     st.plotly_chart(fig2)
        #%%
        # K-Means of Polarity vs Subjectivity
        df_kmeans = df_lemmatized[["Polarity","Subjectivity"]]
        df_kmeans.head()

        df_kmeans.isna().sum()

        df_kmeans = df_kmeans.dropna()
        df_kmeans.isnull().sum()

        df_kmeans.isna().sum()

        comments_array = np.array(df_kmeans)
        #%%
        n_cluster = 4
        pred = KMeans(n_clusters=n_cluster).fit_predict(comments_array)

        df_kmeans["pred"] = pred

        scatter_plot = px.scatter(df_kmeans, x="Polarity", y="Subjectivity",color="pred")
        scatter_plot.update_layout(width = 1000, height=600)
        
        st.session_state['scatter_plot'] = scatter_plot
        #Session-state scatter_plot
        # st.plotly_chart(scatter_plot,use_container_width=True)
        #%%
        #######################################################
        ## SECTION 8 : INTERACTIVE KNOWLEDGE GRAPH
        ## RELATIONAL TAXONOMY
        #######################################################
        
        entity_pairs = []

        for i in tqdm(df_lemmatized['data_lemmaStr']):
            entity_pairs.append(get_entities(i))

        

        relations = [get_relation(i) for i in tqdm(df_lemmatized['data_lemmaStr'])]

        print(pd.Series(relations).value_counts()[300:])

        # extract subject
        source = [i[0] for i in entity_pairs]
        print('LENGTH OF SOURCE : ',len(source))
        # extract object
        target = [i[1] for i in entity_pairs]
        print('LENGTH OF TARGET : ',len(target))
        print('LENGTH OF RELATIONS : ',len(relations))
        df_knowledgeGraph = pd.DataFrame({'source':source, 'target':target, 'relation':relations})
        #df_knowledgeGraph

        #Remove all rows with any null value columns from entity-relation dataframe
        df_KGnan = df_knowledgeGraph.replace('', np.nan)
        df_KGnan['relation'].replace('  ',np.nan)
        df_KGCleaned = df_KGnan.dropna()

        relation_list = df_KGCleaned['relation'].to_list()
        print('relation_list: ',relation_list)
        st.session_state["df_KGCleaned"] = df_KGCleaned
        print("Session state value is ")
        print(st.session_state["df_KGCleaned"])
        st.session_state['has_dataProcessed'] = True


# df_lemmatized = st.session_state['df_lemmatized']
# if isinstance(df_lemmatized, pd.DataFrame) and not df_lemmatized.empty:
#     print("df_lemmatized content is ")
#     print(df_lemmatized['data_lemmaStr'])
#     freq_words(df_lemmatized['data_lemmaStr'])
    
if st.session_state['wordcloud'] is not None:
    plt.imshow(st.session_state['wordcloud'] , interpolation="bilinear")
    plt.axis("off")
    plt.margins(x=0, y=0)
    st.subheader('WORD FREQUENCY WORDCLOUD')
    st.pyplot(use_container_width=True)


if st.session_state['fig'] is not None:
    st.subheader('MONTHLY TIME SERIES WITH REGRESSION')
    st.plotly_chart(st.session_state['fig'], use_container_width=True, height = 600)


if st.session_state['html_string'] is not None:
    st.subheader('MOST WIDELY DISCUSSED TOPICS')
    components.html(st.session_state['html_string'], width=1300, height=800)


if st.session_state['perplexity_str'] is not None:
    st.subheader(st.session_state['perplexity_str'] )


col1, col2 = st.columns(2)
with col1:
    if st.session_state['fig1'] is not None:
        st.subheader('SENTIMENT ANALYTICS : POLARITY')
        st.plotly_chart(st.session_state['fig1'])
with col2:
    colors = ['crimson', 'seagreen', 'lightslategray']
    if st.session_state['fig2'] is not None:
        st.plotly_chart(st.session_state['fig2'])


if st.session_state['scatter_plot'] is not None:
    st.subheader('SENTIMENT ANALYTICS : SUBJECTIVITY VS POLARITY')
    st.plotly_chart(st.session_state['scatter_plot'],use_container_width=True)




df_KGCleaned = st.session_state['df_KGCleaned']
if isinstance(df_KGCleaned, pd.DataFrame) and not df_KGCleaned.empty:
    st.subheader('ENTITY - RELATION : NETWORK GRAPH')
    df_relation = df_KGCleaned['relation'].unique()
    
    selected_relation = st.selectbox('Select relation to visualize', df_relation.tolist() )
    if selected_relation:
        print("selected")
        if(len(selected_relation) == 0):
            st.text('Please select at least 1 relation to get started')
        else:
            print("Running. now")
            G=nx.from_pandas_edgelist(df_KGCleaned[df_KGCleaned['relation']==selected_relation], "source", "target",
                                        edge_attr=True, create_using=nx.MultiDiGraph())

            net = Network(height='800px', width='100%', bgcolor='#222222', font_color='white')
            # load the networkx graph
            net.from_nx(G)
            # show
            # Generate network with specific layout settings
            net.repulsion(node_distance=100, central_gravity=0.2,
                            spring_length=110, spring_strength=0.10,
                            damping=0.95)
            net.show_buttons(filter_=['nodes', 'physics'])
            # Save and read graph as HTML file (on Streamlit Sharing)
            try:
                path = '/tmp'
                net.save_graph(f'{path}/pyvis_graph.html')
                HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

            # Save and read graph as HTML file (locally)
            except:
                path = '/html_files'
                net.save_graph(f'{path}/pyvis_graph.html')
                HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

            # Load HTML file in HTML component for display on Streamlit page
            components.html(HtmlFile.read(), height=600)