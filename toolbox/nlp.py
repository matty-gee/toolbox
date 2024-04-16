import string
from collections import Counter

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('vader_lexicon')

import textstat
import tiktoken
from wordcloud import WordCloud
from textblob import TextBlob
import torch
from transformers import pipeline, set_seed, GPT2LMHeadModel, GPT2Tokenizer, BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer


def read_txt(txt_file):
    with open(txt_file, 'r') as f:
        return f.read()


#----------------------------------------------------------#
# preprocessing functions
#----------------------------------------------------------#


def lowercase(word):
    return word.lower()

def remove_punctuation(words):
    return words.translate(str.maketrans('', '', string.punctuation))

def get_my_pos(word):
    # specify the pos for words in this datatset
    to_check = {'felt': 'v'}
    return to_check.get(word, None)

def get_nltk_pos(word_list):
    """Map nltk POS tags to wordnet POS tags for a word list."""
    mapping = {'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV, 'J': wordnet.ADJ}
    pos_tagged = nltk.pos_tag(word_list)
    return [mapping.get(pos[1][0], wordnet.NOUN) for pos in pos_tagged]

def infrequent_filter(words, min_freq=2):
    # remove words that appear less than min_freq times
    word_freq = Counter(words)
    return [word for word in words if word_freq[word] >= min_freq]

def stopwords_filter(words):
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word not in stop_words]

def remove_punctuation(words):
    return words.translate(str.maketrans('', '', string.punctuation))

def lemmatize(word, pos=None):

    # check if the word is in the exceptions dictionary for the given POS; otherwise, lemmatize the word normally
    # https://stackoverflow.com/questions/33594721/why-nltk-lemmatization-has-wrong-output-even-if-verb-exc-has-added-right-value

    exclude = ['boss']
    if word in exclude:
        return word
    
    # POS tag the word if not provided
    if pos is None: 
        pos = get_my_pos(word) or get_nltk_pos([word])[0]

    # check the morphy exceptions
    exceptions = wordnet._exception_map[pos]
    if word in exceptions:
        lemmatized = exceptions[word][0]
    else:
        lemmatizer = WordNetLemmatizer()
        lemmatized = lemmatizer.lemmatize(word, pos)

    # # do they match?
    # if (verbose) & (lemmative_morphy != lemmatize_wn):
    #     print(f"Mismatch - Morphy: {lemmative_morphy}, WordNet: {lemmatize_wn}")

    return lemmatized

def remove_words(word_list, words_to_remove):
    """
    Removes specified words and their possessive forms from a list of words or a string.

    Parameters:
    - word_list (list or str): The list of words or string from which to remove words.
    - words_to_remove (list): List of words to remove.

    Returns:
    - list or str: The words after removal, in the same format as the input.
    """
    
    # validate input
    if not isinstance(word_list, (list, str)) or not isinstance(words_to_remove, list):
        raise ValueError("Invalid input type")
    is_str = isinstance(word_list, str)
    if is_str: 
        word_list = word_list.split()

    # words to remove
    words_to_remove = [re.escape(w.lower()) for w in words_to_remove] +\
                      [re.escape(f"{w.lower()}'s") for w in words_to_remove] +\
                      [re.escape(f"{w.lower()}'ll") for w in words_to_remove]
    # remove the words
    filtered_words = [w for w in word_list if w.lower() not in words_to_remove]

    # return in the original format
    return ' '.join(filtered_words) if is_str else filtered_words

def preprocess_text(text, exclude_list=None, remove_stopwords=True, min_freq=0, return_tokenized=True):
    # generic preprocessing for NLP tasks
    # expects text in a string format; if its a list, access or join the list of words
    if isinstance(text, list):
        text = text[0] if len(text) == 1 else ' '.join(text)
    text   = remove_words(text, exclude_list) if exclude_list else text # remove specific words
    text   = [lowercase(remove_punctuation(word)) for word in [text]][0] # lowercasee & remove punctuation before tokenizing
    tokens = word_tokenize(text) # tokenize before preprocessing
    tokens = stopwords_filter(tokens) if remove_stopwords else tokens # remove stop words
    tokens = infrequent_filter(tokens) if (min_freq > 1) else tokens # remove infrequent words
    tokens = [lemmatize(word) for word in tokens]  # lemmatize (e.g. running -> run)
    return tokens if return_tokenized else " ".join(tokens) # output as tokenized or joined

def pad_tokens(tokens):
    # Pad input tokens
    max_len = 0
    for i in tokens:
        if len(i) > max_len:
            max_len = len(i)
    return np.array([i + [0]*(max_len-len(i)) for i in tokens])

def make_attention_mask(padded):
    # Create attention masks
    return np.where(padded != 0, 1, 0)

def build_vocab(sentences, verbose=True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


#----------------------------------------------------------#
# Semantic embeddings
#----------------------------------------------------------#


def get_sentencetransformer_embeddings(sentences, model='mpnet'):
    ''' 
        function to return sentence embeddings using the sentence-transformers library
        - sentences (list/array): sentences to embed
        - model (str): name of the model to use
        
        See the following link for top performers in sentence embedding: 
            https://www.sbert.net/docs/pretrained_models.html

    '''
    model_dict = {'mpnet': 'sentence-transformers/all-mpnet-base-v2', 
                  'roberta': 'sentence-transformers/all-distilroberta-v1',
                  'mlml12': 'sentence-transformers/all-MiniLM-L12-v2',
                  'multi_qa': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
                  'mpnet_negation': 'dmlls/all-mpnet-base-v2-negation'}
    if model in model_dict:
        model = model_dict[model]
    model = SentenceTransformer(model)
    return model.encode(sentences)

# GPT-2 models
# gpt2: 12-layer, 768-hidden, 12-heads, 117M parameters
# gpt2-medium: 24-layer, 1024-hidden, 16-heads, 345M parameters
# gpt2-large: 36-layer, 1280-hidden, 20-heads, 774M parameters
# gpt2-xl: 48-layer, 1600-hidden, 25-heads, 1558M parameters

class GPT2:

    def __init__(self,
                 gpt_model='gpt2',
                 padding=True, 
                 truncation=True, 
                 return_tensors="pt", 
                 is_split_into_words=False, 
                 add_prefix_space=True,
                 which_tokens=None, 
                 which_layers=None,
                 which_pooling=None, 
                 remove_padding=False,
                 verbose=False):
    
        set_seed(23)
        
        # embedding parameters
        self.which_tokens = which_tokens
        self.which_layers = which_layers
        self.which_pooling = which_pooling
        self.remove_padding = remove_padding # if want to return vector embeddings w/o padding altogehter
        self.verbose = verbose

        # tokenizer parameters
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.is_split_into_words = is_split_into_words
        self.add_prefix_space = add_prefix_space

        # gpt2 model parameters (specific to gpt2 version)
        self.gpt_model = gpt_model
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model)
        self.model = GPT2LMHeadModel.from_pretrained(gpt_model, output_hidden_states=True)
        self.model.trainable = False
        self.generator = pipeline('text-generation', model=gpt_model)
        
        self.n_layers = self.model.config.n_layer + 1 # number of hidden layers: attention layers + 1 linear layer
        self.n_embd   = self.model.config.n_embd # number of embedding dimensions in each layer 

        if self.tokenizer.pad_token is None:
            # https://github.com/huggingface/transformers/issues/8452
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def tokenize(self, text_sequences):

        if isinstance(text_sequences, str):
            text_sequences = [text_sequences]
        
        tokenized = self.tokenizer.batch_encode_plus(text_sequences,
                                                     padding=self.padding, # pad all sequences to be the same length
                                                     truncation=self.truncation, # truncate sequences that are too long
                                                     return_tensors=self.return_tensors, # return PyTorch tensors
                                                     is_split_into_words=self.is_split_into_words, # is it already split into words?
                                                     add_prefix_space=self.add_prefix_space) # add space before each sequence
        self.input_ids, self.attention_mask = tokenized['input_ids'], tokenized['attention_mask'] # '50256' is padding token
        self.n_seqs, self.n_tokens = self.input_ids.shape # number of sequences & number of tokens in the batch
        self.seg_lens = self.attention_mask.sum(dim=1) # the lengths of the different sequences

    def forward_pass(self):
        self.model.eval() # feed-forward only
        with torch.no_grad(): # no gradients
            outputs = self.model(self.input_ids, attention_mask=self.attention_mask)
        self.logits = outputs.logits.squeeze()
        self.hidden_layers = torch.stack(outputs.hidden_states, dim=0).permute(1,2,0,3)
        assert self.hidden_layers.size() == (self.n_seqs, self.n_tokens, self.n_layers, self.n_embd)
        if self.verbose: print(f"Hidden layers shape: {self.hidden_layers.size()}")
        
    # maybe add the optional arguments here so I can tokenize and forward pass and then play around with outputting diff. embeddings without re-running
    def get_embeddings(self):

        # select specific tokens &/or layers [optional]
        if self.which_tokens is not None:
            # important: each sequence can be of diff. length, so make sure to get the correct token
            if isinstance(self.which_tokens, int):
                if self.which_tokens >= 0:
                    raise ValueError(f"'which_tokens' must be negative")  
                self.hidden_layers  = torch.stack([self.hidden_layers[i, len + self.which_tokens] for i, len in enumerate(self.seg_lens)]).unsqueeze(1) # maintain 4D tensor
                self.attention_mask = torch.ones((self.n_seqs, 1)) # recreate attention mask as 2D tensor of size (n_seqs, 1)
            else:
                raise NotImplementedError(f"which_tokens={self.which_tokens}")
        if self.which_layers is not None:
            if isinstance(self.which_layers, int):
                self.hidden_layers = self.hidden_layers[:,:,self.which_layers,:].unsqueeze(2)
            else:
                self.hidden_layers = self.hidden_layers[:,:,self.which_layers[0]:self.which_layers[1],:]
        
        if self.verbose: print(f"Extracted layers shape: {self.hidden_layers.size()}")

        # pooling to combine layers [optional]
        if self.which_pooling is None:
            self.token_embeddings = self.hidden_layers
        elif self.which_pooling == 'concat':
            self.token_embeddings = self.hidden_layers.reshape(self.n_seqs, self.n_tokens, -1)
        elif self.which_pooling == 'sum':
            self.token_embeddings = self.hidden_layers.sum(dim=2)
        elif self.which_pooling == 'mean':
            self.token_embeddings = self.hidden_layers.mean(dim=2)
        else:
            raise ValueError("Invalid 'which_pooling' value. It must be None, 'concat', 'sum', or 'mean'.")

        # exclude padding tokens from semantic embeddings
        attention_mask = self.attention_mask.bool().unsqueeze(-1)
        if len(self.token_embeddings.size()) == 4:
            attention_mask = attention_mask.unsqueeze(-1) 
        if self.verbose: print(f"Attention mask shape: {attention_mask.size()}")
        self.token_embeddings = (self.token_embeddings * attention_mask).squeeze() # zero-out padded tokens
        if self.token_embeddings.dim() == 2:
            self.token_embeddings = self.token_embeddings.unsqueeze(0)

        # calculate sequence embeddings: the average over the token embeddings
        self.sequence_embeddings = (self.token_embeddings.sum(dim=1) / attention_mask.sum(dim=1).float()).squeeze()   

        # remove padded tokens (do after averaging)
        if self.remove_padding:
            n_incl_tokens = attention_mask.sum(dim=1) # number of tokens in each sequence
            depadded_token_embeddings = []
            for seq in np.arange(self.n_seqs): # for each sequence, get included tokens
                depadded_token_embeddings.append(self.token_embeddings[seq][:n_incl_tokens[seq]])
            self.token_embeddings = depadded_token_embeddings
    
        return {'token': self.token_embeddings, 'sequence': self.sequence_embeddings}

    def get_predictions(self):
        # use logits to predict next token at each token
        self.predicted_ids, self.predicted_tokens = [], []
        for logits in self.logits:
            predicted_id = torch.argmax(logits).item()
            self.predicted_ids.append(predicted_id)
            self.predicted_tokens.append(self.tokenizer.decode(predicted_id))
        self.predicted_text = ('').join(self.predicted_tokens)
        return self.predicted_text
    
    # add a prediction error method...
     
    def generate_text(self,
                      text_prompt,
                      max_length=25,
                      num_return_sequences=5,
                      repetition_penalty=1.5,
                      method='greedy',
                      num_beams=5, # for beam search
                      temperature=0.5, # higher for more 'creativity'
                      top_k=10, # for top-k sampling
                      top_p=0.85): # for top-p sampling
        
        # TODO - figure out the warning about token id and attention mask
        # encode the prompt as input ids
        input_ids = self.tokenizer.encode(text_prompt, return_tensors='pt')

        # use one of different methods of generating
        with torch.no_grad():
            if method == 'greedy':
                output_ids = self.model.generate(input_ids,
                                                 max_length=max_length,
                                                 num_return_sequences=num_return_sequences,
                                                 repetition_penalty=repetition_penalty,
                                                 do_sample=True)      
            elif method == 'beam':
                output_ids = self.model.generate(input_ids,
                                            max_length=max_length, 
                                            num_beams=num_beams, 
                                            no_repeat_ngram_size=2, 
                                            early_stopping=True,
                                            repetition_penalty=repetition_penalty,
                                            num_return_sequences=num_return_sequences)
            elif method == 'sampling':
                output = self.model.generate(input_ids,
                                            max_length=max_length,
                                            return_dict_in_generate=True, 
                                            output_scores=True,
                                            do_sample=True,
                                            temperature=temperature,
                                            repetition_penalty=repetition_penalty,
                                            num_return_sequences=num_return_sequences) # increases chance of high probability words
            elif output_ids == 'top_sampling':
                output_ids = self.model.generate(input_ids,
                                            return_dict_in_generate=True, 
                                            output_scores=True,
                                            max_length=max_length,
                                            do_sample=True,
                                            repetition_penalty=repetition_penalty,
                                            top_k=top_k, 
                                            top_p=top_p)
                
def get_gpt2_embeddings(text, gpt_model='gpt2', **kwargs):
    gpt2 = GPT2(gpt_model=gpt_model, **kwargs)
    gpt2.tokenize(text)
    gpt2.forward_pass()
    return gpt2.get_embeddings()


#----------------------------------------------------------#
# Sentiment analysis
#----------------------------------------------------------#


def get_sentiment(text, preprocess=True):

    ''' 
        Get sentiment scores for a given text using nltk and textblob.
    '''

    if preprocess:
        text = preprocess_text(text, tokenize=False)

    # nltk sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    sentiment = {k.replace('neg', 'negativity').replace('neu', 'neutrality').replace('pos', 'positivity'): v for k, v in sentiment.items()}

    # textblob sentiment
    blob = TextBlob(text)
    sentiment['polarity'] = blob.sentiment.polarity
    sentiment['subjectivity'] = blob.sentiment.subjectivity

    return sentiment



#----------------------------------------------------------#
# Named Entity Recognition (NER)
#----------------------------------------------------------#


ner_pipeline = pipeline('ner', aggregation_strategy='simple')

def run_named_entity_recognition(text):
    # add one of those progress bars things...
    sentences = nltk.tokenize.sent_tokenize(text) # split into sentences; can do some minimal preprocessing too
    dfs = []
    for s, sentence in enumerate(sentences):
        df = pd.DataFrame(ner_pipeline(sentence))
        df['sentence'] = s+1
        dfs.append(df)
    return pd.concat(dfs)