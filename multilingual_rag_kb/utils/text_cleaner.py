import re
import unicodedata
from langdetect import detect
import nltk
import string

nltk.download('punkt')
from nltk.tokenize import sent_tokenize


def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


def clean_text_english(text: str) -> str:
    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove non-printable characters and excessive whitespace
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    
    
    #remove the email
    text = re.sub(r'\S+@\S+', '', text)

    # Only keep English letters/numbers
    # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    #remove the links
    text = re.sub(r'http\S+', '', text)
    #remove this word /s or /s/
    text = re.sub(r'/s', '', text)
    #remove the / alone follwed by space 
    text = re.sub(r'/\s', '', text)
    

    return text.strip()


def clean_text_urdu(text: str) -> str:
    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    #remove the english text 
    # text = re.sub(r'[a-zA-Z\s]', '', text)

    # Remove non-printable characters and excessive whitespace
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    #remove the email
    text = re.sub(r'\S+@\S+', '', text)

    # Remove only punctuation and symbols, keep all letters/numbers (including Urdu)
    # text = re.sub(r'[{}]+'.format(re.escape(string.punctuation)), '', text)

    #remove the links
    text = re.sub(r'http\S+', '', text)
    #remove  
    
    text = re.sub(r'', ' ', text)
        

    #remove the english text between urdu text
    text = re.sub(r'[a-zA-Z\s]+', ' ', text)
    
    #remove  🔴 type emoji 

    text= re.sub(r'🔴', '', text)
    
    #remove this pattern (&      ) replace by space 
    text = re.sub(r'&      ', ' ', text)

    
    


    

     
 


    return text.strip()


def tokenize_sentences(text: str) -> list:
    return sent_tokenize(text)


def preprocess_file(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as infile:
        raw = infile.read()

    # Use file name to select language-specific cleaner
    if 'urd' in input_path.lower():
        cleaned = clean_text_urdu(raw)
    else:
        cleaned = clean_text_english(raw)
    sentences = tokenize_sentences(cleaned)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for sent in sentences:
            outfile.write(sent.strip() + '\n')
