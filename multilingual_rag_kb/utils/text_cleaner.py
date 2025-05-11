import re
import unicodedata
from langdetect import detect
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize


def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


def clean_text(text: str) -> str:
    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove non-printable characters and excessive whitespace
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def tokenize_sentences(text: str) -> list:
    return sent_tokenize(text)


def preprocess_file(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as infile:
        raw = infile.read()

    cleaned = clean_text(raw)
    sentences = tokenize_sentences(cleaned)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for sent in sentences:
            outfile.write(sent.strip() + '\n')
