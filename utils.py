import re
import pycld2 as cld2
import strip_markdown

MIN_AUTHOR_TEXT_LENGTH = 8


def clean_text(text: str) -> str:
    text = text.replace('\r\n', '\n')
    text = re.sub(r'http\S+|www\.\S+|https\S+', 'URL', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def parse_author_text(text: str, check_language=True) -> str | None:
    text = text.strip()  # Remove leading and trailing whitespaces
    #text = re.sub(r'http\S+', '', text)  # Remove URLs
    if not text or text.isspace():
        return None
    if len(text) < MIN_AUTHOR_TEXT_LENGTH:
        return None
    if check_language and not is_text_czech(text):
        return None
    return text


def remove_markdown(text: str) -> str:
    # Create a pattern to match lines starting with '>' (blockquotes),
    # or lines that starts with '^(' and ends with ')' (footnotes)
    pattern = re.compile(r'(^>.*$\n?|^\^\(.*\)$\n?)', re.MULTILINE)
    text = pattern.sub('', text)
    return strip_markdown.strip_markdown(text)


def is_text_czech(text: str) -> bool:
    _, _, details = cld2.detect(text, bestEffort=True, hintLanguage='cs')
    return details[0][1] == 'cs'
