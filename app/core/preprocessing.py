from enum import Enum
import re
from typing import List, Tuple, Dict, TypedDict, Callable, Optional
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ******** uncomment this for additional data download *********
# import nltk
# import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('punkt')
# nltk.download('stopwords')


# *** text processing hook types definition ****
class HookSide(str, Enum):
    Before = 'before'
    After = 'after'


class PreprocessingHook(TypedDict):
    before: Optional[Callable[[str | List[str]], str | List[str]]]
    after: Optional[Callable[[str | List[str]], str | List[str]]]


type PreprocessingHooksConfig = Dict[str, PreprocessingHook]


# *** text processing pipeline steps definition ****
class TextPreprocessingStep(str, Enum):
    Clear = "clear"
    Tokenize = "tokenize"
    StopWords = "stop-words"
    Stemming = "stemming"


# *** additional functions for text preprocessing ****
def text_cleanup(text: str) -> str:
    no_tags_text = re.sub(r"<.*?>", ' ', text.lower())
    no_symbols_text = re.sub(r"[^\w\s]", ' ', no_tags_text)
    no_numbers_text = re.sub(r"\d", ' ', no_symbols_text)
    no_dup_spaces_text = re.sub(r"\s\s+", ' ', no_numbers_text)
    return no_dup_spaces_text


def stopwords_removal(words: List[str]) -> List[str]:
    return [word for word in words if word.lower() not in stopwords.words('english')]


def text_stemmer(words: List[str]) -> List[str]:
    stemmer = PorterStemmer()
    stemmed = []
    for word in words:
        stem_word = stemmer.stem(word)
        stemmed.append(stem_word)

    return stemmed


# **** text preprocessing hook apply function ****
def apply_hook(hooked_text: str | List[str], step: str, side: HookSide, hooks: Optional[PreprocessingHooksConfig] = None) \
        -> str | List[str]:
    if hooks is None:
        return hooked_text
    if step not in hooks.keys():
        return hooked_text
    if side not in hooks[step].keys():
        return hooked_text
    return hooks[step][side](hooked_text)


# **** core text_preprocessing function ****
def text_preprocessing(text: str, pipeline: List[TextPreprocessingStep], hooks: Optional[PreprocessingHooksConfig] = None) -> str:
    local_text = text

    if TextPreprocessingStep.Clear.value in pipeline:
        local_text = apply_hook(local_text, TextPreprocessingStep.Clear.value, HookSide.Before, hooks)
        local_text = text_cleanup(local_text)
        local_text = apply_hook(local_text, TextPreprocessingStep.Clear.value, HookSide.After, hooks)

    if TextPreprocessingStep.Tokenize.value in pipeline:
        local_text = apply_hook(local_text, TextPreprocessingStep.Tokenize.value, HookSide.Before, hooks)
        local_text = word_tokenize(local_text)
        local_text = apply_hook(local_text, TextPreprocessingStep.Tokenize.value, HookSide.After, hooks)

    if TextPreprocessingStep.StopWords.value in pipeline:
        local_text = apply_hook(local_text, TextPreprocessingStep.StopWords.value, HookSide.Before, hooks)
        local_text = stopwords_removal(local_text)
        local_text = apply_hook(local_text, TextPreprocessingStep.StopWords.value, HookSide.After, hooks)

    if TextPreprocessingStep.Stemming.value in pipeline:
        local_text = apply_hook(local_text, TextPreprocessingStep.Stemming.value, HookSide.Before, hooks)
        local_text = text_stemmer(local_text)
        local_text = apply_hook(local_text, TextPreprocessingStep.Stemming.value, HookSide.After, hooks)

    return local_text


# apply clear with default pipeline with no hooks
def clear_text(text: str) -> str:
    tokens = text_preprocessing(text, ('clear', 'tokenize', 'stop-words', 'stemming'), {})
    return ' '.join(tokens)


# example hook to be called before text clear step
def before_clear_hook(text: str) -> str:
    print('Im hooked before clear')
    text = text + ' This is the end'
    return text


# example hook to be called after text stemming step
def after_stemming_hook(words: List[str]) -> List[str]:
    print('Im hooked after stemming')
    return [word for word in words if 'ru' not in word]


# # text to process
# text = 'hello<b>,~!me you were running wishes developers </b> 195$ 2024-04-10'
# # text processing pipeline steps
# pipe = ('clear', 'tokenize', 'stop-words', 'stemming')
# # example text processing hooks configuration dict
# hooks = {'clear': {'before': before_clear_hook}, 'stemming': {'after': after_stemming_hook}}
#
# # example usage
# print(text_preprocessing(text, pipe, hooks))
