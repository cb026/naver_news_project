
"""
knlp_processor.py

Author: CB Park
Date: Oct. 10, 2023
"""

import os
import re
import pickle
import string
import pandas as pd
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords


class KNLP_Preprocessor:
    
    def remove_email_address(self, text):
        pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' # E-mail제거
        text = re.sub(pattern=pattern, repl='', string=text)
        return text
    
    def remove_urls(self, text):
        pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
        text = re.sub(pattern=pattern, repl='', string=text)
        return text
    
    def remove_html_tags(self, text):
        pattern = '<[^>]*>'         # HTML 태그 제거
        text = re.sub(pattern=pattern, repl='', string=text)
        return text
    
    def remove_special_characters(self, text):
        pattern = '[^\w\s\n]'         # 특수기호제거
        text = re.sub(pattern=pattern, repl='', string=text)
        text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', string=text)
        text = re.sub('\n', '.', string=text)
        return text

    def remove_korean_alphabets(self, text):
        pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
        text = re.sub(pattern=pattern, repl='', string=text)
        return text
    
    def remove_numbers(self, text):
        return re.sub(r'\d+', '', text)
    
    def clean_text(self, text):
        text = self.remove_email_address(text)
        text = self.remove_urls(text)
        text = self.remove_html_tags(text)
        text = self.remove_special_characters(text)
        text = self.remove_korean_alphabets(text)
        text = self.remove_numbers(text)
        return text
    
    def remove_punctuations(self, text):
        filtered_text = text.translate(str.maketrans("", "", string.punctuation))
        return filtered_text.strip()

    def clean_corpus(self, text):
        filtered_text = self.clean_text(text)
        filtered_text = self.remove_punctuations(filtered_text)
        return filtered_text

    def get_km_dict_kiwi(self, filepath):
        with open(filepath, 'rb') as f:           
            km_dict = pickle.load(f)
        return km_dict
    
    def parse_tokens_kiwi(self, tokens, km_dict):
        for token in tokens:
            print(f'\n{token}')
            print(f'   {type(token)}')
            print(f'   {token.form}')
            if token.tag in km_dict.keys():
                print(f'   {token.tag} :  {km_dict[token.tag]}')
            else:
                print(f'    {token.tag} is not in the vocabulary')
            print(f'   start:  {token.start}')
            print(f'   end:    {token.end}')
            print(f'   length: {token.len}')
            
    def clean_contents(self, contents, kiwi, stopwords):
        contents_clean = self.clean_corpus(contents)
        token_kiwi = kiwi.tokenize(contents_clean, stopwords=stopwords)
        tokens = [token.form for token in token_kiwi]
        contents_clean = ' '.join(tokens)
        return contents_clean
    

def main():
    print(f'\n1.Load text')

    path = r'D:\naver_news_project\data'
    filename = '송영길_20231010.csv'
    filepath = os.path.join(path, filename)
    df = pd.read_csv(filepath)
    print(df.info())
    # display(df.head(3))


    print(f'\n2.Clean text')

#     kp = KNLP_Preprocessor()
    # # cleaning corpus
    # index = 0
    # contents = df['con문재tents'].to_list()[index]
    # print(f'\nLnegth of contents before cleaning : {len(contents)}')
    # contents_clean = kp.clean_corpus(contents)
    # print(f'Lnegth of contents after cleaning  : {len(contents_clean)}')
    # print(f'{contents_clean}')


    print(f'\n3.Tokenize')
    # load Korean morpheme dictionary
    kp = KNLP_Preprocessor()
    km_dict = kp.get_km_dict_kiwi(filepath=r'D:\naver_news_project\data\korean_morpheme_kiwi.pkl')
    print(len(km_dict))
    # print(km_dict)

    # initialize Kiwi instance
#     kiwi = Kiwi()
#     stopwords = Stopwords()
    # # stop_words = stopwords.stopwords 
    # # print(len(stop_words))
    # # print(stop_words)

    # token_kiwi = kiwi.tokenize(contents_clean, stopwords=stopwords)
    # # print(len(token_kiwi))
    # # print(token_kiwi[:10])
    # # parse_tokens_kiwi(token_kiwi[:10], km_dict)
    # tokens = [token.form for token in token_kiwi]
    # print(len(tokens))
    # print(tokens)

    # contents_clean = ' '.join(tokens)
    # print(contents_clean )

    print(f'\n4.Create Clean Corpus')

    kp = KNLP_Preprocessor()
    kiwi = Kiwi()
    stopwords = Stopwords()
    index = 0
    contents = df['contents'].to_list()[index]
    contents_clean = kp.clean_contents(contents, kiwi, stopwords=stopwords)
    print(contents_clean)

    df['contents_clean'] = df['contents'].apply(lambda x: kp.clean_contents(x, kiwi,\
                                                                            stopwords=stopwords)\
                                               if type(x) == str else x)

    path = r'D:\naver_news_project\data'
    filename = '송영길_news_20231010_clean.csv'
    filepath = os.path.join(path, filename)
    df.to_csv(filepath, index=False)
    print(f'"{filepath}" is saved.')
    df = pd.read_csv(filepath)
    print(df.info())
    print(df.head(3))
    print(f'\n======== The end of process ===========')
    
if __name__ == '__main__':
    main()
