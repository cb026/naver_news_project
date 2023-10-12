
"""
knlp_processor.py

Author: CB Park
Date  : Oct. 10, 2023
Update: Oct. 12, 2023
"""
import os
import re
import pickle
import string
import pandas as pd
import urllib.request
from tqdm import tqdm
from konlpy.tag import Okt
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec


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
    
    def tokenize_knlpy(self, stopwords, dataset, feature = 'document'):
        okt = Okt()
        tokenized_data = []
        for sentence in tqdm(dataset[feature]):
            tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
            stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
            tokenized_data.append(stopwords_removed_sentence)
        return tokenized_data

    def tokenize_kiwi(self, text):
        kiwi = Kiwi()
        stopwords = Stopwords()
        token_kiwi = kiwi.tokenize(text, stopwords=stopwords)
        tokens = [token.form for token in token_kiwi]
        return tokens

    def create_kiwi_tokenized_corpus(self, dataset, feature='document_clean'):
        tokenized_data = []
        for index, document in tqdm(enumerate(dataset[feature].to_list()), desc='tokenized_data'):
            if document:
                tokens = document.split()
                tokenized_data.append((index, tokens))
        return tokenized_data

def main():

    print(f'\n1.Load Data')

    # # 네이버 영화 리뷰 데이터를 다운로드합니다.
    # result = urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
    # # print(result)
    # # ('ratings.txt', <http.client.HTTPMessage object at 0x0000028070100350>)'

    # #영화 리뷰 데이터를 데이터프레임으로 로드하고 상위 5개의 행을 출력해봅시다.'
    # train_data = pd.read_table('ratings.txt')
    # print(train_data.info())

    path = r'D:\WordVector\word2vec_Korean\data'
    filename = 'naver_movie_reviews.csv'
    filepath = os.path.join(path, filename)
    # train_data.to_csv(filepath, index=False)
    train_data = pd.read_csv(filepath)
    # print(train_data.info())
    # # 총 20만개의 샘플이 존재하는데, 결측값 유무를 확인합니다.
    # # NULL 값 존재 유무
    # print(train_data.isnull().values.any())
    # #결측값이 존재하므로 결측값이 존재하는 행을 제거합니다.
    # train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
    # print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인


    print(f'\n2.Clean text')

#     kp = KNLP_Preprocessor()
#     # cleaning corpus
#     df = train_data.copy()
#     index = 0
#     contents = df['document'].to_list()[index]
#     print(f'\nLnegth of contents before cleaning : {len(contents)}')
#     contents_clean = kp.clean_corpus(contents)
#     print(f'Lnegth of contents after cleaning  : {len(contents_clean)}')
#     print(f'{contents}')
#     print(f'{contents_clean}')


    print(f'\n3.Tokenize')

    # load Korean morpheme dictionary
    # kp = KNLP_Preprocessor()
    # km_dict = kp.get_km_dict_kiwi(filepath=r'D:\naver_news_project\data\korean_morpheme_kiwi.pkl')
    # # print(len(km_dict))
    # # # print(km_dict)

#     # initialize Kiwi instance
#     kiwi = Kiwi()
#     stopwords = Stopwords()
#     # stop_words = stopwords.stopwords 
#     # print(len(stop_words))
#     # print(stop_words)

#     token_kiwi = kiwi.tokenize(contents_clean, stopwords=stopwords)
#     # print(len(token_kiwi))
#     # print(token_kiwi[:10])
#     # parse_tokens_kiwi(token_kiwi[:10], km_dict)
#     tokens = [token.form for token in token_kiwi]
#     print(len(tokens))
#     print(tokens)

#     contents_clean = ' '.join(tokens)
#     print(contents_clean )

    print(f'\n4.Create Clean Corpus')
    # CPU times: total: 3min 24s
    # Wall time: 3min 27s

    # kp = KNLP_Preprocessor()
    # kiwi = Kiwi()
    # stopwords = Stopwords()

    # df = train_data
    # df['document_clean'] = df['document'].apply(lambda x: kp.clean_contents(x, kiwi,\
    #                                                                         stopwords=stopwords)\
    #                                            if type(x) == str else x)

    path = r'D:\WordVector\word2vec_Korean\data'
    filename = 'naver_moview_review_20231011_clean.csv'
    filepath = os.path.join(path, filename)
    # df.to_csv(filepath, index=False)
    # print(f'"{filepath}" is saved.')
    df = pd.read_csv(filepath)
    df.dropna(subset=['document_clean'],inplace=True)
    print(df.info())
    display(df.head(3))


    print(f'\n5.Create Tokenized Data wit knlpy ')
    # # CPU times: total: 3min 6s
    # # Wall time: 2min 34s

    # kp = KNLP_Preprocessor()
    # stop_words = Stopwords().stopwords
    # tokenized_data_knlpy = kp.tokenize_knlpy(stop_words, df, feature = 'document_clean')

    # print(len(tokenized_data_knlpy))
    # print(tokenized_data_knlpy[0])


    print(f'\n6.Create Tokenized Data wit kiwipiepy ')
    data = kp.create_kiwi_tokenized_corpus(df, feature='document_clean')
    print(len(data))
    print(data[0])

    tokenized_data_kiwi = [item[1] for item in data]
    print(len(tokenized_data_kiwi))
    print(tokenized_data_kiwi[0])

if __name__ == '__main__':
    main()
