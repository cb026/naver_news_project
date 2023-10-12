# %%writefile naver_news_collector.py

# CPU times: total: 3min 10s
# Wall time: 24min 45s

"""
naver_news_collector.py
Author: CB Park
Date: Oct. 9 , 2023
"""

import re
import os
import sys
import json
import urllib.request
import pandas as pd
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from pprint import pprint

client_id = os.environ['naver_client_id']
client_secret = os.environ['naver_client_secret'] 


class NaverNews:
    def __init__(self, query):
        self.query = query

    def collect_naver_news(self, category='news', display=100, \
                           start=1, end =1000, sort="sim" ):
        """
        query :검색어. UTF-8로 인코딩되어야 합니다.
        options
        display: 한 번에 표시할 검색 결과 개수(기본값: 10, 최댓값: 100)
        start: 검색 시작 위치(기본값: 1, 최댓값: 1000)
        sort: 검색 결과 정렬 방법
        - sim: 정확도순으로 내림차순 정렬(기본값)
        - date: 날짜순으로 내림차순 정렬

        ref: https://developers.naver.com/docs/serviceapi/search/news/news.md#%EB%89%B4%EC%8A%A4-%EA%B2%80%EC%83%89-%EA%B2%B0%EA%B3%BC-%EC%A1%B0%ED%9A%8C
        Author: CB Park
        Date: Oct. 10, 2023
        """
        query_parsed = urllib.parse.quote(self.query)
        base_url = "https://openapi.naver.com/v1/search"

        idx = 0
        news_df = pd.DataFrame(columns=('title', 'original link', 'link', \
                                        'description', 'publication date'))
        for start_index in range(start, end, display):
            url = f'{base_url}/{category}?query={query_parsed}'\
                  f'&display={str(display)}'\
                  f'&start={str(start_index)}'\
                  f'&sort={sort}'        
            print(f"{idx+1} Requesting URL: {url}")  # Print the URL for debugging
            request = urllib.request.Request(url)
            request.add_header("X-Naver-Client-Id",client_id)
            request.add_header("X-Naver-Client-Secret",client_secret)
            response = urllib.request.urlopen(request)
            rescode = response.getcode()

            if(rescode==200):
                response_body = response.read()
                response_dict = json.loads(response_body.decode('utf-8'))
                items = response_dict['items']
                for item_index in range(0, len(items)):
                    remove_tag = re.compile('<.*?>')
                    title = re.sub(remove_tag, '', items[item_index]['title'])
                    original_link = items[item_index]['originallink']
                    link = items[item_index]['link']
                    description = re.sub(remove_tag, '', items[item_index]['description'])
                    pub_date = items[item_index]['pubDate']
                    news_df.loc[idx] = [title, original_link, link, description, pub_date]
                    idx +=1
            else:
                print("Error Code:" + rescode)
        return news_df


class NewsCollector():
    def get_naver_links(self, dataset, feature='link'):
        # collect naver links for the news
        naver_links = []
        for row in dataset.iterrows():
            link = row[1][feature]
            naver_links.append(link)
        return naver_links

    def get_soup(self, url):
        soup =None
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
        else:
            print(f'Error: {response.status_code}')
        return soup

    def get_news_contents(self, soup):
        article = soup.find('article', class_="go_trans _article_content")
        if article:
            contents = article.text
        else:
            contents = None
            print(f"Error: couldn't get the contents")
        return contents

    def create_news_contents_dataset(self, func, naver_links):
        print(f'function: {func}')
        index = 0
        df = pd.DataFrame(columns = ['link', 'contents', 'label'])
        for url in naver_links:
            print(f'{index+1}: {url}')
            news_contents = None
            try:
                soup = self.get_soup(url)
                news_contents = func(soup)
                label = 1
            except Exception as e:
                label = 0
                print(f'Error: {e}')
            link = url
            contents = news_contents
            df.loc[index] = [link, contents, label]
            index +=1
        return df    
    
    def create_news_contents_dataset_2(self, naver_links):
        index = 0
        df = pd.DataFrame(columns = ['link', 'contents', 'label'])
        for url in naver_links:
            print(f'{index+1}: {url}')
            news_contents = None
            try:
                soup = self.get_soup(url)
                if 'etoday' in url:
                    func = self.get_contents_etoday
                elif 'newspim' in url:
                    func = self.get_contents_newspim
                elif 'asiatoday' in url:
                    func = self.get_contents_asiatoday  
                elif 'ajunews' in url:
                    func = self.get_contents_ajunews 
                elif 'newstomato' in url:
                    func = self.get_contents_newstomato  
                elif 'upinews' in url:
                    func = self.get_contents_upinews  
                elif 'shinailbo' in url:
                    func = self.get_contents_shinailbo     
                elif 'newscj' in url:
                    func = self.get_contents_newscj         
                elif 'ilyo' in url:
                    func = self.get_contents_ilyo          
                elif 'bbsi' in url:
                    func = self.get_contents_bbsi       
                elif 'seoulwire' in url:
                    func = self.get_contents_seoulwire               
                elif 'getnews' in url:
                    func =  self.get_contents_getnews                      
                elif 'sisafocus' in url:
                    func =  self.get_contents_sisafocus            

                news_contents = func(soup)
                label = 1
            except Exception as e:
                label = 0
                print('Error')
            link = url
            contents = news_contents
            df.loc[index] = [link, contents, label]
            index +=1
        return df
    
    def get_contents_shinailbo(self, soup):
        try:
            contents = soup.find('div', class_='user-content').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_upinews(self, soup):
        try:
            contents = soup.find('div', class_='viewConts').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_newstomato(self, soup):
        try:
            contents = soup.find('div', class_='rns_text').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_ajunews(self, soup):
        try:
            contents = soup.find('div', class_='article_con').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_asiatoday(self, soup):
        try:
            contents = soup.find('div', id='section_main').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_newspim(self, soup):
        try:
            contents = soup.find('div', class_='contents').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_etoday(self, soup):
        try:
            contents = soup.find('div', class_='articleView').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_newscj(self, soup):
        try:
            contents = soup.find('div', class_='article-body').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_ilyo(self, soup):
        try:
            contents = soup.find('div', class_='articleView').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_bbsi(self, soup):
        try:
            contents = soup.find('div', class_='article-body').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_seoulwire(self, soup):
        try:
            contents = soup.find('div', class_='article-body').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_getnews(self, soup):
        try:
            contents = soup.find('div', class_='article-body').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_sisafocus(self, soup):
        try:
            contents = soup.find('div', class_='article-body').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_gukjenews(self, soup):
        try:
            contents = soup.find('div', class_='article-body').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_newdaily(self, soup):
        try:
            contents = soup.find('div', id = 'article_conent').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_kyeongin(self, soup):
        try:
            contents = soup.find('div', class_='news_content').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_sisaweek(self, soup):
        try:
            contents = soup.find('article', id ='article-view-content-div').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_newstomato(self, soup):
        try:
            contents = soup.find('div', class_='rns_text').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_hankooki(self, soup):
        try:
            contents = soup.find('div', class_='article-body').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_kmaeil(self, soup):
        try:
            contents = soup.find('div', id='article-view-content-div').text
        except:
            contents = None
            print('Error')
        return contents

    def get_contents_pennmike(self, soup):
        try:
            contents = soup.find('article', id='article-view-content-div').text
        except:
            contents = None
            print('Error')
        return contents

def get_news_source(base_url):
    if 'naver' in base_url:
        source = 'naver'
    else:
        source = base_url.split('.')[1]
    return source


def main():
    query = '시행령'#'탄핵'#"예산" #'인공지능'#'국민의힘'#'더불어민주당'#"선거"#"윤석열"

    #'이재명'#'국정조사'#'송영길'#'한동훈'#'문재인'#'친명계'#'비명계' #'유인촌'#'김행'#'홍범도'#'이태원참사'
    #'국민의힘'#'더불어민주당'#'검찰'#'윤석열'#'언론탄압'#'수박'#'선거' #'문재인' #'검찰'#'국민의힘'#'더불어민주당'
    date = '20231011'
    path = r'D:\naver_news_project\data'
    filename = f'{query}_{date}.csv'
    filepath = os.path.join(path, filename)
    print(filepath)


    print(f'\n1. Get Naver News List:')
    nn = NaverNews(query)
    news_df = nn.collect_naver_news(category='news', display=100, start=1, end =1000, sort="sim" )
    # print(news_df.info())
    # display(news_df)


    print(f'\n2. Get Naver News Contents: 1st trial')
    nc = NewsCollector()
    naver_links = nc.get_naver_links(dataset=news_df, feature='link')
    # print(len(naver_links))

    func = nc.get_news_contents
    contents_df = nc.create_news_contents_dataset(func, naver_links)
    # print(len(contents_df))
    # display(contents_df)

    print(f'\n3. Get Naver News Contents: 2nd trial')
    news_df = news_df.drop_duplicates(subset=['link'])
    contents_df = contents_df.drop_duplicates(subset=['link'])
    mdf = pd.merge(news_df, contents_df, how='inner', on='link')
    # print(mdf.info())
    missed_df = mdf[mdf['contents'].isna()]
    # print(missed_df.info())
    missed_links = missed_df['link'].to_list()
    # print(missed_links)
    contents2_df = nc.create_news_contents_dataset_2(missed_links)
    contents2_df['index'] = missed_df.index
    # print(contents2_df.info())

    print(f'\n4. Final Dataset and Save it to a file')
    # add missed contents to the news dataset
    for row in contents2_df.iterrows():
        idx = row[1]['index']
        contents = row[1]['contents']   
        mdf.loc[idx, ['contents']] = contents


    # Add news source
    mdf['base_url'] = mdf['link'].apply(lambda x: x.split('/')[2])
    mdf['source'] = mdf['base_url'].apply(lambda x: get_news_source(x))

    mdf.to_csv(filepath, index=False)
    mdf.to_csv(filepath, index=False)
    mdf = pd.read_csv(filepath)
    print(f'"{filepath}" is saved')
    print(mdf.info())

if __name__ == '__main__':
    main()
