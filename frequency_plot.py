
"""
frequency_plot.py

Author: CB Park
Date: Oct. 10, 2023
"""

import re
import pickle
import os
import pandas as pd
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
import numpy as np
import squarify
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set Seaborn style
sns.set_style('white')
# %config InlineBackend.figure_format='retina'
# The %config InlineBackend.figure_format='retina' command is used in Jupyter Notebook 
# or JupyterLab environments to enable high-resolution (retina-quality) plotting 
# for Matplotlib figures displayed inline. When you set the figure format to 'retina', 
# Matplotlib will render the plots at a higher resolution suitable for high-density displays 
# like those found in many modern laptops.

import matplotlib.font_manager as fm
fontpath = r'C:\Windows\Fonts\batang.ttc'
font = fm.FontProperties(fname=fontpath, size=10)
plt.rc('font', family='batang')


class FrequencyPlot:
    def __init__(self, freq_dict):
        # freq_dict: frequensy dataset as a dictionary
        self.freq_dict = freq_dict
        
    def plot_barh(self, figsize=(10, 22), fontsize=12):
        """
        plot horizontak bar
        """
        plt.rcParams['font.size'] = fontsize
        y_pos = np.arange(len(self.freq_dict))
        plt.figure(figsize=figsize)
        plt.barh(y_pos, self.freq_dict.values())
        plt.title('Word Count')
        plt.yticks(y_pos,self.freq_dict.keys() )
        plt.show()

    def plot_treemap(self, figsize=(6, 6), fontsize=12, alpha=0.7):
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = fontsize
        norm =  mpl.colors.Normalize(vmin=min(self.freq_dict.values()),
                                    vmax=max(self.freq_dict.values())
                                    )
        colors = [mpl.cm.Reds(norm(value)) for value in self.freq_dict.values()]
        squarify.plot(label=self.freq_dict.keys(),
                    sizes = self.freq_dict.values(),
                    color = colors,
                    alpha = alpha)
        plt.show()
        
    def plot_wordcloud(self, figsize=(6, 6), font_path=fontpath):
        wc = WordCloud(background_color='white', font_path=fontpath )
        wc.generate_from_frequencies(self.freq_dict)
        figure = plt.figure(figsize=figsize)
        ax = figure.add_subplot(1,1,1)
        ax.axis('off')
        ax.imshow(wc)
        plt.show()
        
        
        
from collections import Counter
def get_most_common_words(nouns, num=10):
    counts = Counter(nouns)
    most_common_words = dict(counts.most_common(num))
    return most_common_words

def remove_single_char_word(nouns):
    """
    # 1음절 단어 제거
    """
    final_noun_words = []
    for word in nouns:
        if len(word) > 1:
            final_noun_words.append(word)
    return final_noun_words

def collect_noun_words(filtered_text):
    """
    # 명사만 추출
    """
    kiwi = Kiwi()
    stopwords = Stopwords()
    nouns = []
    kiwi_tokens = kiwi.tokenize(filtered_text, stopwords=stopwords)
    for index, token in enumerate(kiwi_tokens):
        if 'NN' in token.tag:
            nouns.append(token.form)
    final_noun_words = remove_single_char_word(nouns)
    return final_noun_words

def main():
    print(f'\n1. Load Data')
    path = r'D:\naver_news_project\data'
    filename = '문재인_news_20231010_clean.csv'
    filepath = os.path.join(path, filename)
    df = pd.read_csv(filepath)
    # print(df.info())


    # 단어빈도 시각화
    # index =0 
    # filtered_text = df['contents_clean'][index]

    filtered_text = df['contents_clean'].to_list()
    filtered_text = [item for item in filtered_text if type(item)==str ]
    filtered_text = ' '.join(filtered_text)

    nouns = collect_noun_words(filtered_text)
    top_nouns = get_most_common_words(nouns, num=50)
    print(top_nouns)

    fp = FrequencyPlot(top_nouns)
    fp.plot_barh(figsize=(10, 22), fontsize=12)
    # 트리맵(Treemap) 시각화¶
    fp.plot_treemap(figsize=(6, 6), fontsize=12, alpha=0.7) 
    # 워드클라우드(WordCloud) 시각화
    fp.plot_wordcloud(figsize=(6, 6), font_path=fontpath)
    
if __name__ == '__main__':
    main()
