# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 18:23:37 2017

@author: ypc
"""

import re
import os
import codecs
import tarfile
import json as js

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
english_stopwords = stopwords.words('english')


def clean_math(string):
    while string.count('$') > 1:
        pos0 = string.find('$')
        pos1 = string.find('$', pos0+1)
        string = (string[:pos0] + string[pos1+1:]).strip()
    return string
        

def clean_str(string):
    """
    Input:
        string: One line in a latex file.
    Return：
        string cleaned.
    """
   
    # Remove mathematical formulas between $$
    string = clean_math(string)

    # Remove "ref" 
    string = re.sub(r'~(.*)}', '', string)
    string = re.sub(r'\\cite(.*)}', '', string)
    string = re.sub(r'\\newcite(.*)}', '', string)
    string = re.sub(r'\\ref(.*)}', '', string)
    
    # Remove stopwords
    texts_tokenized = [word.lower() for word in word_tokenize(string)]
    texts_filtered_stopwords = [word for word in texts_tokenized if not word in english_stopwords]
    string = ' '.join(texts_filtered_stopwords)
    string = string.replace(',', '')
    string = string.replace('.', '')
    string = string.replace('?', '')
    string = string.replace('!', '')
    string = string.replace('/', '')
    string = string.replace('$', '')
    string = string.replace('~', '')
    string = string.replace('\\', '')
    string = string.replace('{', '')
    string = string.replace('}', '')
    string = string.replace('#', '')
    string = string.replace('&', '')
    string = string.replace('@', '')
    string = string.replace('%', '')
    string = string.replace('^', '')
    string = string.replace('*', '')
    string = string.replace('-', '')
    string = string.replace('=', '')
    string = string.replace('[', '')
    string = string.replace(']', '')
    string = string.replace('+', '')
    string = string.replace('(', '')
    string = string.replace(')', '')
    return string
    
    
def process_text_list(text_list):
    """
    Input:
        text_list: Content of a latex file and each element represents a line.
    Return:
        A list, which is the cleaned content of a latex file.
    """

    result = ''
    for line in text_list:
        line = line.strip()
        if line.startswith('%') or line.startswith('\\') or line == '':
            pass
        elif line[0].isdigit():
            pass
        else:
            result += clean_str(line)
    return result


# Extract Introduction, related work, etc.================================================================
def split(tex_list, start_char, end_char):
    lines = tex_list
    length = len(lines)
    start = None
    end = None
    i = 0
    while i < length and (end is None):
        if start is None:
            if lines[i].startswith(start_char):
                start = i + 1
        else:
            if lines[i].startswith(end_char):
                end = i
        i += 1
    if (start is not None) and (end is None):
        end = length
    return lines[start:end]


def extract(tex_list, segment=False):
    data = tex_list
    text = ' '.join(data)
    intro = ' '.join(split(tex_list, '\section{Intro', '\section{'))
    related = ' '.join(split(tex_list, '\section{Related', '\section{'))
    conclusion = ' '.join(split(tex_list, '\section{Conclu', '\section{'))
    methods = text.replace(intro, '').replace(related, '').replace(conclusion, '')
    if segment:
        pass
    else:
        return list(map(process_text_list, 
                    [intro.split('\n'), related.split('\n'), methods.split('\n'), conclusion.split('\n')]))


def main(file_dir):
    result = {}
    file_names = os.listdir(file_dir)
    for file_name in file_names:
        try:
            f_name = os.path.join(file_dir, file_name)
            tex_list = make_single_tex(f_name)
            result[file_name] = extract(tex_list)
        except:
            continue
    return result
    