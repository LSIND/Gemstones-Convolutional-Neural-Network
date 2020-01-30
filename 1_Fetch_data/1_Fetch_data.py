#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Get images of gemstones from minerals.net'''

import requests
from bs4 import BeautifulSoup as bs
import shutil
import os

global gems
gems = 'https://www.minerals.net/'

def get_gemstones_names(ta):
    '''get gemstones names and links to pages
    Returns a dictionary {gemstone : link }'''
    names_paths = {}
    for data in ta:
        for link in data.find_all('a'):
            if(link.text!=''):
                names_paths[link.text] = link.get('href')
    return names_paths


def download_images(key, value):
    img_link = ''
    print('-'*8, key, '-'*8)
    gem_link = gems+value
    print(gem_link)
    html_gems = requests.get(gem_link).text
    soup = bs(html_gems, 'html.parser')
    table_images=soup.find_all('table',{'id':'ctl00_ContentPlaceHolder1_DataList1'})
    if(table_images):
        if not os.path.exists(key):
            os.mkdir(key)

        for data in table_images:
            for link in data.find_all('img'):
                img_link = gems+link.get('src').replace('-t','')
                print(img_link)
                r = requests.get(img_link, stream=True)
                if r.status_code == 200:    # OK
                    with open(key+'/'+img_link.split('/')[-1], 'wb') as f:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)
    else:
        print('no images found for', key)

if __name__ == '__main__':
    url = gems + 'GemStoneMain.aspx'
    html = requests.get(url).text
    soup = bs(html, 'html.parser')
    table_gems=soup.find_all('table',{'id':'ctl00_ContentPlaceHolder1_DataList1'})
    gem_dict = get_gemstones_names(table_gems)
    for key, value in gem_dict.items():
        download_images(key, value)
