#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Get images of gemstones from www.rasavgems.com'''

import requests
from bs4 import BeautifulSoup as bs
import shutil
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

global gems
gems = "https://www.rasavgems.com/"


def get_gemstones_names(ta):
    '''get gemstones names and links to pages
    Returns a dictionary {gemstone : link }'''
    names_paths = {}
    for data in ta:
        for link in data.find_all('a'):
            if(link.text!=''):
                names_paths[link.text.strip()] = link.get('href')
    return names_paths


def download_images(key, value):
    img_link = ''
    print('-'*8, key, '-'*8)
    gem_link = gems+value
    print(gem_link)

    wd = webdriver.Chrome("C:\\Users\\DVChemkaeva\\Downloads\\chromedriver_win32\\chromedriver.exe")
    wd.get(gem_link)
    wait = WebDriverWait(wd,15)
    try:
        element = wait.until(
        EC.presence_of_element_located((By.ID, "Product_List")))
        imgs_form = wd.find_element_by_id('Product_List')
    finally:
        html_gems = wd.page_source
        wd.quit()

    soup = bs(html_gems, 'html.parser')
    ta=soup.find_all('div', {'class','product-one'})

    if(ta):
        if not os.path.exists(key):
            os.mkdir(key)
        for data in ta:
            for link in data.find_all('img'):
                img_link = link.get('src').replace('../','')
                print(img_link)
                r = requests.get(img_link, stream=True)
                if r.status_code == 200:
                    src_img = key+'/'+img_link.split('/')[-1]
                    with open(src_img, 'wb') as f:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)

if __name__ == '__main__':
    html = requests.get(gems).text
    soup = bs(html, 'html.parser')
    ta=soup.find_all('ul',{'id':'leftNavigation'})
    gem_dict = get_gemstones_names(ta)
    for key, value in gem_dict.items():
        download_images(key, value)
