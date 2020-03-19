# I. Create Gemstones Dataset
> FULL DATASET OF GEMSTONES IMAGES CAN BE FOUND AT [MY KAGGLE PAGE](https://www.kaggle.com/lsind18/gemstones-images): it's already divided into train and test data. This dataset contains 3,000+ images of different gemstones.

## 1. Download gemstones images
Example of fetching data from different sources in 2 ways: scraping static content and scraping dunamic content.

Install packages:  
```Console
pip install bs4 
pip install requests
pip install selenium
```
*Beautiful Soup (bs4)* = Python library for pulling data out of HTML and XML files.  
*requests* = HTTP library  
*selenium* = bindings for Selenium WebDriver; automate web browser interaction from Python.  

### [Parse static content](1_Fetch_data/fetch_data.py):
Example of scraping [minerals.net](https://www.minerals.net) website:
1. Get the page with [list of all gemstones](https://www.minerals.net/GemStoneMain.aspx) and find HTML-element with them:
```python
url = 'https://www.minerals.net/GemStoneMain.aspx'
html = requests.get(url).text
soup = bs(html, 'html.parser')
table_gems=soup.find_all('table',{'id':'ctl00_ContentPlaceHolder1_DataList1'})
```
2. Parse links to the pages of each gemstone and create dictionary `{gemstone name : link }`

3. Parse each page to get pictures of gemstones
```python
table_images=soup.find_all('table',{'id':'ctl00_ContentPlaceHolder1_DataList1'})
```

### [Parse dynamic content](1_Fetch_data/fetch_dyn_data.py) using `selenium`:
Example of scraping [www.rasavgems.com](https://www.rasavgems.com) website. The website uses javascript.
1. Import webdriver
```python
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
```
2. Download web driver for you browser and include it into code 
```python
wd = webdriver.Chrome("....\\chromedriver_win32\\chromedriver.exe") 
```
3. Get data using automatic interaction
```python
element = wait.until(
        EC.presence_of_element_located((By.ID, "Product_List")))
        imgs_form = wd.find_element_by_id('Product_List')
```

We created one folder with subfolders with gemstones pictures inside. They can be checked manually: finally I got 87 classes of gemstones.

## 2. Create a dataset for NN: Train and Test sets

### [Rename images](https://github.com/LSIND/Gemstones-Neural-Network/blob/master/1_Fetch_data/2_Rename_Files.py) in created folders

1. Using `os` module rename files in every folder with name `FolderName_N.extension`. For example, for Ametrine folder rename files as `ametrine_0.jpg`, `ametrine_1.jpg` etc.

2. Avoid common problems: folder iss empty or file with such name already exists.
```Console
...
input\\Chalcedony
CCOV1520CSCABIOCCMLT-1003_1.jpg                     --> chalcedony_0.jpg
chalcedony-round-multi-color-ccmlt500x500.jpg       --> chalcedony_1.jpg
....
input\\Labradorite
labradorite-gem-253954a.jpg                         --> labradorite_0.jpg
...
```

### [Split images](https://github.com/LSIND/Gemstones-Neural-Network/blob/master/1_Fetch_data/2_Split_to_Train_Test.py) into train and test data: 90% : 10%
> Train set is used to teach a neural network. Test set is used to check if the neural network understand a gemstone or not.

1. Use [split_folders](https://pypi.org/project/split-folders/)  
`pip install split_folders`   

This library provides splitting folders with files into train, validation and test (dataset) folders. Split with a ratio to only split into training and validation set `(.9, .1)`.

```python
split_folders.ratio('input', output="data", seed=1337, ratio=(.9, .1))
```
2. [Check number of files and folders](https://github.com/LSIND/intro-to-python3-analysis/tree/master/CountFilesAndFolders/main.py):
```Console
------------> gemstones-images:  2 folders,  0 files
--------------> test :   87 folders,  0 files
----------------> Alexandrite :  0 folders,  4 files
----------------> Almandine :    0 folders,  4 files
...
...
--------------> train :  87 folders,  0 files
----------------> Alexandrite :  0 folders,  31 files
----------------> Almandine :    0 folders,  29 files
----------------> Amazonite :    0 folders,  28 files
```

**FULL DATASET OF GEMSTONES IMAGES CAN BE FOUND AT [MY KAGGLE PAGE](https://www.kaggle.com/lsind18/gemstones-images): it's already divided into train and test data. This dataset contains 3,000+ images of different gemstones.**  
There are 87 classes of different gemstones. The images are in various sizes and are in .jpg format. All gemstones have various shapes - round, oval, square, rectangle, heart.  
`Each class in train set contains 27 - 48 images, in test set - 4 - 6 images.`
