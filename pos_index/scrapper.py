import requests
import time
from bs4 import BeautifulSoup
import sys
import os 

# create ECT and store html files into it 
out_dir = 'ECT'
out_dir = os.path.join(os.getcwd(), out_dir)
if not os.path.exists(out_dir):
	os.mkdir(out_dir)


def get_date(c):
    end = c.find('|')
    return c[0:end-1]

def get_ticker(c):
    beg = c.find('(')
    end = c.find(')')
    return c[beg+1:end]

def grab_page(url):
    print("attempting to grab page: " + url)
    page = requests.get(url)
    page_html = page.text
    
    soup = BeautifulSoup(page_html, 'html.parser')

    meta = soup.find("div",{'class' : 'a-info clearfix'})
    content = soup.find(id="a-body")

    if meta is None or content is None:
        print("skipping this link, no content here")
        return
    else:
        text = content
        mtext = meta.text

        filename = get_ticker(mtext) + "_" + get_date(mtext)
        file = open(os.path.join(out_dir, filename.lower() + ".html"), 'w')
        file.write(str(text))
        file.close
        # print(filename.lower()+ " sucessfully saved")


def process_list_page(i):
    origin_page = "https://seekingalpha.com/earnings/earnings-call-transcripts" + "/" + str(i)
    print("getting page " + origin_page)
    page = requests.get(origin_page)
    page_html = page.text
    #print(page_html)
    soup = BeautifulSoup(page_html, 'html.parser')
    alist = soup.find_all("li",{'class':'list-group-item article'})
    for i in range(0,len(alist)):
        url_ending = alist[i].find_all("a")[0].attrs['href']
        url = "https://seekingalpha.com" + url_ending
        grab_page(url)
        time.sleep(.5)

for i in range(1,200): #choose what pages of earnings to scrape
    process_list_page(i)