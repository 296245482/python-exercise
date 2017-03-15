import urllib
import urllib2
import re
import os

class Spider:

    def __init__(self):
        self.siteURL = 'http://vision.middlebury.edu/stereo/data/scenes2014/zip/'
        self.fileUrls = []
 
    def getContent(self, url):
        request = urllib2.Request(url)
        response = urllib2.urlopen(request)
        return response.read()
 
    def findFileUrls(self, htmlCode):
        pattern = re.compile(r'\s]"></td><td><a\shref="(.*?)">.*?zip</a>', re.S) # re.S
        match = re.findall(pattern, htmlCode)
        if match is None:
            return None
        else:
            for url in match:
                fileName = url
                url = self.siteURL + url
                with open("files", "ab") as outputFile:
                    outputFile.write(url + "\n")

 
spider = Spider()
siteURL = 'http://vision.middlebury.edu/stereo/data/scenes2014/zip/'
htmlCode = spider.getContent(siteURL)
spider.findFileUrls(htmlCode)