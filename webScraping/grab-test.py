#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys
import urllib2

filesystemencoding = sys.getfilesystemencoding()

url = "http://www.baidu.com"

request = urllib2.Request(url)
response = urllib2.urlopen(request)
htmlCode = response.read().decode("utf-8", "ignore")
print htmlCode.encode(filesystemencoding, "ignore")


# 对htmlCode 进行正则表达式匹配，查找对应内容
# xxxxxxxxx