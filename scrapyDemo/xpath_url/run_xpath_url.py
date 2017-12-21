#!/usr/bin/env python3
# coding: utf-8
# File: run_xpath_url.py
# Author: lxw
# Date: 12/1/17 3:48 PM

from scrapy import cmdline


cmdline.execute("scrapy crawl xpath_exp -L WARNING".split())
