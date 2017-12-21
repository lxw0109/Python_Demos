# -*- coding: utf-8 -*-
import scrapy


class XpathExpSpider(scrapy.Spider):
    name = "xpath_exp"
    allowed_domains = ["192.168.1.236"]
    # start_urls = ["http://192.168.1.236/"]
    url = "http://192.168.1.236/liuxiaowei/m.html"

    def start_requests(self):
        yield scrapy.Request(url=self.url, callback=self.parse, method="GET")


    def parse(self, response):
        print(response.text)
