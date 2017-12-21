#!/usr/bin/env python3
# coding: utf-8
# File: configparser_demo.py
# Author: lxw
# Date: 8/9/17 2:26 PM

import configparser


def config_write():
    config = configparser.ConfigParser()
    config['default'] = {'ServerAliveInterval': '45', 'Compression': 'yes', 'CompressionLevel': '9'}
    config['default']['ForwardX11'] = 'yes'

    config['bitbucket.org'] = {}
    config['bitbucket.org']['User'] = 'hg'

    config['topsecret.server.com'] = {}
    topsecret = config['topsecret.server.com']
    topsecret['Port'] = '50022'  # mutates the parser
    topsecret['ForwardX11'] = 'no'  # same here

    with open('example.ini', 'w') as configfile:
        config.write(configfile)


def config_read():
    config = configparser.ConfigParser()
    print("config.sections():{}".format(config.sections()))
    config.read('example.ini')
    print("config.sections():{}".format(config.sections()))
    topsecret = config['topsecret.server.com']
    for key in config['bitbucket.org']:
        print(key)
    print(config['bitbucket.org']['ForwardX11'])



def main():
    config = configparser.ConfigParser()
    config["redis"] = {
        "host": "192.168.1.131",
        "port": 6579,
        "key": "cnki_patent"
    }
    config["mongodb"] = {
        "host": "192.168.1.36",
        "port": 27017,
        "db": "scrapy",
        "collection": "cjo0620"
    }
    config["redis"]["author"] = "lxw"
    config["mongodb"]["author"] = "lxw"

    with open("./example.ini", "w") as config_file:
        config.write(config_file)


if __name__ == "__main__":
    # main()
    config_write()
    config_read()

