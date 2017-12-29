#!/usr/bin/env python3
# coding: utf-8
# File: configparser_demo.py
# Author: lxw
# Date: 8/9/17 2:26 PM

import configparser


def config_write():
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'ServerAliveInterval': '45', 'Compression': 'yes', 'CompressionLevel': '9'}
    config['DEFAULT']['ForwardX11'] = 'yes'

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
    print("config.sections():{}\n".format(config.sections()))

    # case-sensitive
    # print(config["default"])    # KeyError: 'default
    print(config["DEFAULT"])    # <Section: DEFAULT>
    for key in config["DEFAULT"]:
        print(key)
    print("")

    # case-sensitive
    topsecret = config['topsecret.server.com']
    # topsecret = config['tOPSECRET.server.com']    # KeyError: 'tOPSECRET.server.com
    # topsecret = config['TOPSECRET.SERVER.COM']    # KeyError: 'TOPSECRET.SERVER.COM

    # case-insensitive
    print(topsecret['FORWARDX11'])
    print(topsecret['ForWARDX11'])
    print(topsecret['FOrWaRdX11'])
    print(topsecret['FoRWaRdx11'])
    print(topsecret['foRwardX11'], end="\n\n")

    # NOTE: including those specified in "DEFAULT".
    for key in config['bitbucket.org']:
        print(key)



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

