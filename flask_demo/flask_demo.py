#!/usr/bin/env python3
# coding: utf-8
# File: flask_demo.py
# Author: lxw
# Date: 11/13/17 3:36 PM


from flask import Flask
from flask import request
from flask import url_for

app = Flask(__name__)

@app.route("/")
def index():
    return "Welcome!"

@app.route("/hello/")    # NOTE: 最好所有的路由都以"/"结尾
def hello():
    return "Hello world!"

@app.route("/user/<username>/")
def show_username(username):
    return "User {}".format(username)

@app.route("/post/<int:post_id>/")
def show_post(post_id):
    return "Post {}".format(post_id)

@app.route("/login/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        return _do_the_login()
    else:
        return _show_login_form()

def _do_the_login():
    return "in _do_the_login()"

def _show_login_form():
    return "in _show_login_form()"

if __name__ == '__main__':
    with app.test_request_context():
        print(url_for("index"))
        print(url_for("hello"))
        print(url_for("show_username", username="lxw"))
        print(url_for("show_post", post_id=-10))
        print(url_for("static", filename="style.css"))


