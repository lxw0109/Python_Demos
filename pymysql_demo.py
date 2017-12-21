#!/usr/bin/env python3
# coding: utf-8
# File: pymysql_demo.py
# Author: lxw
# Date: 10/10/17 11:03 PM
# Reference: http://www.runoob.com/python3/python3-mysql.html


import pymysql


def main():
    HOST = "192.168.1.41"
    USER = "lxw"
    PASSWD = "lxw0109"
    DB_NAME = "TESTDB"

    db = pymysql.connect(HOST, USER, PASSWD, DB_NAME)
    cursor =db.cursor()

    """
    # 1. fetchone()
    cursor.execute("SELECT VERSION()")

    data = cursor.fetchone()
    print("Database Version:{}".format(data))
    """
    """
    # 2. create database
    cursor.execute("DROP TABLE IF EXISTS EMPLOYEE")
    sql = "" "CREATE TABLE EMPLOYEE(
      FIRST_NAME CHAR(20) NOT NULL,
      LAST_NAME CHAR(20),
      AGE INT,
      SEX CHAR(1),
      INCOME FLOAT
    )
    "" "
    cursor.execute(sql)
    """
    """
    # 3. insert data
    sql = "" "INSERT INTO EMPLOYEE(FIRST_NAME, LAST_NAME, AGE, SEX, INCOME)
        VALUES ("Xiaolong", "Wang", 30, "M", 10000)
    "" "
    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        db.rollback()
        print(e)
    """
    """
    # 4. retrieve
    sql = "" "SELECT * FROM EMPLOYEE WHERE INCOME > {}"" ".format(8000)

    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            print("row:{}".format(row))
            fname = row[0]
            lname = row[1]
            age = row[2]
            sex = row[3]
            income = row[4]
            print("first_name:{0}, last_name:{1}, age:{2}, sex:{3}, income:{4}".format(fname, lname, age, sex, income))
    except Exception as e:
        print("Error: unable to fetch data. {}".format(e))
    """

    """
    # 5. update
    sql = "" "UPDATE EMPLOYEE SET AGE=AGE+1 WHERE SEX="M" "" "
    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        db.rollback()
        print(e)
    """

    # 6. delete
    sql = "DELETE FROM EMPLOYEE WHERE AGE > 30"
    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        db.rollback()
        print(e)

    db.close()


if __name__ == "__main__":
    main()