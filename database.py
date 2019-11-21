import mysql.connector

def create():
    mycursor.execute("CREATE DATABASE opencv")
    mycursor.execute("use opencv")
    mycursor.execute("create table image_properties(id INT AUTO_INCREMENT PRIMARY KEY, path  VARCHAR(255) NOT NULL, size VARCHAR(255) NOT NULL, type VARCHAR(255) NOT NULL)")
    mycursor.execute("create table features(fid INT AUTO_INCREMENT PRIMARY KEY, skin_tone VARCHAR(255) NOT NULL, hair_color VARCHAR(255) NOT NULL, category VARCHAR(255) NOT NULL, location VARCHAR(255) NOT NULL, upper_color VARCHAR(255) NOT NULL, lower_color VARCHAR(255) NOT NULL, id INT NOT NULL)")
    mycursor.execute("create table suggestions(color_ub  VARCHAR(255) NOT NULL, color_lb VARCHAR(255) NOT NULL, fid INT NOT NULL)")

def delete():
    mycursor.execute("drop database opencv")

mydb = mysql.connector.connect(host="localhost", user="root", passwd="abcd1234")

mycursor = mydb.cursor()
if(int(input("enter 1 : create, 2 : delete : "))==1):
    create()
else:
    delete()
