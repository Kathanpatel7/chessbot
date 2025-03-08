#!/usr/bin/env python
import csv

# Specify the path to your CSV file
csv_file = "/home/kathan/catkin_ws/src/Publisher/Final_coordinates.csv"  # Replace with your CSV file path

# The value to search for in the first column
search_value = "a1"

# Open and read the CSV file
with open(csv_file, newline='') as file:
    reader = csv.reader(file)
    
    found = False  # Flag to check if the value is found
    
    for row in reader:
        if row and row[0] == search_value:
            found = True
            print(f"Found {search_value} in the first column of the CSV file.")
            print(f"X coordinate =  {row[1]}, Y coordinate = {row[2]}")
            break

    if not found:
        print("The value 'e4' was not found in the CSV file.")

