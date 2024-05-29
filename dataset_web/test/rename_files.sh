#!/bin/bash

# Initialize a counter
counter=0

# Iterate through each .png file in the folder
for file in *.png; do
    # Create the new filename with the counter
    new_filename="$counter.png"

    # Rename the file
    mv "$file" "$new_filename"

    # Increment the counter
    ((counter++))
done
