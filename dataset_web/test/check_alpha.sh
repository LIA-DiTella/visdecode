for file in *.png; do
    if identify -format '%[channels]' "$file" | grep -q 'a'; then
        echo "$file has transparency."
    else
        echo "$file does not have transparency."
    fi
done
