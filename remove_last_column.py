import csv

# filepath: c:\Users\Kevin\Desktop\actCat-c\timeless.csv
input_file = 'timeless.csv'
output_file = 'timeless_cleaned.csv'

with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        # Remove the last column
        writer.writerow(row[:-1])

print(f"Last column removed. Cleaned file saved as {output_file}.")