import re

input_file = 'input.txt'
output_file = 'output.txt'

# Read the input file
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the unwanted pattern using regular expression
cleaned_content = re.sub(r'{{.*?}}', '', content)

# Write the cleaned content to a new file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(cleaned_content)