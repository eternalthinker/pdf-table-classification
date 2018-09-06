import sys
import re
import os
import glob


re_tags_with_attrs = re.compile(r"(<[a-z]+) .*?(/?>)")

def clean_table(filepath):
    basename = os.path.basename(filepath)
    filename, ext = os.path.splitext(basename)

    with open(filepath, "r") as table_file, open("{}.txt".format(filename), "w") as clean_table_file:
        content = table_file.read()

        # Clean tag attributes and separate with space
        content = re.sub(re_tags_with_attrs, r"\1\2", content)
        content = re.sub(r"(<)", r" \1", content)
        content = re.sub(r"(>)", r"\1 ", content)
        # Trim delimiters around edges
        content = content[1:-1]

        # Clean up numbers
        content = re.sub(r"[\(]([0-9]+[,\.]?)+[0-9]+[\)]", r"(NUMERICAL_DATA)", content)
        content = re.sub(r" ([0-9]+[,\.])+[0-9]+ ", r" NUMERICAL_DATA ", content)

        # Remove tags
        # content = re.sub(r"<.*?>", r"", content)

        clean_table_file.write(content)


def clean_all_html():
    file_list = glob.glob(os.path.join(os.getcwd(), '*.html'))
    print("Cleaning %s files" % len(file_list))
    for f in file_list:
        clean_table(f)


if __name__ == "__main__":
    clean_all_html()
