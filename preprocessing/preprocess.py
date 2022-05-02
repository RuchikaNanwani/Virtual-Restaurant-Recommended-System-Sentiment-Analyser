# Importing the packages for text preprocessing

import re
import sys
from utils import write_status
from nltk.stem.porter import PorterStemmer


def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(review):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    review = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', review)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    review = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', review)
    # Love -- <3, :*
    review = re.sub(r'(<3|:\*)', ' EMO_POS ', review)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    review = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', review)
    # Sad -- :-(, : (, :(, ):, )-:
    review = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', review)
    # Cry -- :,(, :'(, :"(
    review = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', review)
    return review


def preprocess_review(review):
    processed_review = []
    # Convert to lower case
    review = review.lower()
    # Replaces URLs with the word URL
    review = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', review)
    # Replace @handle with the word USER_MENTION
    review = re.sub(r'@[\S]+', 'USER_MENTION', review)
    # Replaces #hashtag with hashtag
    review = re.sub(r'#(\S+)', r' \1 ', review)
    # Remove RT (rereview)
    review = re.sub(r'\brt\b', '', review)
    # Replace 2+ dots with space
    review = re.sub(r'\.{2,}', ' ', review)
    # Strip space, " and ' from review
    review = review.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    review = handle_emojis(review)
    # Replace multiple spaces with a single space
    review = re.sub(r'\s+', ' ', review)
    words = review.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            if use_stemmer:
                word = str(porter_stemmer.stem(word))
            processed_review.append(word)

    return ' '.join(processed_review)


# method to handle the user provided csv, if the csv file is not as per the test file format

def preprocess_csv(csv_file_name, processed_file_name, test_file=False):
    save_to_file = open(processed_file_name, 'w')

    with open(csv_file_name, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            review_id = line[:line.find(',')]
            if not test_file:
                line = line[1 + line.find(','):]
                positive = int(line[:line.find(',')])
            line = line[1 + line.find(','):]
            review = line
            processed_review = preprocess_review(review)
            if not test_file:
                save_to_file.write('%s,%d,%s\n' %
                                   (review_id, positive, processed_review))
            else:
                save_to_file.write('%s,%s\n' %
                                   (review_id, processed_review))
            write_status(i + 1, total)
    save_to_file.close()
    print('\nSaved processed reviews to: %s' % processed_file_name)
    return processed_file_name

# creating function main and calling the rest of the functions

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python preprocess.py <raw-CSV>')
        exit()
    use_stemmer = False
    csv_file_name = sys.argv[1]
    processed_file_name = sys.argv[1][:-4] + '-processed.csv'
    if use_stemmer:
        porter_stemmer = PorterStemmer()
        processed_file_name = sys.argv[1][:-4] + '-processed-stemmed.csv'
    preprocess_csv(csv_file_name, processed_file_name, test_file=False)