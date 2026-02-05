













!pip install nltk



"""
could you please help me build a python script for google colab which imports and reads a txt file. it should then go through each row in the txt file and for each row count the syllables in each sentence. create a dictionary with keys 1, 2, 3, 4... up to 40. Then for every key it should create a list of strings as the value. then the script should go through all all the rows in the txt file and count the syllables in the senctences on each row. then all the senctences should be inserted as separate strings into the lists in the dictionary. Each sentence should be put together with the key which has the key number corresponding to the number of syllables in the sentence. finally as the dictionary has been created, there should be another function which takes a string as an input. the function should count the number of syllables in the input sentence. then it should go to the dictionary and retrieve a random sentence from the list corresponding to the key that has the same number of syllables.
"""


import nltk
nltk.download('cmudict')


######

import random
from collections import defaultdict
import re

# Load the CMU Pronouncing Dictionary
import nltk.corpus
cmu_dict = nltk.corpus.cmudict.dict()

def count_syllables(word):
    """Estimate the number of syllables in a word using the CMU Pronouncing Dictionary."""
    word = word.lower()
    if word in cmu_dict:
        # Return the minimum syllable count from all pronunciations
        return min([len(list(y for y in x if y[-1].isdigit())) for x in cmu_dict[word]])
    else:
        # Fallback: count vowels as syllables for unknown words
        return len(re.findall(r'[aeiouy]+', word))

####


"""
# In Google Colab, upload your text file first:
from google.colab import files

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
"""
directory = "/content/drive/My Drive/text_processing/"
directory = 'text_processing/'
    #output_file = directory + '/output.txt'

#uploaded = directory + "diverse.txt" #files.upload()
filename = directory + "2023-01-27- slum_merge.txt"

# Assuming the filename is 'input.txt'
#filename = list(uploaded)#uploaded.keys())[0]

# Read the lines
with open(filename, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]



###

# Initialize the dictionary with keys from 1 to 40
syllable_dict = {i: [] for i in range(1, 41)}

# Process each line
for line in lines:
    # Split into sentences if needed; here, assuming each line is a sentence
    sentences = [line]
    for sentence in sentences:
        # Tokenize into words
        words = re.findall(r'\b\w+\b', sentence)
        # Count syllables per sentence
        total_syllables = sum(count_syllables(word) for word in words)
        # Assign to the dictionary if within range
        if 1 <= total_syllables <= 40:
            syllable_dict[total_syllables].append(sentence)

##############
#print(syllable_dict)




#########

import re

"""
the following function calculates the number of syllables in a sentence and returns another sentence from a given dictionary which has the same amount of syllables. please also add another "rhyme finder" function which, based on the input sentence and a list of sentences, finds a sentence which rhymes with the input. the user should be able to specify how many of the syllables at the end of the output sentence that should rhyme with the input
"""


# Basic syllable counter (heuristic)
def count_syllables(word):
    word = word.lower()
    # Remove non-alphabetic characters
    word = re.sub(r'[^a-z]', '', word)
    if len(word) == 0:
        return 0
    # Count vowel groups as syllables
    vowels = 'aeiou'
    count = 0
    prev_char_was_vowel = False
    for char in word:
        if char in vowels:
            if not prev_char_was_vowel:
                count += 1
                prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False
    # Adjust for silent 'e' at the end
    if word.endswith('e'):
        count = max(1, count - 1)
    return count


def count_syllables_in_sentence(sentence):
    words = sentence.split()
    total_syllables = sum(count_syllables(word) for word in words)
    return total_syllables


def find_sentence_with_same_syllables(target_sentence, dictionary):
    target_syllables = count_syllables_in_sentence(target_sentence)
    sentence_list = dictionary[target_syllables]
    #print(sentence_list)
    for sentence in sentence_list:
        if count_syllables_in_sentence(sentence) == target_syllables:
            return sentence, sentence_list
    return None  # No match found


def get_ending_syllables(sentence, n):
    """
    Extract the last n syllables from the sentence.
    """
    words = sentence.split()
    syllable_list = []

    # Build list of (word, syllable_count)
    for word in words:
        syllables = count_syllables(word)
        syllable_list.append((word, syllables))

    # Collect from the end until we reach n syllables
    ending_words = []
    total_syllables = 0
    for word, syl_count in reversed(syllable_list):
        ending_words.insert(0, word)
        total_syllables += syl_count
        if total_syllables >= n:
            break
    return ' '.join(ending_words)

"""
def rhyme_finder(input_sentence, sentence_list, rhyme_syllable_count=1):
    input_rhyme_part = get_ending_syllables(input_sentence, rhyme_syllable_count).lower()

    for sentence in sentence_list:
        candidate_rhyme_part = get_ending_syllables(sentence, rhyme_syllable_count).lower()
        if candidate_rhyme_part == input_rhyme_part:
            return sentence
    return None  # No rhyming sentence found
"""
"""
def rhyme_finder(input_sentence, sentence_list, rhyme_syllable_count=1):
    input_rhyme_part = get_ending_syllables(input_sentence, rhyme_syllable_count).lower()
    print("input_rhyme_part")
    print(input_rhyme_part)
    input_ending_word = get_last_word(input_sentence).lower()

    for sentence in sentence_list:
        candidate_rhyme_part = get_ending_syllables(sentence, rhyme_syllable_count).lower()
        candidate_ending_word = get_last_word(sentence).lower()
        print("candidate_rhyme_part")
        print(candidate_rhyme_part)

        if candidate_rhyme_part == input_rhyme_part and candidate_ending_word != input_ending_word:
            return sentence
    return None  # No suitable rhyming sentence found

# Helper function to get the last word of a sentence
def get_last_word(sentence):
    words = sentence.strip().split()
    return words[-1] if words else ''
"""
def rhyme_finder(input_sentence, sentence_list, rhyme_syllable_count=1):
    """
    Finds a sentence in sentence_list that rhymes with input_sentence based on the last rhyme_syllable_count syllables,
    but ensures the ending word is different from that of input_sentence.
    """
    input_rhyme_part = get_ending_syllables(input_sentence, rhyme_syllable_count).lower()
    input_ending_word = get_last_word(input_sentence).lower()

    for sentence in sentence_list:
        candidate_rhyme_part = get_ending_syllables(sentence, rhyme_syllable_count).lower()
        candidate_ending_word = get_last_word(sentence).lower()

        # Compare only the last three letters of the rhyme parts
        if (len(candidate_rhyme_part) >= 3 and len(input_rhyme_part) >= 3):
            rhyme_match = (candidate_rhyme_part[-3:] == input_rhyme_part[-3:])
        else:
            # If either rhyme part is shorter than 3 characters, compare the entire strings
            rhyme_match = (candidate_rhyme_part == input_rhyme_part)

        if rhyme_match and candidate_ending_word != input_ending_word:
            return sentence
    return None  # No suitable rhyming sentence found

#########
# Example usage:
if __name__ == "__main__":

    input_sentence = "hoppar ner på gatan"  # "The cat sat on the mat"

    # Find sentence with same syllable count
    matching_sentence, syllable_list = find_sentence_with_same_syllables(input_sentence, dictionary=syllable_dict)
    # print("Matching sentence by syllables:", matching_sentence)

    # Input sentence
    # test_sentence =

    print(f"Input sentence: {input_sentence}")
    print(f"Random sentence with same syllables: {matching_sentence}")

    # Find a rhyming sentence with last 2 syllables matching
    rhyme_sentence = rhyme_finder(input_sentence, syllable_list, rhyme_syllable_count=1)
    print("Rhyming sentence:", rhyme_sentence)




