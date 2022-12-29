from spellchecker import SpellChecker as spell, SpellChecker


#Get all the text from a list that contains nested lists
def get_text(lst):
    text = []
    for item in lst:
        if isinstance(item, list):
            text.extend(get_text(item))
        else:
            text.append(item)
    return list(set(text)) #Remove duplicate word

def trainTitleTextFile(arr):
    with open('../Resources/title.txt', 'w') as f:
        # Iterate through the array and write each element to the file
        for item in arr:
            f.write(str(item) + '\n')

def trainDescriptionTextFile(arr):
    with open('../Resources/description.txt', 'w') as f:
        # Iterate through the array and write each element to the file
        for item in arr:
            f.write(str(item) + '\n')

def title_auto_correct(query):
    spell = SpellChecker()
    spell.word_frequency.load_text_file('../Resources/title.txt')
    correctedquery = spell.correction(query)
    return correctedquery

def description_auto_correct(query):
    spell = SpellChecker()
    spell.word_frequency.load_text_file('../Resources/description.txt')
    correctedquery = spell.correction(query)
    return correctedquery