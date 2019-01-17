import xml.etree.ElementTree as ET
import os


def select_files_in_folder(directory, ext):
    for file in os.listdir(directory):
        if file.endswith('.%s' % ext):
            yield os.path.join(directory, file)


def parse_xml_data():
    sentences = {}
    for file in select_files_in_folder('data', 'xml'):
        tree = ET.parse(file)
        root = tree.getroot()
        print('READING %s ...' % file)
        for article in root:
            for sentence in article:
                sentences[sentence.attrib['n']] = {'words': []}
                for tag in sentence:
                    datapoint = {'n': tag.attrib['n'], 'pos': tag.attrib['pos'],
                                 'verified': tag.attrib['verified'], 'word': tag.text}
                    sentences[sentence.attrib['n']]['words'].append(datapoint)
    return sentences


s = parse_xml_data()
print('Total number of sentences: %s' % len(s))
words = []
for sentence in s:
    for word in s[sentence]['words']:
        words.append(word)
print('Total number of words: %s' % len(words))
