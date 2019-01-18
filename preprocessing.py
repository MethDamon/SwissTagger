import xml.etree.ElementTree as ET
import os


def select_files_in_folder(directory, ext):
    for file in os.listdir(directory):
        if file.endswith('.%s' % ext):
            yield os.path.join(directory, file)


def parse_xml_data():
    articles = []
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
            articles.append(sentences)
    return articles
