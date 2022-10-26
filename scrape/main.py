import random
from spacy.lang.en import English
from conference import *

nlp = English()
nlp.add_pipe("sentencizer")

tmp = Conference.ACL(2022).retrieve("2022-ACL-LONG")

lst = tmp.papers['2022-ACL-LONG']
random.shuffle(lst)
for i in range(10):
    paper = lst[i]
    paper.download_pdf()
    doc = nlp(paper.content)

    lst2 = []
    for st in doc.sents:
        lst2.append(st.text)
    with open(f"./{paper.title.replace(' ',  '-')}.txt", "w") as f:
        for line in lst2:
            f.write(line + "\n")


