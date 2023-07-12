import os
import pickle
from email.parser import Parser
import encodings
import re
from textblob import TextBlob
import csv
from encoder import Model
from pycorenlp import StanfordCoreNLP

def read_file():
    rootdir = "/home/nasim/Desktop/introductionDM/proposal/enron_mail/maildir/"
    emailbody_list = []
    ff = open("email_body.txt", "w+")
    for directory, subdirectory, filenames in os.walk(rootdir):
        for filename in filenames:
            f = open(os.path.join(directory, filename), "r", encoding='ISO-8859-1')
            data = f.read()
            email_parse = Parser().parsestr(data)

            if email_parse.get_payload():
                email_body = email_parse.get_payload()
                email_body = email_body.replace("\n", " ")
                email_body = email_body.replace("\t", " ")
                email_body = re.sub(r'([\w\.-]+)@([\w\.-]+)', r'--eMAIL--', email_body)
                # email_body = re.sub('\w[0-9]\w', '--nUMBER--', email_body)
                email_body = re.sub(r'http\S+', '--uRL--', email_body)

            email_cntnt = email_parse['Message-ID'] + " " + email_body
            emailbody_list.append(email_cntnt)
            ff.write(email_cntnt + "\n")

    with open('emailbody_list.pickle', 'wb') as c:
        pickle.dump(emailbody_list, c)
    ff.close()


def annonymize_body():
    f1 = open("email_body.ann", "r")
    f2 = open("email_body2.txt", "r")
    fw = open("email_body_new", "w+")
    body_ann = f1.readline()
    email_body = f2.read(512)
    i = 512
    last_indx = 0
    while body_ann:
        body_ann = body_ann.split()
        if body_ann[1] == "PER":
            while int(body_ann[3]) >= int(i):
                email_body = email_body + f2.read(512)
                i += 512

            fw.write(email_body[:int(body_ann[2]) - int(last_indx)] + "__PERSON__ ")
            email_body = email_body[int(body_ann[3]) - int(last_indx):]
            last_indx = body_ann[3]

        body_ann = f1.readline()

    email_body = email_body + f2.read(512)
    if email_body:
        fw.write(email_body)
        while email_body:
            email_body = f2.read(512)
            fw.write(email_body)

    fw.close()


def analyze():
    fr = open("email_body_small", "r")
    body_ann = fr.readline()
    fw = open("test.csv", 'w')
    writer = csv.writer(fw, delimiter='\t')
    i = 0
    model = Model()
    result = StanfordCoreNLP('http://localhost:9000')
    while body_ann:
        j = 0
        i = i + 1
        sentences = body_ann.split(".")
        for sntc in sentences:
            j = j + 1
            testimonial = TextBlob(sntc)  # TEXTBLOB Sentiment Analyze

            res = result.annotate(sntc, properties={'annotators': 'sentiment, ner, pos', 'outputFormat': 'json',
                                                    'timeout': 1000, })

            text = [sntc] #Neuron Sentiment
            xx = model.transform(text)
            sentiment_unit = xx[:, 2388]

            id = str(i) + "." + str(j)

            writer.writerow([id, testimonial.subjectivity, testimonial.polarity, res, sentiment_unit])
        body_ann = fr.readline()


# read_file()
# annonymize_body()
analyze()
