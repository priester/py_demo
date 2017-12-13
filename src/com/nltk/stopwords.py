
import nltk

sw = set(nltk.corpus.stopwords.words('english'))
print('Stop words', list(sw)[:7])

# ȡ��gutenberg���Ͽ��еĲ����ļ�
gb = nltk.corpus.gutenberg
print('Gutenberg files', gb.fileids()[-5:])

# ȡmilton-paradise.txt�ļ��е�ǰ����,��Ϊ�������õĹ������
text_sent = gb.sents("milton-paradise.txt")[:2]
print('Unfiltered', text_sent)

# ����ͣ����
for sent in text_sent:
    filtered = [w for w in sent if w.lower() not in sw]
    print('Filtered', filtered)
    # ȡ���ı��������ı�ǩ
    tagged = nltk.pos_tag(filtered)
    print("Tagged", tagged)

    words = []
    for word in tagged:
        if word[1] != 'NNP' and word[1] != 'CD':
            words.append(word[0])
    print(words)