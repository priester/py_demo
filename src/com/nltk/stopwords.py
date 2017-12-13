
import nltk

sw = set(nltk.corpus.stopwords.words('english'))
print('Stop words', list(sw)[:7])

# 取得gutenberg语料库中的部分文件
gb = nltk.corpus.gutenberg
print('Gutenberg files', gb.fileids()[-5:])

# 取milton-paradise.txt文件中的前两句,作为下面所用的过滤语句
text_sent = gb.sents("milton-paradise.txt")[:2]
print('Unfiltered', text_sent)

# 过滤停用字
for sent in text_sent:
    filtered = [w for w in sent if w.lower() not in sw]
    print('Filtered', filtered)
    # 取得文本内所含的标签
    tagged = nltk.pos_tag(filtered)
    print("Tagged", tagged)

    words = []
    for word in tagged:
        if word[1] != 'NNP' and word[1] != 'CD':
            words.append(word[0])
    print(words)