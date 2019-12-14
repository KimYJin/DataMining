import os
import re

word_list = []  # 모든 NNG, NNP 를 추출해서 저장할 리스트
folder = './Corpus/Input_Data/'
for path, dirs, files in os.walk(folder):
    for filename in files:
        ext = os.path.splitext(filename)[-1]    # 확장자명
        if ext == '.txt':
            file = open(path+'/'+filename, 'r', encoding='utf8')
            lines = file.readlines()  # readlines() : 해당 파일의 모든 라인을 리스트 형태로 리턴.

            # 해당 파일에서 라인 별로 NNG, NNP 만 추출
            for line in lines:
                if line is not '\n' and line is not '':
                    matches = line.split('\t')
                    if len(matches) > 1:
                        if "NNP" in matches[1] or "NNG" in matches[1]:
                            if '+' in matches[1]:
                                matches2 = matches[1].split('+')
                                for temp in matches2:
                                    if "NNP" in temp or "NNG" in temp:
                                        word = re.sub('ⓒ', '', temp)    # (c)샘 제외시키기
                                        word_list.append(word.replace("\n", ""))
                            else:
                                word = re.sub('ⓒ', '', matches[1])  # (c)샘 제외시키기
                                word_list.append(word.replace("\n", ""))

# 각 단어의 카운트를 세고, 딕셔너리로 만든다. (key: 단어, value: count)
word_frequency = {}
for word in word_list:
    count = word_frequency.get(word, 0)  # key(word)의 value(count)반환. 정의되지 않은 키는 0을 default 로 반환
    word_frequency[word] = count+1

# value(빈도수)를 기준으로 내림차순 정렬을 하고, 빈도수가 같을 경우 key(단어)의 사전 순으로 오름차순 정렬한다.
sorted_word_frequency = sorted(word_frequency.items(), key=lambda x: (-x[1], x[0]))

# Output.txt 파일을 생성하여, 결과 write.
file = open('./Top5000Word.txt', 'w', encoding='utf8')
for i in range(5034):
    file.write(sorted_word_frequency[i][0] + '\t' + str(sorted_word_frequency[i][1]) + '\n')
file.close()

