import os
import math
import re

Top_5000_Words = []

# 빈도수가 높은 상위 5000개의 단어 읽어오기 + 정렬
file = open('./Top5000Word.txt', 'r', encoding='utf8')
line = file.readlines()  # 해당 파일의 모든 라인을 리스트 형태로 리턴.
for i in range(5000):
    matches = line[i].split('\t')
    if "NNP" in matches[0] or "NNG" in matches[0]:
        Top_5000_Words.append(matches[0].replace("\n", ""))
file.close()
Top_5000_Words.sort()

Input_doc_num = 0   # 각 Category 의 Document 수
tf_D = {}  # (key:파일명, value:Top 5000 단어의 빈도수 리스트)
idf = {}  # (key:단어, value: 해당 단어가 총 몇개의 문서에 포함되는지 계산한 수)
tf_idf_D = {}  # tf(t,d) * idf(t,D)

for word in Top_5000_Words:
    idf[word] = 0

# 학습 데이터 만들기
for path, dirs, files in os.walk('./Corpus/Input_Data/'):
    for filename in files:
        ext = os.path.splitext(filename)[-1]    # 확장자명
        if ext == '.txt':
            file = open(path+'/'+filename, 'r', encoding='utf8')
            lines = file.readlines()  # 해당 파일의 모든 라인을 리스트 형태로 리턴.

            word_list = []  # 각 파일의 모든 NNG, NNP 를 저장하는 리스트

            # 각 파일에서 라인 별로 NNG, NNP 만 추출
            for line in lines:
                if line is not '\n' and line is not '':
                    matches = line.split('\t')
                    if len(matches) > 1:
                        if "NNP" in matches[1] or "NNG" in matches[1]:
                            if '+' in matches[1]:
                                matches2 = matches[1].split('+')
                                for temp in matches2:
                                    if "NNP" in temp or "NNG" in temp:
                                        word = re.sub('ⓒ', '', temp)  # (c)샘 제외시키기
                                        word_list.append(word.replace("\n", ""))
                            else:
                                word = re.sub('ⓒ', '', matches[1])  # (c)샘 제외시키기
                                word_list.append(word.replace("\n", ""))

            # 각 파일에서 Top 5000 Words 각 단어의 f(t,d) 구하기
            tf_d = {}
            for word in Top_5000_Words:
                tf_d[word] = 0
            for word in word_list:
                if word in Top_5000_Words:
                    tf_d[word] = tf_d[word] + 1
            tf_D[filename] = tf_d

            # Top 5000 Words 각 단어가 해당 문서에서 나타났으면 +1
            for word in Top_5000_Words:
                if tf_d[word] > 0:
                    idf[word] = idf[word] + 1

            # 해당 폴더의 문서의 개수
            Input_doc_num += 1

'''
print(tf_D['(POS)child_2.txt']['가능성/NNG'])
print(idf['가능성/NNG'])
print("--------")
'''

# f(t,d)로 tf(t,d) 구하기
for file_name in tf_D:
    for word in tf_D[file_name]:
        tf_D[file_name][word] = math.log10(tf_D[file_name][word] + 1)

# idf 구하기
for word in idf:
    idf[word] = math.log10(abs(Input_doc_num) / abs(1 + idf[word]))

# tf_idf 구하기 및 정규화
for file_name in tf_D:
    tf_idf_d = {}
    tf_idf_sum = 0
    for word in tf_D[file_name]:
        tf_idf_d[word] = tf_D[file_name][word] * idf[word]
        tf_idf_sum += math.pow(tf_idf_d[word], 2)   # 정규화. 모든 tf_idf 값을 제곱하여 더하기
    tf_idf_sum = math.sqrt(tf_idf_sum)  # 정규화. 문서의 모든 tf_idf 제곱 합의 루트 구하기
    for word in tf_idf_d:
        tf_idf_d[word] /= tf_idf_sum
    tf_idf_D[file_name] = tf_idf_d

'''
print(tf_D['(POS)child_2.txt']['가능성/NNG'])
print(idf['가능성/NNG'])
print(tf_idf_D['(POS)child_2.txt']['가능성/NNG'])
print(tf_D['(POS)child_2.txt'])
'''

# Input_Data 파일쓰기
for path, dirs, files in os.walk('./Corpus/Input_Data/'):
    for filename in files:
        temp_path = path.split('/')
        category = temp_path[3]

        ext = os.path.splitext(filename)[-1]    # 확장자명
        if ext == '.txt':
            temp_path = path.split('/')
            file = open('./201433753_김윤진/Input_Data/'+str(category)+'/'+filename, 'w', encoding='utf8')

            for word in Top_5000_Words:
                file.write(str(tf_idf_D[filename][word])+"\t")
        file.close()

#######################################################################
# Test_Feature_Data 만들기
#######################################################################

tf_D = {}
tf_idf_D = {}

for path, dirs, files in os.walk('./Corpus/Test_Data/'):
    for filename in files:
        ext = os.path.splitext(filename)[-1]    # 확장자명
        if ext == '.txt':
            file = open(path+'/'+filename, 'r', encoding='utf8')
            lines = file.readlines()  # 해당 파일의 모든 라인을 리스트 형태로 리턴.

            word_list = []  # 각 파일의 모든 NNG, NNP 를 저장하는 리스트

            # 각 파일에서 라인 별로 NNG, NNP 만 추출
            for line in lines:
                if line is not '\n' and line is not '':
                    matches = line.split('\t')
                    if len(matches) > 1:
                        if "NNP" in matches[1] or "NNG" in matches[1]:
                            if '+' in matches[1]:
                                matches2 = matches[1].split('+')
                                for temp in matches2:
                                    if "NNP" in temp or "NNG" in temp:
                                        word = re.sub('ⓒ', '', temp)  # (c)샘 제외시키기
                                        word_list.append(word.replace("\n", ""))
                            else:
                                word = re.sub('ⓒ', '', matches[1])  # (c)샘 제외시키기
                                word_list.append(word.replace("\n", ""))

            # 각 파일에서 Top 5000 Words 각 단어의 f(t,d) 구하기 및 tf(t,d) 구하기
            tf_d = {}
            for word in Top_5000_Words:
                tf_d[word] = 0
            for word in word_list:
                if word in Top_5000_Words:
                    tf_d[word] += 1
            for word in tf_d:
                tf_d[word] = math.log10(tf_d[word] + 1)
            tf_D[filename] = tf_d

'''
# f(t,d)로 tf(t,d) 구하기
for file_name in tf_D:
    for word in tf_D[file_name]:
        tf_D[file_name][word] = math.log10(tf_D[file_name][word] + 1)
'''
# tf_idf 구하기 및 정규화
for file_name in tf_D:
    tf_idf_d = {}
    tf_idf_sum = 0
    for word in tf_D[file_name]:
        tf_idf_d[word] = tf_D[file_name][word] * idf[word]
        tf_idf_sum += math.pow(tf_idf_d[word], 2)   # 정규화. 모든 tf_idf 값을 제곱하여 더하기
    tf_idf_sum = math.sqrt(tf_idf_sum)  # 정규화. 문서의 모든 tf_idf 제곱 합의 루트 구하기
    for word in tf_idf_d:
        tf_idf_d[word] /= tf_idf_sum
    tf_idf_D[file_name] = tf_idf_d

# Test_Feature_Data 만들기
for path, dirs, files in os.walk('./Corpus/Test_Data/'):
    for filename in files:
        temp_path = path.split('/')
        category = temp_path[3]
        ext = os.path.splitext(filename)[-1]    # 확장자명
        if ext == '.txt':
            temp_path = path.split('/')
            file = open('./201433753_김윤진/Test_Feature_Data/'+str(category)+'/'+filename, 'w', encoding='utf8')
            for word in Top_5000_Words:
                file.write(str(tf_idf_D[filename][word])+"\t")
        file.close()

######################################################################
# Val_Feature_Data 만들기
#######################################################################

tf_D = {}
tf_idf_D = {}

for path, dirs, files in os.walk('./Corpus/Val_Data/'):
    for filename in files:
        ext = os.path.splitext(filename)[-1]    # 확장자명
        if ext == '.txt':
            file = open(path+'/'+filename, 'r', encoding='utf8')
            lines = file.readlines()  # 해당 파일의 모든 라인을 리스트 형태로 리턴.

            word_list = []  # 각 파일의 모든 NNG, NNP 를 저장하는 리스트

            # 각 파일에서 라인 별로 NNG, NNP 만 추출
            for line in lines:
                if line is not '\n' and line is not '':
                    matches = line.split('\t')
                    if len(matches) > 1:
                        if "NNP" in matches[1] or "NNG" in matches[1]:
                            if '+' in matches[1]:
                                matches2 = matches[1].split('+')
                                for temp in matches2:
                                    if "NNP" in temp or "NNG" in temp:
                                        word = re.sub('ⓒ', '', temp)  # (c)샘 제외시키기
                                        word_list.append(word.replace("\n", ""))
                            else:
                                word = re.sub('ⓒ', '', matches[1])  # (c)샘 제외시키기
                                word_list.append(word.replace("\n", ""))

            # 각 파일에서 Top 5000 Words 각 단어의 f(t,d) 구하기
            tf_d = {}
            for word in Top_5000_Words:
                tf_d[word] = 0
            for word in word_list:
                if word in Top_5000_Words:
                    tf_d[word] = tf_d[word] + 1
            tf_D[filename] = tf_d

# f(t,d)로 tf(t,d) 구하기
for file_name in tf_D:
    for word in tf_D[file_name]:
        tf_D[file_name][word] = math.log10(tf_D[file_name][word] + 1)

# tf_idf 구하기 및 정규화
for file_name in tf_D:
    tf_idf_d = {}
    tf_idf_sum = 0
    for word in tf_D[file_name]:
        tf_idf_d[word] = tf_D[file_name][word] * idf[word]
        tf_idf_sum += math.pow(tf_idf_d[word], 2)   # 정규화. 모든 tf_idf 값을 제곱하여 더하기
    tf_idf_sum = math.sqrt(tf_idf_sum)  # 정규화. 문서의 모든 tf_idf 제곱 합의 루트 구하기
    for word in tf_idf_d:
        tf_idf_d[word] /= tf_idf_sum
    tf_idf_D[file_name] = tf_idf_d

# Val_Feature_Data 만들기
for path, dirs, files in os.walk('./Corpus/Val_Data/'):
    for filename in files:
        temp_path = path.split('/')
        category = temp_path[3]
        ext = os.path.splitext(filename)[-1]    # 확장자명
        if ext == '.txt':
            temp_path = path.split('/')
            file = open('./201433753_김윤진/Val_Feature_Data/'+str(category)+'/'+filename, 'w', encoding='utf8')
            for word in Top_5000_Words:
                file.write(str(tf_idf_D[filename][word])+"\t")
            file.close()
