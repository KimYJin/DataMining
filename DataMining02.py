import os
import math

Top_5000_Words = []

# 빈도수가 높은 상위 5000개의 단어 읽어오기
file = open('./Output.txt', 'r', encoding='utf8')
line = file.readlines()  # 해당 파일의 모든 라인을 리스트 형태로 리턴.
for i in range(5000):
    matches = line[i].split('\t')
    if "NNP" in matches[0] or "NNG" in matches[0]:
        Top_5000_Words.append(matches[0].replace("\n", ""))
file.close()

# 각 폴더의 문서 수
Input_doc_num = 0

tf_D = {}  # 각 파일 마다 Top 5000 Words 각 단어의 빈도수 측정
idf = {}  # Top 5000 단어가 각각 몇개의 문서에 나타났는지 측정
tf_idf_D = {}  # tf(t,d) * idf(t,D)

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
                                        word_list.append(temp.replace("\n", ""))
                            else:
                                word_list.append(matches[1].replace("\n", ""))

            # 각 파일에서 Top 5000 Words 각 단어의 f(t,d) 구하기
            tf_d = {}
            for word in word_list:
                if word in Top_5000_Words:
                    count = tf_d.get(word, 0)  # key(word)의 value(count)반환. 정의되지 않은 키는 0을 default 로 반환
                    tf_d[word] = count + 1
            tf_D[filename] = tf_d

            # Top 5000 Words 각 단어가 해당 문서에서 나타났으면 +1
            for word in Top_5000_Words:
                if word in tf_d.keys():
                    count = idf.get(word, 0)
                    idf[word] = count + 1

            # 해당 폴더의 문서의 개수
            Input_doc_num += 1

# f(t,d)로 tf(t,d) 구하기
for file_name in tf_D:
    for word in tf_D[file_name]:
        tf_D[file_name][word] = math.log10(tf_D[file_name][word] + 1)

# idf 구하기 - Top 5000 Words 의 각 단어가 전체 문서 집합 D 중에서 몇개의 문서에 나타났는지
for word in Top_5000_Words:
    if word not in idf.keys():
        idf[word] = 0
for word in Top_5000_Words:
    idf[word] = math.log10(abs(Input_doc_num) / abs(1 + idf[word]))

# tf_idf 구하기
for file_name in tf_D:
    tf_idf_d = {}
    for word in tf_D[file_name]:
        tf_idf_d[word] = tf_D[file_name][word] * idf[word]
    tf_idf_D[file_name] = tf_idf_d

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
                if word in tf_idf_D[filename].keys():
                    file.write(str(tf_idf_D[filename][word]))
                else:
                    file.write('0')  # tf(t,d) 0 이면 무조건 0
                file.write("\t")
        file.close()

######################################################################
# 평가 데이터 만들기

tf_D = {}  # 각 파일 마다 Top 5000 Words 각 단어의 빈도수 측정
tf_idf_D = {}  # tf(t,d) * idf(t,D)

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
                                        word_list.append(temp.replace("\n", ""))
                            else:
                                word_list.append(matches[1].replace("\n", ""))

            # 각 파일에서 Top 5000 Words 각 단어의 f(t,d) 구하기
            tf_d = {}
            for word in word_list:
                if word in Top_5000_Words:
                    count = tf_d.get(word, 0)  # key(word)의 value(count)반환. 정의되지 않은 키는 0을 default 로 반환
                    tf_d[word] = count + 1
            tf_D[filename] = tf_d

# f(t,d)로 tf(t,d) 구하기
for file_name in tf_D:
    for word in tf_D[file_name]:
        tf_D[file_name][word] = math.log10(tf_D[file_name][word] + 1)

# tf_idf 구하기
for file_name in tf_D:
    tf_idf_d = {}
    for word in tf_D[file_name]:
        tf_idf_d[word] = tf_D[file_name][word] * idf[word]
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
                if word in tf_idf_D[filename].keys():
                    file.write(str(tf_idf_D[filename][word]))
                else:
                    file.write('0')  # tf(t,d) 0 이면 무조건 0
                file.write("\t")
        file.close()

######################################################################

tf_D = {}  # 각 파일 마다 Top 5000 Words 각 단어의 빈도수 측정
tf_idf_D = {}  # tf(t,d) * idf(t,D)

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
                                        word_list.append(temp.replace("\n", ""))
                            else:
                                word_list.append(matches[1].replace("\n", ""))

            # 각 파일에서 Top 5000 Words 각 단어의 f(t,d) 구하기
            tf_d = {}
            for word in word_list:
                if word in Top_5000_Words:
                    count = tf_d.get(word, 0)  # key(word)의 value(count)반환. 정의되지 않은 키는 0을 default 로 반환
                    tf_d[word] = count + 1
            tf_D[filename] = tf_d

# f(t,d)로 tf(t,d) 구하기
for file_name in tf_D:
    for word in tf_D[file_name]:
        tf_D[file_name][word] = math.log10(tf_D[file_name][word] + 1)

# tf_idf 구하기
for file_name in tf_D:
    tf_idf_d = {}
    for word in tf_D[file_name]:
        tf_idf_d[word] = tf_D[file_name][word] * idf[word]
    tf_idf_D[file_name] = tf_idf_d

# Test_Feature_Data 만들기
for path, dirs, files in os.walk('./Corpus/Val_Data/'):
    for filename in files:
        temp_path = path.split('/')
        category = temp_path[3]

        ext = os.path.splitext(filename)[-1]    # 확장자명
        if ext == '.txt':
            temp_path = path.split('/')
            file = open('./201433753_김윤진/Val_Feature_Data/'+str(category)+'/'+filename, 'w', encoding='utf8')

            for word in Top_5000_Words:
                if word in tf_idf_D[filename].keys():
                    file.write(str(tf_idf_D[filename][word]))
                else:
                    file.write('0')  # tf(t,d) 0 이면 무조건 0
                file.write("\t")
        file.close()