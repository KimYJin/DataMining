def column_value(matrix, i):
    return [row[i] for row in matrix]


# 정답
answer = open("answer.txt", "r", encoding="utf-8")

# 예측
output = open("prediction.txt", "r", encoding="utf-8")
output_line = output.read().split()     # 리스트. 예측한 것을 공백 기준으로 분류하여 리스트 생성
matrix = [[0] * 9 for i in range(9)]
count = 0

for start in range(0, len(output_line), 9):     # output_line 리스트의 아이템을 9개씩 묶기 위한 start 변수
    temp = output_line[start:start + 9]     # 각 문서의 9개의 카테고리 별 예측 값을 리스트로 저장
    max_index = temp.index(max(temp))       # 각 문서의 9개의 카테고리 별 예측 값 중 최대값의 index 구하기
    answer_index = int(answer.readline())   # answer 파일에서 정답 구하기
    # print(max_index, answer_index)
    if max_index == answer_index:
        count += 1
    matrix[max_index][answer_index] += 1
'''
for i in range(0, 9):
    print(matrix[i])
'''
# precision 값과 recall 값 리스트로 추가
precision_list = []
recall_list = []
for index in range(0, 9):
    # print(matrix)
    precisions = matrix[index][0:9]
    recalls = column_value(matrix, index)
    success = matrix[index][index]

    if sum(precisions) == 0:
        precision_list.insert(index, 0)
    else:
        precision_list.insert(index, success / sum(precisions))
    if sum(recalls) == 0:
        recall_list.insert(index, 0)
    else:
        recall_list.insert(index, success / sum(recalls))
#print(precision_list)
#print(recall_list)
print(">> Performance ")
#print(sum(precision_list) / 9)
#print(sum(recall_list) / 9)
f1_macro_score = (2 * sum(precision_list) / 9 * sum(recall_list) / 9) / (sum(precision_list) / 9 + sum(recall_list) / 9)
print(" - Macro_F1 : ", round(f1_macro_score, 4))
print("")

# TP, FP, FN 이용해서 F1 Score 내기
TP = 0
FP = 0
FN = 0
for index in range(0, 9):
    TP += matrix[index][index]
    FP += sum(matrix[index][0:9])
    FN += sum(column_value(matrix, index))
FP = FP - TP
FN = FN - TP
'''
print(TP)
print(FP)
print(FN)
'''
precision_score = TP / (TP + FP)
recall_score = TP / (TP + FN)
f1_micro_score = (2 * precision_score * recall_score) / (precision_score + recall_score)

print(" - Total Precision : ", round(precision_score, 4))
print(" - Total Recall : ", round(recall_score, 4))
print(" - Micro_F1 : ",  round(f1_micro_score, 4))

answer.close()
output.close()
