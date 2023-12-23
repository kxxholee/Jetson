import os
import glob
import argparse
from collections import Counter

'''
    이 코드의 동기 : 데이터셋 안에 다른 클래스를 가진 데이터셋이 너무 많음
    이 코드의 역할 : 데이터셋 안의 전체 txt파일에 대해서, class를 읽고,
                  각 class의 데이터가 몇개씩 있는지를 반환
    
    -d, --directory : class가 있는 데이터셋 디렉토리 경로 (train, val)
'''

# 띄어쓰기로 구분된 class(숫자) 분리
def read_first_word(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        first_word = first_line.split()[0]
        return first_word

def get_yolo_classes_and_counts(directory):
    # 지정된 디렉토리에 있는 모든 텍스트 파일들에 대해서
    text_files = glob.glob(os.path.join(directory, '*.txt'))

    # dict에 저장
    class_counts = Counter()

    # 텍스트파일의 첫글자가 클래스임 : 첫글자들을 읽어서, 각 클래스의 개수 세기
    for file_path in text_files:
        try:
            class_name = read_first_word(file_path)
            class_counts[int(class_name)] += 1
        except (IndexError, ValueError):
            # 혹시모를 에러 처리
            continue

    # item에 따라서 클래스들 sort (item : 클래스 번호 [0, 1, 2] 중 하나)
    sorted_class_counts = sorted(class_counts.items())

    return sorted_class_counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Count number and list of classes from YOLO format labels.")
    parser.add_argument('-d', '--directory', type=str, help='Directory containing YOLO format TXT files')
    args = parser.parse_args()

    class_counts = get_yolo_classes_and_counts(args.directory)

    # 출력
    print(f" - Class Counts - ")
    for class_num, count in class_counts:
        print(f"Class {class_num}: {count}")

