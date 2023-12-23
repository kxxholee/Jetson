import json
import os
import glob
import argparse

'''
    이 코드의 동기 : AIhub에서 가져온 데이터셋은, 동명의 json파일로 이미지 설명이 되어 있음
                  YOLO모델 학습/검증에 알맞은 .txt파일로 변환함
    이 코드의 역할 : 데이터셋 안의 전체 json파일에 대해서, class와 bbox를 읽고,
                  class와 좌표에 따라 알맞은 txt파일을 생성함
    
    -d, --directory : json파일이 있는 데이터셋 디렉토리 경로 (train, val)
'''

# json파일을 yolo형식에 맞는 텍스트 파일로 변환
def convert_json_to_yolo(json_file):
    # json파일 open
    with open(json_file, 'r') as file:
        data = json.load(file)

    for image in data['images']:
        # 현재 json파일이 가리키는 img는 images내부의 file_name에 들어가 있음
        img_width = image['width']
        img_height = image['height']
        base_name = os.path.splitext(image['file_name'])[0]
        # 새로 만들 텍스트파일의 경로 생성
        txt_file_path = os.path.join(os.path.dirname(json_file), f"{base_name}.txt")

        yolo_format = []
        # box정보는 annotation의 bbox에 좌표가 들어 있음
        for annotation in data['annotations']:
            if annotation['bbox']:  # bbox가 존재하면 (None인 경우가 있음)
                x_min, y_min = annotation['bbox'][0]
                x_max, y_max = annotation['bbox'][1]
                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                category_id = annotation['category_id'] - 1  # 클래스 인덱스 조정
                yolo_format.append(f"{category_id} {x_center} {y_center} {width} {height}")

        # YOLO 형식으로 저장
        with open(txt_file_path, 'w') as file:
            file.write('\n'.join(yolo_format))

# 경로의 모든 json에 대해서 실행
def convert_all_json(directory):
    for json_file in glob.glob(os.path.join(directory, '*.json')):
        convert_json_to_yolo(json_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert JSON files to YOLO format labels.')
    parser.add_argument('-d', '--directory', type=str, help='Directory containing JSON files')
    
    args = parser.parse_args()
    convert_all_json(args.directory)

