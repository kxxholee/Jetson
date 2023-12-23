#!/bin/bash

# 이 파일은 필요한 데이터셋 filtering을 위해서 사용함. (노트북에서 실행)
# 텍스트 파일들을 모두 읽어서, 0, 1, 2만 남기는 것이 목표
# class가 0, 1, 2가 아니면 trash로 보냄 (rm은 위험해서 안씀)
# train과 val 디렉토리 모두에 대해서 실행함

SOURCE_DIR_A="/home/vnla/Workbench/Testing/Ultra/dataset/val"

for file in "$SOURCE_DIR_A"/*.txt; do
  class=$(head -n 1 "$file" | cut -d' ' -f1)
  if [[ ! ($class -eq 0 || $class -eq 1 || $class -eq 2) ]]; then
    base_name=$(basename "$file" .txt)
    gio trash "$SOURCE_DIR_A"/$base_name.*
  fi
done

SOURCE_DIR_B="/home/vnla/Workbench/Testing/Ultra/dataset/train"

for file in "$SOURCE_DIR_B"/*.txt; do
  class=$(head -n 1 "$file" | cut -d' ' -f1)
  if [[ ! ($class -eq 0 || $class -eq 1 || $class -eq 2) ]]; then
    base_name=$(basename "$file" .txt)
    gio trash "$SOURCE_DIR_B"/$base_name.*
  fi
done