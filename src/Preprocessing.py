import pandas as pd
import re
import emoji
from soynlp.normalizer import repeat_normalize
import os

def preprocess_dataframe(df):
    # 1) 한글, 영어, 숫자, 특수문자, 유니코드 이모지를 제외한 글자 제거
    
    # emoji.EMOJI_DATA.keys() : 모든 이모지 키 값
    # ''.join() : 하나의 문자열로 병합
    emojis = ''.join(emoji.EMOJI_DATA.keys())
    # 한글, 영어, 숫자, 특수문자, 유니코드 이모지를 제외
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()a-zA-Z0-9ㄱ-힣{emojis}]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    
    df['text'] = (
        df['text']
        # df.astype(str) : 문자열 강제 변환
        .astype(str)
        # re.sub() : 패턴과 일치하는 문자열 변환
        .apply(lambda x: pattern.sub(' ', x))
        .apply(lambda x: url_pattern.sub('', x))
        # strip() : 앞뒤 공백 제거
        .apply(lambda x: x.strip())
        
        # 2) 중복 문자열 축약

        # repeat_normalize() : 반복되는 문자열의 반복 횟수 설정
        .apply(lambda x: repeat_normalize(x, num_repeats=2))
    )

    # 3) 10글자 이하 제거

    # str.len() : 문자열의 길이
    df.drop(index = df[df['text'].str.len() <= 10].index, inplace=True)

    # 4) 중복 문장 제거

    # df.drop_duplicates() : 중복 행 제거
    df.drop_duplicates(ignore_index=True, inplace=True)

    # 5) 행 셔플

    # df.sample(frac=1) : 전체 데이터(100%) 셔플
    # reset_index(drop=True) : 기존 인덱스 무시, 새 인덱스 부여
    df = df.sample(frac=1).reset_index(drop=True)

    return df

def file_save(path):
    # filename_extension : file path "."을 기준으로 분리하고 배열의 마지막에 저장된 문자열을 통해 파일 확장자 확인
    filename_extension = path.split(".")[-1]
    # txt -> dataframe
    if filename_extension == "txt":
        df = pd.read_csv(path, sep="\t", header=None)
    # csv -> dataframe
    elif filename_extension == "csv":
        df = pd.read_csv(path, sep=",", header=None)

    # df.dropna() : 결측치 제거
    df.dropna(axis=0, inplace=True)
    
    if len(df.columns)==1:
        df.columns = ['text']
    
    elif len(df.columns)==2:
        df.columns = ['text', 'label']
    
    df = preprocess_dataframe(df)

    # 파일을 저장할 폴더 경로
    path_folder = "./data/processed/model"

    # 해당 폴더가 없으면
    if not os.path.exists(path_folder):
        # 폴더 생성
        os.makedirs(path_folder)

    # dataframe -> txt
    # 지정된 경로에 txt 파일로 변환한 파일 저장
    filename = os.path.splitext(os.path.basename(path))[0]
    save_path = f'{path_folder}/{filename}_preprocessed.txt'

    # index = False : 인덱스 포함 X
    df.to_csv(save_path, sep = '\t', index = False)
    print(f"✅ 파일 저장 위치 : {save_path}")

if __name__ == "__main__":
    # file_name : 전처리 수행할 파일명(ex. pretraining.txt, dapt.txt, finetuning.csv)
    file_name = "pretraining.txt"

    path = f"./data/raw/model/{file_name}"
    file_save(path)