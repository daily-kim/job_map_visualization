# job_map_visualization

## Requirements

- Download fasttext weight from
  https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz

- Save weight file to /data

- Install requirements.txt

## Data description

- job_data.csv
  - (커리어넷) 직업-직업설명
- job_major_subject.csv
  - (커리어넷) 직업-학과-과목 관계
- major_info.csv
  - (커리어넷)학과-학과설명
- subject_common_info.csv
  - (수기입력)1학년 공통과목-과목설명
  - http://ncic.re.kr/mobile.index2.do
- subject_info.csv
  - (크롤링)2,3학년 공통과목-과목설명 + 1학년 공통과목
  - subject_common_info 를 포함함

# How to Use

- jobname, grade, s2s_threshold, similarity_method 지정하여 생성

  - jobname : 직업명
  - grade : 학년(고등), 0 입력시 1,2,3 통합 진로맵 생성
  - s2s_threshold : subject_subject간 edge 생성 threshold
  - similariry_method : fasttext 만 구현됨

- example  
  `python main.py --jobname="통계학연구원" --grade=3 --s2s_threshold=0.9 --similarity_method="fasttext"  `

- 작업 완료 후 result
- pretrained model이기 때문에 weight를 불러오는 시간이 오래걸리게 됨, 추후 batch 단위로 작업할 수 있도록 수정예정
