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

- argument

  - jobname:(str, default: '통계학연구원') 직업명

  - grade:
    (int, default: 0) 진로맵 대상 학년, 0:전체, 1:1학년, 2:2학년, 3:3학년

  - similarity_method:
    (str, default: 'fasttext') 유사도 측정 방법을 지정합니다. 'fasttext' 또는 'sbert'

  - s2s_threshold:
    (float, default: 0.9) 과목간 간의 유사도 threshold
  - node_decay:
    (bool, default: True) 노드 감소 여부
  - node_decay_method:
    (str, default: 'num_total') 노드 감소 방법을 지정, ‘weight, 'num_total' 또는 'num_per_grade'
  - node_decay_weight:
    (float, default: 1.3) ‘weight 방식에 대한 노드 감소 기준
  - node_decay_num_total:
    (int, default: 15) 'num_total' 방법에 대한 노드 감소 기준
  - node_decay_num_per_grade:
    (int, default: 5) 'num_per_grade' 방법에 대한 노드 감소 기준
  - node_scale:
    (tuple, default: (0.5, 1)) 그래프 생성 시 노드 크기 범위
  - edge_scale:
    (tuple, default: (0.5, 1)) 그래프 생성 시 엣지 크기 범위
  - imgdir:
    (str, default: 'result/img/') 그래프 이미지를 저장할 경로
  - weightdir:
    (str, default: 'result/weight/') 가중치를 저장할 경로

- example  
  `python src/main.py --jobname="통계학연구원" --grade=3 --s2s_threshold=0.9 --similarity_method="fasttext"  `

- 작업 완료 후 result
- pretrained model이기 때문에 weight를 불러오는 시간이 오래걸리게 됨, 추후 batch 단위로 작업할 수 있도록 수정예정
