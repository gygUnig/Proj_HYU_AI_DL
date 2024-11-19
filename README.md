# Detect AI-Generated Korean Text via KoBERT by Building Custom Dataset

### 한양대학교 2024년 2학기 AI+X:딥러닝 과목 프로젝트   
기계공학부 2018013309 김승희   
기계공학부 2018014002 유용준


## 1. Introduction
최근 인공지능의 가짜 뉴스 문제가 커지고 있다.

해외의 경우 Detect AI Generated Text와 관련된 기업이나 대회가 열리는 등 중요성이 부각되고 있으나, 한국어에 대해서는 기업도 없고 데이터셋도 없으며 연구도 진행이 덜 되었다.

그래서 우리는 AI Generated "Korean" Text에 대한 커스텀 데이터셋을 직접 구축하고 이를 판별하는 모델을 만들겠다.

## 2. Related Work
- 관련 기업 1: https://www.scribbr.com/ai-detector/
- 관련 기업 2: https://quillbot.com/ai-content-detector
- 관련 기업 3 ~: detect ai generated text 등으로 검색하면 더 많이 있음  
=> 해외 기업의 경우 다국어 지원을 하지만 한국어 전용은 없다.

- 관련 Competition: https://www.kaggle.com/competitions/llm-detect-ai-generated-text  
=> 해당 캐글 대회 역시 영어이기 때문에 한국어에 대해서 적용할 수 없다는 문제점이 존재한다.

## 3. Dataset Construction
- 한국어 AI Generated Text 분류 데이터셋이 존재하지 않기 때문에 직접 구축할 예정
- 데이터셋 이름: Ko-Detect-LLM-Text (임시)
- 방법: 한국어로 이뤄진 Source 자연어 데이터셋을 구하고, 해당 데이터를 GPT-4o-mini 등의 LLM으로 재구성해서 AI Generated Text / Human Written Text 이진 분류 데이터셋을 구축한다.

## 4. Methods
학습 방법 - 구축한 이진 분류 데이터셋으로 파인튜닝
모델 선정 - KoBERT

## 5. Experiments Analysis
실험 결과 분석 내용 작성

## 6. Conclusion
결론 내용 작성
