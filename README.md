# AI 도우미 프로젝트

이미지 분석과 텍스트 생성을 결합한 AI 도우미 웹 애플리케이션입니다.

## 설치 방법

1. 필요한 패키지 설치: 

```bash
pip install torch torchvision transformers
```

2. 프로젝트 폴더로 이동:

```bash
cd new-year-project-one
```

## 실행 방법

1. 다음 명령어로 애플리케이션 실행:

```bash
python app.py
```

2. 웹 브라우저에서 표시되는 URL 접속

## 주요 기능

- 이미지 분석: 업로드된 이미지의 내용을 분석
- 텍스트 생성: 입력된 프롬프트를 기반으로 텍스트 생성
- 통합 분석: 이미지와 텍스트를 함께 분석

## 사용된 모델

- 이미지 분석: google/vit-base-patch16-224
- 텍스트 생성: skt/kogpt2-base-v2
