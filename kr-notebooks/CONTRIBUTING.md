# 한국어 번역 프로젝트에 참여하는 방법

"The Debugging Book" 번역 프로젝트는 일반적인 workflow를 따릅니다.
다음과 같은 계층 구조를 따라서 변경사항을 업데이트 합니다.

```
https://github.com/uds-se/debuggingbook (원본 저장소)
    |
    |--- https://github.com/darkrsw/debuggingbook (한국어 번역 프로젝트 저장소)
            |
            |--- https://github.com/.../debuggingbook
            |--- https://github.com/.../debuggingbook
            |--- https://github.com/.../debuggingbook
```

## 변경을 반영하는 방법

한두개의 문장 또는 간단한 코드 변경은 "darkrsw/debuggingbook" 저장소의 branch를 만들어서 pull request를 생성하길 추천합니다. 한 챕터 이상의 번역 작업은 fork를 해서 변경사항을 pull request를 생성하길 추천합니다.

단순 오탈자는 issue를 생성해서 보고해도 무방합니다.


## 문체
최근 대부분의 기술서적인 '~하다'체를 사용하고 있으므로, 이 번역 프로젝트도 같은 문체를 사용합니다. 어느정도 번역이 진행된 이후에는 문체의 번경이 매우 어려울것으로 예상됩니다. 혹시, 반드시 문체를 변경해야하는 중요한 사유가 있다면 빠른 feedback 부탁드립니다.

## 번역 workflow
각 챕터별로 다음 단계를 거쳐서 번역을 진행합니다.

1. 1차 번역
2. "darkrsw/debuggingbook"로 merge
3. 1차 검수
4. 2차 번역
5. "darkrsw/debuggingbook"로 merge
6. 2차 검수
7. "uds-se/debuggingbook"로 merge

### 1차 번역
1차 번역은 번역의 검수를 쉽게 하기 위해서 각 cell별로 한국어를 위에, 영문 원본을 아래에 두는 방식으로 번역합니다.

### 2차 번역
2차 번역부터는 한국어 부분만 남겨둡니다.


### 검수
한 챕터의 단계별 번역이 완료되면, 반드시 번역자와 다른 사람이 검수를 하도록 합니다.


## 업데이트
"uds-se/debuggingbook"의 원본 프로젝트가 주기적으로 업데이트 되므로 번역 프로젝트도 주기적으로 동기화되어야 합니다. 각 챕터의 번역 책임자와 검수자는 주기적으로 각 챕터별 변경사항을 파악하여 업데이트 합니다.

주요 변경사항은 "notebooks" 디렉토리의 *.ipynb파일들에서 발생합니다. 주기적으로 "kr-notebooks" 디렉토리의 대응되는 *.ipynb파일들과 내용이 같은지 확인합니다.

## 역할
* 챕터별 번역 책임자: 챕터별로 전체적인 번역 상황을 관리합니다.
* 번역 참여자: 챕터 전체 또는 일부를 번역 합니다.
* 검수자: 챕터 전체 또는 일부를 검수합니다.
