### 26.09.23

## 실험 계획

**test model = gemma 2B**

1. base 모델에 데이터 넣고 SAE 분포 확인
2. base 모델에 학업 수준에 roPE 주기, 이후 SAE분포 확인
3. fine-tunning 한 SAE에 데이터 넣고 SAE 분포 확인
4. fine-tunning 한 SAE에 학업 수준에 roPE 주고 SAE 분포 확인

### 26.09.24

## 데이터셋 구성

GPQA + MMLU 

고등학생 수준 = ['high_school_biology', 'high_school_chemistry', 'high_school_physics'] ㄷ MMLU['subject']<br/>
학부 수준 = ['colledge_biology', colledge_chmistry', 'colledge_physics'] ㄷ MMLU['subject']<br/>
대학원 수준( reviewed by Ph.D ) = GPQA['Question'] ㄷ GPQA_main.csv \# GPQA based on biology, chemistry and physics<br/>

Dataset for experiments will build following columns:<br/>
Prompt | education level | subject

if MMLU<br/>
    Prompt = MMLU['question'] + MMLU['choices']<br/>
else<br/>
    Prompt = GPQA['Question']<br/>

if MMLU<br/>
    education level = split(MMLU['subject'])[level_idx]<br/>
else<br/>
    education level = Graduate<br/>

if MMLU<br/>
    subject = split(MMLU['subject'])[subject_idx]<br/>
else<br/>
    subject = unique(GPQA['High-level domain'])<br/>




