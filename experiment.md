### 26.09.23

## 실험 계획

**test model = gemma 2B**

1. base 모델에 데이터 넣고 SAE 분포 확인
2. base 모델에 학업 수준에 roPE 주기, 이후 SAE분포 확인
3. fine-tunning 한 SAE에 데이터 넣고 SAE 분포 확인
4. fine-tunning 한 SAE에 학업 수준에 roPE 주고 SAE 분포 확인

### 26.09.24

## 데이터셋 구성

GPQA(main) + MMLU 

고등학생 수준 = ['high_school_biology', 'high_school_chemistry', 'high_school_physics'] ㄷ MMLU['subject']<br/>
학부 수준 = ['colledge_biology', colledge_chmistry', 'colledge_physics'] ㄷ MMLU['subject']<br/>
대학원 수준( reviewed by Ph.D ) = GPQA['Question'] ㄷ GPQA_main.csv \# GPQA based on biology, chemistry and physics<br/>

Dataset for experiments will build following columns:<br/>
Question | education_level | subject

if MMLU<br/>
    Question = MMLU['question'] + MMLU['choices']<br/>
else<br/>
    Question = GPQA['Question']<br/>

if MMLU<br/>
    education_level = split(MMLU['subject'])[level_idx]<br/>
else<br/>
    education_level = Graduate<br/>

if MMLU<br/>
    subject = split(MMLU['subject'])[subject_idx]<br/>
else<br/>
    subject = unique(GPQA['High-level domain'])<br/>


### 26.09.24

## MMLU 객관식 문항에서 주관식 문항으로 변경

\#example<br/>
<img src = dataset/example_of_reconstructed_MMLU.png>

<br/>test : 1010, validation : 106, total : 1116 
<br/>(same number as the original set, which excluded dev set. test and validation are included)

$$
\text{subjective\_question} = \text{Qwen}_{\theta = \text{9B}}(\text{question}, \text{choices}, \text{answer})
$$

