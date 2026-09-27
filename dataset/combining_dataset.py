import pandas as pd 
from sklearn.model_selection import train_test_split  
from datasets import load_dataset

gpqa = pd.read_csv("./gpqa_main.csv")
gpqa['education_level'] = 'graduate'
print(gpqa['education_level'].value_counts())

train_set, valid_set = train_test_split(gpqa, test_size=0.1, random_state=42)

train_set = train_set[['Question', 'Correct Answer', 'High-level domain', 'education_level']]
valid_set = valid_set[['Question', 'Correct Answer', 'High-level domain', 'education_level']]



mmlu_train = pd.read_csv("mmlu_test.csv")
mmlu_valid = pd.read_csv("mmlu_validation.csv")

mmlu_subjects = [
	"high_school_biology",
	"high_school_chemistry",
	"high_school_physics",
	"college_biology",
	"college_chemistry",
	"college_physics",
]
mmlu_answers = {}
for subject in mmlu_subjects:
	for split in ["test", "validation"]:
		dataset = load_dataset("cais/mmlu", subject, split=split)
		mmlu_answers.update(dict(zip(dataset["question"], dataset["answer"])))

mmlu_train["answer"] = mmlu_train["original_question"].map(mmlu_answers)
mmlu_valid["answer"] = mmlu_valid["original_question"].map(mmlu_answers)

if mmlu_train["answer"].isna().any() or mmlu_valid["answer"].isna().any():
	raise ValueError("MMLU answer를 원본 질문과 매칭하지 못한 행이 있습니다.")

mmlu_train = mmlu_train.rename(columns={
	'converted_question': 'Question',
	'domain': 'High-level domain',
})[['Question', 'answer', 'High-level domain', 'education_level']]
mmlu_valid = mmlu_valid.rename(columns={
	'converted_question': 'Question',
	'domain': 'High-level domain',
})[['Question', 'answer', 'High-level domain', 'education_level']]

train_set = train_set.rename(columns={'Correct Answer': 'answer'})
valid_set = valid_set.rename(columns={'Correct Answer': 'answer'})

trainset = pd.concat([train_set, mmlu_train], ignore_index=True)
validset = pd.concat([valid_set, mmlu_valid], ignore_index=True)

trainset.to_csv("trainset.csv", index=False)
validset.to_csv("validset.csv", index=False)

print(f"trainset: {len(trainset)} rows")
print(f"validset: {len(validset)} rows")

