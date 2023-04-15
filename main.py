import pandas as pd
from transformers import AutoTokenizer, T5ForConditionalGeneration

model_name = "IlyaGusev/rut5_base_sum_gazeta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

df = pd.read_excel('./data.xlsx', index_col=0)

str_prod = input()


def update_input(input_val):
    list_prod = [x.strip(' ').lower() for x in input_val.split(',')]
    return list_prod


new_val = update_input(str_prod)


def to_dict_val(name_prod):
    new = {}
    for i in name_prod:
        for j in df[df['Применение'].str.lower().str.contains(i)]['Влияние на организм'].values:
            if j.strip(' ') not in new:
                new[j.strip(' ')] = [i]
            else:
                new[j.strip(' ')].append(i)
    return new


dict_val = to_dict_val(new_val)


def pred_res(name_prod, dict_val):
    result = []
    for i in name_prod:
        text_list = []
        for key, val in dict_val.items():
            if i in val:
                text_list.append(key)

        text_str = ','.join(text_list)
        input_ids = tokenizer(
            [text_str],
            max_length=600,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"]

        output_ids = model.generate(
            input_ids=input_ids,
            no_repeat_ngram_size=2
        )[0]

        summary = tokenizer.decode(output_ids, skip_special_tokens=True)
        result.append(f'{i.capitalize()} {summary.lower()}')

    return result

print(*pred_res(new_val, dict_val))
