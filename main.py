from transformers import AutoTokenizer, T5ForConditionalGeneration
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def update_input(input_val):
    list_prod = [x.strip(' ').lower() for x in input_val.split(',')]
    return list_prod

def to_dict_val(name_prod):
    new = {}
    for i in name_prod:
        for j in df[df['Применение'].str.lower().str.contains(i)]['Влияние на организм'].values:
            if j.strip(' ') not in new:
                new[j.strip(' ')] = [i]
            else:
                new[j.strip(' ')].append(i)
    return new

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
        result.append(f'{i.capitalize()}: {summary.lower()}')

    return result

@st.cache_resource
def load_model():
    model_name = "IlyaGusev/rut5_base_sum_gazeta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


st.title('Вредность еды')

tokenizer, model = load_model()

df = pd.read_excel('./data.xlsx', index_col=0)

str_prod = st.text_input('Введите еду')

result = st.button('Рассчитать')
if result:
    for i in pred_res(update_input(str_prod), to_dict_val(update_input(str_prod))):
        st.write(i)

    dict_color = {'Очень высокая': 'maroon',
                  'Высокая': 'red',
                  'Средняя': 'orange',
                  'Низкая': 'limegreen',
                  'Очень низкая': 'lime',
                  'Безопасен': 'lawngreen'}

    for i in update_input(str_prod):
        labels = []
        colors = []

        for j in df[df['Применение'].str.lower().str.contains(i)][['Опасность']].value_counts().to_frame().reset_index()['Опасность']:
            labels.append(j)
            colors.append(dict_color[j])

        fig1, ax1 = plt.subplots(figsize = (6, 10))
        ax1.pie(df[df['Применение'].str.lower().str.contains(i)]['Опасность'].to_frame().value_counts(), labels = labels, colors=colors, autopct='%1.2f%%', textprops={'fontsize': 14})
        plt.title(label=f'Степень вреда химических добавок: {i}', fontdict={"fontsize": 16}, pad=20)
        st.pyplot(fig1)
