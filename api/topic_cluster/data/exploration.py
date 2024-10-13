import re
import logging
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()

def remove_stop_lemmatize(text, stopwords):
    text = text.replace('-', ' ')
    text = re.sub(r'([^a-zA-Z ]+?)', '', text)
    if stopwords:
        text = text.split(' ')
        return ' '.join(lemmatizer .lemmatize(word.lower() for word in text if word.lower() not in stopwords))
    else:
        return text

def create_df_for_siamese(mapping_file, grievance_data, sheet_name='Sheet2'):
    try:
        mapping_df = pd.read_excel(mapping_file, sheet_name=sheet_name)
        grievance_data_df = pd.read_json(grievance_data)
    except Exception as e:
        logging.error(f'Error occured while loading data : {e}')

    grievance_data_df['CategoryV7'] = grievance_data_df['CategoryV7'].apply(
        lambda x: int(x['$numberLong']) if pd.notnull(x) and isinstance(x, dict) and '$numberLong' in x else int(x) if pd.notnull(x) else None
    )
    merged_df = pd.merge(
        grievance_data_df,
        mapping_df,
        left_on=['CategoryV7', 'org_code'],
        right_on=['Code', 'OrgCode'],
        how='inner',
    )
    merged_df['CategoryV7'] = merged_df['CategoryV7'].fillna(0).astype(int)
    merged_df['CategoryV7'] = merged_df['CategoryV7'].astype(int, errors='ignore')
    merged_df = merged_df[['CategoryV7', 'Description', 'subject_content_text']].copy()
    merged_df['subject_content'] = merged_df['subject_content'].apply(remove_stop_lemmatize, args=(stopwords, ))
    
    return merged_df

