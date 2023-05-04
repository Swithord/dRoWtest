import openai
from openai.embeddings_utils import get_embedding
import pandas as pd
import numpy as np

openai.api_key = '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'

# df = pd.read_json('Site Diary.json')['data'].to_frame()
# df = df.truncate(after=10, axis='rows')
# df['data'] = df['data'].astype(str)
# df['embedding'] = df['data'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
# df.to_csv('embeddings.csv')
df = pd.read_csv('embeddings.csv')
df['embedding'] = df['embedding'].apply(np.array)
df.to_csv('embeddings2.csv')
# with open('Site Diary.json') as f:
#     contracts = json.load(f)
#
# for contract in contracts:
#     embeddings.append([get_embedding(str(contract['data']), engine='text-embedding-ada-002')])
#
# df = pd.DataFrame(embeddings)
