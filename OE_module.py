import opennre
import pandas as pd
from itertools import combinations
import json
import torch
from opennre.encoder import BERTEntityEncoder
from opennre.model import SoftmaxNN

class my_module:
    def __init__(self):
        # Укажите путь к вашим файлам
        model_path = 'ckpt/nerel_rubert.pth.tar'
        rel2id_path = 'rel2id.json'  # Файл с соотношением ID и отношений
        pretrain_path = 'rubert-base-cased'  # Путь к предобученному rubert

        # Загрузите rel2id (если у вас есть)
        with open(rel2id_path) as f:
            rel2id = json.load(f)

        # Создайте энкодер и модель
        sentence_encoder = BERTEntityEncoder(
            max_length=128,  # Должно совпадать с тем, что использовалось при обучении
            pretrain_path=pretrain_path  # Путь к rubert
        )
        self.model = SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

        # Загрузите веса (учитывая устройство: CPU/GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()  # Переключите в режим предсказания


    def run_module(self, df: pd.DataFrame, chunk_df: pd.DataFrame):
        pairs = []
        for chunk_id, group in df.groupby('chunk_id'):
            entities = list(zip(
                group['entity_name'], 
                group['entity_type'], 
                group['start'], 
                group['end']
            ))
            # Генерируем все возможные уникальные пары
            for (ent1, type1, start1, end1), (ent2, type2, start2, end2) in combinations(entities, 2):
                pairs.append({
                    'entity_1': ent1,
                    'entity_type_1': type1,
                    'start_1': start1,
                    'end_1': end1,
                    'entity_2': ent2,
                    'entity_type_2': type2,
                    'start_2': start2,
                    'end_2': end2,
                    'chunk_id': chunk_id
                })
        pairs = pd.DataFrame(pairs)
        relations = []
        for _, row in pairs.iterrows():
            chunk_text = chunk_df.loc[row['chunk_id'] == chunk_df['chunk_id'], 'text'].values[0]
            json_obj = {
                'text': chunk_text,
                'h': {
                    'pos': (row['start_1'], row['end_1']),
                    #'type': row['entity_type_1'],
                    #'name': row['entity_1']
                },
                't': {
                    'pos': (row['start_2'], row['end_2']),
                    #'type': row['entity_type_2'],
                    #'name': row['entity_2']
                }
            }
           
            relations.append(self.model.infer(json_obj))
        pairs['relation'] = relations

        return pairs