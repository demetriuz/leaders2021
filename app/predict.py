import pandas as pd
import numpy as np
import settings
import typing


class Predictor:
    MAX_LFM_RECOMMENDATIONS = 3
    MAX_ALGORITHMIC_RECOMMENDATIONS = 2
    MAX_RECOMMENDATIONS = MAX_LFM_RECOMMENDATIONS + MAX_ALGORITHMIC_RECOMMENDATIONS

    def __init__(self):
        books_features = ['recId', 'title', 'collapse_id', 'year_value', 'author_fullName', 'author_id',
                          'rubric_id', 'ageRestriction_id', 'ageRestriction', 'outputCount']
        books_df = pd.read_pickle(f'{settings.PREPARED_DATA_PATH}/books_full_df.pickle')[books_features]
        self.books_df = books_df
        self.books_df_collapsed = books_df.drop_duplicates('collapse_id')

        interaction_df_lt16 = pd.read_pickle(f'{settings.PREPARED_DATA_PATH}/interaction_df_lt16.pickle')
        interaction_df_gte16 = pd.read_pickle(f'{settings.PREPARED_DATA_PATH}/interaction_df_gte16.pickle')
        interaction_df = pd.read_pickle(f'{settings.PREPARED_DATA_PATH}/interaction_df.pickle')

        def merge_books(interaction_df):
            return interaction_df.merge(
                self.books_df_collapsed[['title', 'collapse_id', 'year_value', 'author_fullName', 'author_id',
                                         'rubric_id', 'ageRestriction_id', 'ageRestriction', 'outputCount']],
                on=['collapse_id'],
                how='left'
            )

        self.interaction_df_lt16 = merge_books(interaction_df_lt16)
        self.interaction_df_gte16 = merge_books(interaction_df_gte16)
        self.interaction_df = merge_books(interaction_df)

        self.collapse_id_index_lt16 = self.interaction_df_lt16.collapse_id.unique()
        self.collapse_id_index_gte16 = self.interaction_df_gte16.collapse_id.unique()

        self.lfm_model_lt16 = pd.read_pickle(f'{settings.MODELS_DATA_PATH}/lfm-model-lt16.pickle')
        self.lfm_model_gte16 = pd.read_pickle(f'{settings.MODELS_DATA_PATH}/lfm-model-gte16.pickle')

    def get_age_distribution(self, history):
        history = history[history['ageRestriction'] != -1]
        stat = (
            history.groupby('ageRestriction')
            .agg({'ageRestriction': 'count'})
            .rename(columns={'ageRestriction': 'count'})
            .reset_index()
        )
        stat['ratio'] = stat['count'] / history.shape[0]

        return {
            'gte16': stat[stat['ageRestriction'] >= 16]['ratio'].sum(),
            'lt16': stat[stat['ageRestriction'] < 16]['ratio'].sum()
        }

    def get_recommends_by_author(self, author_id: str, known_collapse_id_list: typing.List[str]):
        """ Возвращает популярные книги автора, с которыми пользователь еще не взаимодействовал  """

        author_books_df = self.books_df_collapsed[
            (self.books_df_collapsed['author_id'] == author_id) &
            (~self.books_df_collapsed['collapse_id'].isin(known_collapse_id_list))
        ]

        top_author_books_df = (
            author_books_df
            .groupby('collapse_id')
            .agg({'outputCount': 'sum'})
            .reset_index()
            .rename(columns={'outputCount': 'outputCount_sum'})
        )

        author_books_df = (
            author_books_df
            .merge(top_author_books_df, on='collapse_id', how='inner')
            .sort_values(by='outputCount_sum', ascending=False)  # первые - популярные
        )

        return author_books_df

    def get_lfm_recommendations(self, last_record, known_collapse_id_list: typing.List[str]) -> pd.DataFrame:
        """
        Находит похожие книги через Cosine similarity, перемножая вектора книг из LightFM
        https://github.com/lyst/lightfm/issues/244
        """

        if last_record['ageRestriction'] == -1:
            return pd.DataFrame()

        if last_record['ageRestriction'] >= 16:
            collapse_id_index = self.collapse_id_index_gte16
            lfm_model = self.lfm_model_gte16
        else:
            collapse_id_index = self.collapse_id_index_lt16
            lfm_model = self.lfm_model_lt16

        # получение внутреннего индекса item-а в LightFM
        item_id = np.where(collapse_id_index == last_record.collapse_id)[0][0]

        N = 1000
        (_, item_representations) = lfm_model.get_item_representations()

        # Cosine similarity
        scores = item_representations.dot(item_representations[item_id])
        item_norms = np.linalg.norm(item_representations, axis=1)
        item_norms[item_norms == 0] = 1e-10
        scores /= item_norms
        best = np.argpartition(scores, -N)[-N:]

        best_indexes = sorted(zip(best, scores[best] / item_norms[item_id]), key=lambda x: -x[1])

        lfm_books_df = self.books_df_collapsed[
            self.books_df.collapse_id.isin(
                np.array(collapse_id_index)[[idx for (idx, score) in best_indexes]]
            )
        ]
        lfm_books_df = lfm_books_df[~lfm_books_df['collapse_id'].isin(known_collapse_id_list)]

        # выберем книги из той же рубрики
        lfm_books_df = lfm_books_df[lfm_books_df['rubric_id'] == last_record.rubric_id]

        if last_record['ageRestriction'] >= 16:
            lfm_books_df = lfm_books_df[lfm_books_df['ageRestriction'] >= 16]
        elif last_record['ageRestriction'] == 12:
            lfm_books_df = lfm_books_df[lfm_books_df['ageRestriction'].between(6, 12)]
        elif last_record['ageRestriction'] <= 6:
            lfm_books_df = lfm_books_df[lfm_books_df['ageRestriction'] <= 6]

        # import ipdb; ipdb.set_trace();
        print('*'*100)
        print('Book:')
        print(last_record[['title', 'author_fullName']])
        print('---')
        print('Recommendation:')
        print(lfm_books_df.head(1).iloc[0][['title', 'author_fullName']])
        print('*'*100)
        print('\n')

        return lfm_books_df.head(1)

    def get_history(self, user_id):
        """ Возвращает историю пользователя. Любые взаимодействия. """

        return (
            self.interaction_df[self.interaction_df['readerID'] == user_id]
            .sort_values('startDate', ascending=True)  # последние - свежие
        )

    def get_top_in_rubric(self, rubric_id):
        """ Возвращает топ книг из рубрики основываясь на `outputCount` """

        rubric_only_df = self.books_df_collapsed[self.books_df_collapsed.rubric_id == rubric_id]
        top_in_rubric_df = (
            rubric_only_df
            .groupby('collapse_id')
            .agg({'outputCount': 'sum'})
            .reset_index()
            .rename(columns={'outputCount': 'outputCount_sum'})
            .sort_values(by='outputCount_sum')
        )
        rubric_only_df = (
            rubric_only_df
            .merge(top_in_rubric_df, on='collapse_id', how='left')
            .sort_values(by='outputCount_sum', ascending=False)  # первые - популярные
        )
        return rubric_only_df

    def recommend(self, history):
        # убираем дубликаты, оставляя свежее
        history = history.iloc[::-1].drop_duplicates('collapse_id').iloc[::-1]

        known_collapse_id_list = list(history.collapse_id)
        res = pd.DataFrame()

        age_distribution = self.get_age_distribution(history)

        # брать последние книги из этих кластеров и рекомендовать по ним
        if age_distribution['gte16'] <= 0.25:
            history_gte16 = history[history['ageRestriction'] >= 16].tail(1)
            history_lt16 = history[history['ageRestriction'] < 16].tail(2)
        elif age_distribution['gte16'] <= 0.5:
            history_gte16 = history[history['ageRestriction'] >= 16].tail(2)
            history_lt16 = history[history['ageRestriction'] < 16].tail(1)
        else:
            history_gte16 = history[history['ageRestriction'] >= 16].tail(3)
            history_lt16 = history[history['ageRestriction'] < 16].tail(0)

        age_awared_history = pd.concat([history_gte16, history_lt16])
        history = history[~history['collapse_id'].isin(age_awared_history['collapse_id'])]

        history = pd.concat([history, age_awared_history])

        # пробуем получить 3 рекомендации по 3-м последним книгам в истории
        for i in range(-1, -len(history) - 1, -1):
            last_record = history.iloc[i]
            lfm_recommendations = self.get_lfm_recommendations(
                last_record,
                known_collapse_id_list=known_collapse_id_list
            )
            if len(lfm_recommendations) == 0:
                continue

            res = pd.concat([res, lfm_recommendations])

            known_collapse_id_list.extend(list(lfm_recommendations['collapse_id']))

            if len(res) >= self.MAX_LFM_RECOMMENDATIONS:
                break

        # пробуем получить оставшиеся 2 книги алгоритмическим подходом
        for i in range(-1, -len(history)-1, -1):
            last_record = history.iloc[i]
            books_by_author = self.get_recommends_by_author(
                author_id=last_record['author_id'],
                known_collapse_id_list=known_collapse_id_list
            ).head(2)

            res = pd.concat([res, books_by_author])
            if len(res) >= self.MAX_RECOMMENDATIONS:
                break

            known_collapse_id_list.extend(list(books_by_author['collapse_id']))

            if len(res) >= self.MAX_RECOMMENDATIONS:
                break

        # если 5 книг не набралось, дополняем книгами, подготовленными для холодного старта
        if len(res) < self.MAX_RECOMMENDATIONS:
            # 479;Художественная литература
            # 496;Историческая и приключенческая литература
            # 534;Литература для детей и юношества
            # 551;Зарубежная художественная литература для детей и юношества
            # 511;Фэнтэзи
            rubric_df = pd.concat([
                self.get_top_in_rubric(rubric_id=479).head(10),
                self.get_top_in_rubric(rubric_id=496).head(10),
                self.get_top_in_rubric(rubric_id=534).head(10),
                self.get_top_in_rubric(rubric_id=551).head(10),
                self.get_top_in_rubric(rubric_id=511).head(10),
            ]).sample((self.MAX_RECOMMENDATIONS - len(res)))
            res = pd.concat([res, rubric_df])

        return res.head(5)
