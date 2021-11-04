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
                          'rubric_id', 'ageRestriction_id', 'outputCount']
        books_df = pd.read_pickle(f'{settings.PREPARED_DATA_PATH}/books_full_df.pickle')[books_features]
        self.books_df = books_df
        self.books_df_collapsed = books_df.drop_duplicates('collapse_id')

        # TODO: config
        interaction_df = pd.read_pickle(f'{settings.PREPARED_DATA_PATH}/interaction_df_train.pickle')
        interaction_df_full = pd.read_pickle(f'{settings.PREPARED_DATA_PATH}/interaction_df.pickle')

        self.interaction_df = interaction_df.merge(
            self.books_df_collapsed[['title', 'collapse_id', 'year_value', 'author_fullName', 'author_id',
                                     'rubric_id', 'ageRestriction_id', 'outputCount']],
            on=['collapse_id'],
            how='left'
        )

        self.interaction_df_full = interaction_df_full.merge(
            self.books_df_collapsed[['title', 'collapse_id', 'year_value', 'author_fullName', 'author_id',
                                     'rubric_id', 'ageRestriction_id', 'outputCount']],
            on=['collapse_id'],
            how='left'
        )

        self.collapse_id_index = interaction_df.collapse_id.unique()
        self.lfm_model = pd.read_pickle(f'{settings.MODELS_DATA_PATH}/lfm-model.pickle')

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

        # получение внутреннего индекса item-а в LightFM
        item_id = np.where(self.collapse_id_index == last_record.collapse_id)[0][0]

        N = 1000
        (_, item_representations) = self.lfm_model.get_item_representations()

        # Cosine similarity
        scores = item_representations.dot(item_representations[item_id])
        item_norms = np.linalg.norm(item_representations, axis=1)
        item_norms[item_norms == 0] = 1e-10
        scores /= item_norms
        best = np.argpartition(scores, -N)[-N:]

        best_indexes = sorted(zip(best, scores[best] / item_norms[item_id]), key=lambda x: -x[1])

        lfm_books_df = self.books_df_collapsed[
            self.books_df.collapse_id.isin(
                np.array(self.collapse_id_index)[[idx for (idx, score) in best_indexes]]
            )
        ]
        lfm_books_df = lfm_books_df[~lfm_books_df['collapse_id'].isin(known_collapse_id_list)]

        # выберем книги из той же рубрики
        lfm_books_df = lfm_books_df[lfm_books_df['rubric_id'] == last_record.rubric_id]

        # # корректируем рекомендации в зависимости от возрастных ограничений
        allowed_age_restrictions_map = {
            0: [0, 6630, 6634, 6633],  # not defined
            6632: [6631, 6632],  # 18+
            6631: [6631, 6632],  # 16+
            6633: [6634, 6633],  # 12+
            6634: [6634, 6630],  # 6+
            6630: [6630, 6634],  # 0+
        }
        if last_record.ageRestriction_id in allowed_age_restrictions_map.keys():
            lfm_books_df = lfm_books_df[
                lfm_books_df.ageRestriction_id.isin(allowed_age_restrictions_map[last_record.ageRestriction_id])
            ]

        return lfm_books_df.head(1)

    def get_history(self, user_id):
        """ Возвращает историю пользователя. Любые взаимодействия. """

        return (
            self.interaction_df_full[self.interaction_df_full['readerID'] == user_id]
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

        # пробуем получить 3 рекомендации по 3-м последним книгам в истории
        for i in range(-1, -len(history) - 1, -1):
            last_record = history.iloc[i]
            lfm_recommendations = self.get_lfm_recommendations(
                last_record,
                known_collapse_id_list=known_collapse_id_list
            )
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
