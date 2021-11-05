import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset

import settings


def get_data(interaction_df, is_full_dataset=True):
    if not is_full_dataset:
        print('Before filtering: ', interaction_df.shape[0])

        validation_dict = pd.read_pickle(f'{settings.PREPARED_DATA_PATH}/validation_dict.pickle')
        for user, items in validation_dict.items():
            interaction_df = interaction_df[
                ~(
                    (interaction_df['readerID'] == user) &
                    (interaction_df['collapse_id'].isin(items))
                )
            ]

        interaction_df.to_pickle(f'{settings.PREPARED_DATA_PATH}/interaction_df_train.pickle')

        print('After filtering: ', interaction_df.shape[0])

    interaction_df = interaction_df.rename(columns={'readerID': 'user', 'collapse_id': 'item'})
    interaction_df = interaction_df[['user', 'item']]

    return interaction_df


def get_dataset(interaction_df):
    dataset = Dataset()
    dataset.fit(
        users=interaction_df['user'].unique(),
        items=interaction_df['item'].unique()
    )
    return dataset


def train_lfm(dataset, interaction_df):
    (interactions, _) = dataset.build_interactions([(x[0], x[1]) for x in interaction_df.values])

    model = LightFM(loss='warp', random_state=42)
    model.fit(
        interactions,
        epochs=300
    )
    return model


def main():
    for suffix in ['gte16', 'lt16']:
        interaction_df = pd.read_pickle(f'{settings.PREPARED_DATA_PATH}/interaction_df_{suffix}.pickle')
        interaction_df = get_data(interaction_df)

        dataset = get_dataset(interaction_df)
        lfm_model = train_lfm(
            dataset=dataset,
            interaction_df=interaction_df
        )
        pd.to_pickle(lfm_model, settings.MODELS_DATA_PATH.joinpath(f'lfm-model-{suffix}.pickle'))


if __name__ == '__main__':
    main()
