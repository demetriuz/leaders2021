import predict
import settings
import pandas as pd
from tqdm import tqdm


def main():
    predictor = predict.Predictor()

    interaction_df_aggregated = (
        predictor.interaction_df_full
        .groupby('readerID')
        .agg({'readerID': 'count'})
        .rename(columns={'readerID': 'count'})
        .reset_index()
    )
    interaction_df_aggregated = interaction_df_aggregated[interaction_df_aggregated['count'] >= 10]

    N = 1000

    validation_dict = {}
    for user_id in tqdm(interaction_df_aggregated.sample(n=N, random_state=42)['readerID'], total=N):
        validation_dict[user_id] = predictor.get_history(user_id).tail(5)['collapse_id']

    pd.to_pickle(validation_dict, f'{settings.PREPARED_DATA_PATH}/validation_dict.pickle')


if __name__ == '__main__':
    main()
