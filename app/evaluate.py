import predict
from tqdm import tqdm
import pandas as pd
import settings


def main():
    predictor = predict.Predictor()
    validation_dict = pd.read_pickle(f'{settings.PREPARED_DATA_PATH}/validation_dict.pickle')

    total_hits = 0
    user_hits = 0

    N = 1000

    validate_readerID_list = validation_dict.keys()

    for user_id in tqdm(validate_readerID_list, total=N):
        history = predictor.get_history(user_id)
        history_validation = history.tail(5)

        cutted_history = history[~history['collapse_id'].isin(history_validation['collapse_id'])]

        recommendations_df = predictor.recommend(history=cutted_history).head(5)

        item_hits = len(set(history_validation['recId']).intersection(recommendations_df['recId']))
        total_hits += item_hits
        if item_hits:
            user_hits += 1

    print('total_hits:', total_hits)
    print('user_hits:', user_hits)


if __name__ == '__main__':
    main()
