import predict
import settings
import pandas as pd


def main():
    predictor = predict.Predictor()

    knigi_df = pd.read_pickle(f'{settings.PREPARED_DATA_PATH}/knigi_df.pickle')

    recs_list = []

    for user_id in knigi_df.readerID.unique():
        print(f'get recommendations for user: {user_id}')

        history = predictor.get_history(user_id)
        recommendations = predictor.recommend(history=history)

        books_id_map = {f'book_id_{i+1}': recommendations.iloc[i].recId for i in range(5)}
        recs_ = pd.DataFrame(data={'user_id': [user_id], **books_id_map})

        recs_list.append(recs_)

    res = pd.concat(recs_list)
    res.to_csv(f'{settings.PREPARED_DATA_PATH}/submission.csv', sep=';', index=False)
    return res


if __name__ == '__main__':
    main()
