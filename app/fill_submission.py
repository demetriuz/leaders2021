import csv
import pandas as pd
import predict
import settings
import tqdm


FROM_USER_ID = 0


def main():
    predictor = predict.Predictor()

    df = pd.read_pickle(f'{settings.PREPARED_DATA_PATH}/interaction_df.pickle')

    readerID_list = sorted(df[df['readerID'] > FROM_USER_ID]['readerID'].unique())

    with open(f'{settings.PREPARED_DATA_PATH}/submission_full.csv', 'a+') as f:
        writer = csv.writer(f, delimiter=';')

        if FROM_USER_ID == 0:
            writer.writerow(['user_id', 'book_id_1', 'book_id_2', 'book_id_3', 'book_id_4', 'book_id_5'])

        i = 0
        for user_id in tqdm.tqdm(readerID_list, total=len(readerID_list)):
            history = predictor.get_history(user_id)
            age = predictor.get_age(user_id)
            recommendations = predictor.recommend(history=history, user_age=age)

            writer.writerow([user_id] + [recommendations.iloc[i].recId for i in range(5)])

            i += 1
            # if i > 10:
            #     break


if __name__ == '__main__':
    main()
