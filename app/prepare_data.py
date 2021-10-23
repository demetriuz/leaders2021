import pandas as pd
import settings


def prepare_books_full(circulation_df, knigi_df):
    book_df_fields = [
        'id',
        'title',
        'smart_collapse_field',
        'year_value',
        'author_fullName',
        'author_id',
        'rubric_id',
        'ageRestriction_id',
        'outputCount',
    ]

    books_full_df = pd.read_csv(f'{settings.RAW_DATA_PATH}/books_full_df.csv.zip')[book_df_fields]
    books_df = pd.read_json(f'{settings.RAW_DATA_PATH}/books.json')[book_df_fields]
    books_full_df = pd.concat([books_full_df, books_df])

    books_full_df = books_full_df.reset_index(drop=True)

    books_full_df['id'] = books_full_df['id'].fillna(0).astype(int)
    books_full_df = books_full_df[
        books_full_df['id'].isin(circulation_df['recId']) |
        books_full_df['id'].isin(knigi_df['recId'])
    ]

    books_full_df['title'] = books_full_df['title'].fillna('')
    books_full_df['ageRestriction_id'] = books_full_df['ageRestriction_id'].fillna(0).astype(float).astype(int)
    books_full_df['year_value'] = books_full_df['year_value'].fillna(0).astype(float).astype(int)

    books_full_df = books_full_df.rename(columns={
        'id': 'recId',
        'smart_collapse_field': 'collapse_id'
    })

    return books_full_df


def load_circulation_df():
    circulation_df = pd.DataFrame()

    for i in range(1, 17):
        print(f'Read circulaton_{i}.csv')
        circulation_df_tmp = pd.read_csv(f'{settings.RAW_DATA_PATH}/circulaton_{i}.csv', delimiter=';', encoding='cp1251')
        circulation_df = pd.concat([circulation_df, circulation_df_tmp])

    circulation_df = circulation_df[['readerID', 'catalogueRecordID', 'startDate', 'state']]
    circulation_df = circulation_df[circulation_df.state == 'На руках']

    circulation_df['startDate'] = pd.to_datetime(circulation_df['startDate'])
    circulation_df = circulation_df.rename(columns={'catalogueRecordID': 'recId'})
    circulation_df = circulation_df[['readerID', 'recId', 'startDate']]

    return circulation_df


def load_knigi_df():
    knigi_df = pd.read_csv(f'{settings.RAW_DATA_PATH}/dataset_knigi_1.csv', delimiter=';')

    def url_to_id(row):
        return int(row['source_url'].rsplit('/', 2)[1])

    knigi_df['recId'] = knigi_df.apply(url_to_id, axis=1)
    knigi_df = knigi_df.rename(columns={'user_id': 'readerID', 'dt': 'startDate'})

    return knigi_df


def create_interaction_df(circulation_df, knigi_df, books_full_df):
    interaction_df = pd.concat([circulation_df, knigi_df[['readerID', 'recId', 'startDate']]])
    books_full_df = books_full_df.drop_duplicates(subset=['recId'])

    interaction_df = interaction_df.merge(
        books_full_df[['recId', 'collapse_id']],
        on=['recId'],
        how='left'
    )

    interaction_df['collapse_id'] = interaction_df.apply(lambda row: row['collapse_id'] if isinstance(row['collapse_id'], str) else row['recId'], axis=1)

    return interaction_df


if __name__ == '__main__':
    print('load_knigi_df...')
    knigi_df = load_knigi_df()
    knigi_df.to_pickle(f'{settings.PREPARED_DATA_PATH}/knigi_df.pickle')

    print('load_circulation_df...')
    circulation_df = load_circulation_df()
    print('circulation_df len: ', len(circulation_df))
    circulation_df.to_pickle(f'{settings.PREPARED_DATA_PATH}/circulation_df.pickle')

    print('prepare_books_full...')
    books_full_df = prepare_books_full(circulation_df, knigi_df)
    books_full_df.to_pickle(f'{settings.PREPARED_DATA_PATH}/books_full_df.pickle')

    print('create_interaction_df...')
    interaction_df = create_interaction_df(circulation_df, knigi_df, books_full_df)
    print('interaction_df len:', len(interaction_df))
    interaction_df.to_pickle(f'{settings.PREPARED_DATA_PATH}/interaction_df.pickle')
