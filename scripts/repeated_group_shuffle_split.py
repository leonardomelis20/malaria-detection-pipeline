import os
import pandas
from sklearn.model_selection import GroupShuffleSplit 

#ho provato a tenere un test separato ma ho ottenuto un CV poco affidabile con classe rare instabili

metadata_path = r'C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\mpidb_metadata.csv'

output_splits = r'C:\Users\laura\OneDrive\Documenti\UNIVCA\TERZO ANNO\TIROCINIO\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\csvs\splits'

if not os.path.exists(output_splits):
    os.makedirs(output_splits)

metadata_df = pandas.read_csv(metadata_path)

if 'filepath' not in metadata_df.columns or 'label' not in metadata_df.columns or 'group_id' not in metadata_df.columns or 'filename' not in metadata_df.columns:
    raise ValueError("The metadata CSV must contain 'filepath', 'label', 'group_id', and 'filename' columns.")

print(f"Totale campioni: {len(metadata_df)}, conteggio per classe: {metadata_df['label'].value_counts()}, Gruppi unici per classe: {metadata_df.groupby('label')['group_id'].nunique()}")

x = metadata_df['filepath']
y = metadata_df['label']
groups = metadata_df['group_id']

gss = GroupShuffleSplit(n_splits=200, test_size=0.25, random_state=42)

expected_labels = set(metadata_df['label'].unique())

attempt = 0

saved_run = 1

for train_idx, test_idx in gss.split(x, y, groups):
    train_df = metadata_df.iloc[train_idx]
    test_df = metadata_df.iloc[test_idx]

    #resetta gli indici di train_df e test_df
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_groups = set(train_df['group_id'])
    test_groups = set(test_df['group_id'])

    overlap_groups = train_groups.intersection(test_groups)

    if overlap_groups:
        print(f"Run {attempt}: Attenzione! I seguenti group_id sono presenti sia in train che in test: {overlap_groups}")
        attempt += 1
        continue

    train_labels = set(train_df['label'])
    test_labels = set(test_df['label'])

    missing_in_train = expected_labels - train_labels
    missing_in_test = expected_labels - test_labels

    if missing_in_train:
        print(f"Run {attempt}: Attenzione! Le seguenti classi sono assenti nel train set: {missing_in_train}")

    if missing_in_test:
        print(f"Run {attempt}: Attenzione! Le seguenti classi sono assenti nel test set: {missing_in_test}. ")
       
    if missing_in_train or missing_in_test:
        attempt += 1
        continue

    print(
        f"Run: {attempt}\n"
        f"  train samples size: {len(train_df)}\n"
        f"  test samples size: {len(test_df)}\n"
        f"  train groups size: {len(train_groups)}\n"
        f"  test groups size: {len(test_groups)}"
    )
    
    print(
        f"Num of Labels in train: {len(train_labels)}, "
        f"Num of Labels in test: {len(test_labels)}"
    )

    train_df = train_df.copy()
    test_df= test_df.copy()

    train_df['split'] = 'train'
    test_df['split'] = 'test'

    split_df = pandas.concat([train_df, test_df], ignore_index=True)

    print(split_df.columns)
    print(split_df.head())

    split_df = split_df.sort_values(by=['split', 'label', 'group_id', 'filename'])

    split_df = split_df.reset_index(drop=True)

    split_filename = f'split_run_{saved_run:02d}.csv'

    split_df.to_csv(os.path.join(output_splits, split_filename), index=False)

    
    print(f"Run {attempt} salvata in {os.path.join(output_splits, split_filename)}\n")
    attempt += 1
    saved_run += 1

    if saved_run >= 11:
        print("Sono state salvate 10 split, interrompo il ciclo.")
        break
    