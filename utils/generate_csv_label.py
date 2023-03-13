import pandas as pd
import os


def _generate_csv_label(root_dir, csv_label_path):
    train_folder = os.path.join(root_dir, 'extracted', 'train')
    csvs = os.listdir(train_folder)

    label_info = {}
    label_info['action'] = []
    label_info['label'] = []
    count_i = 0
    for base_name in csvs:
        if '.DS_Store' in base_name:
            continue
        label_info['action'].append(base_name)
        label_info['label'].append(count_i)
        count_i += 1
    label_pd = pd.DataFrame.from_dict(label_info)
    label_pd.to_csv(csv_label_path)