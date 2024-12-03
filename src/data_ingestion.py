import pandas as pd

def get_stratified_data(filename, n_sample=2_000, seed=None) -> pd.DataFrame:
    df = pd.read_csv(filename)
    grouped_df = df.groupby('label')
    return grouped_df.apply(lambda x: x.sample(n_sample // 2, random_state=seed)).droplevel(0)
