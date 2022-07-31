import numpy as np
import pandas as pd


def amex_metric(y_true: np.array, y_pred: np.array) -> float:
    def top_four_percent_captured(y_true: np.array, y_pred: np.array) -> float:
        pd.DataFrame(y_pred, columns=['prediction'])
        df = (pd.concat([pd.DataFrame(y_true, columns=['target']), pd.DataFrame(y_pred, columns=['prediction'])], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()

    def weighted_gini(y_true: np.array, y_pred: np.array) -> float:
        df = (pd.concat([pd.DataFrame(y_true, columns=['target']), pd.DataFrame(y_pred, columns=['prediction'])], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: np.array, y_pred: np.array) -> float:
        y_true_pred = pd.DataFrame(y_true, columns=['target']).rename(
            columns={'target': 'prediction'})
        return weighted_gini(pd.DataFrame(y_true, columns=['target']), pd.DataFrame(y_pred, columns=['prediction'])) / weighted_gini(pd.DataFrame(y_true, columns=['target']), y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return -0.5 * (g + d)
