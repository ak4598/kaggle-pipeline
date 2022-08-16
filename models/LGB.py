from models import Hyperopt
from lightgbm import LGBMClassifier


class LGB(Hyperopt):
    def __init__(self, cfg):
        super(LGB, self).__init__(cfg, LGBMClassifier)

    def save(self):
        print('Saving best model...')

        dt = datetime.now().strftime('%Y%m%d%H%M%S')

        final_score_str = "%.6f" % (self.score)

        dataset_path = Path(__file__).parent.parent.parent.resolve() / \
            'data' / 'Amex' / 'dataset' / 'output'

        output_dir = "LGB_%s_%s" % (final_score_str, dt)
        output_path = Path(__file__).parent.parent.parent.resolve() / \
            'output' / 'Amex' / output_dir

        os.mkdir(output_path)

        model_sav = "LGB_%s_%s.sav" % (final_score_str, dt)
        pickle.dump(self.model, open(output_path / model_sav, 'wb'))

        print("Best model saved.")

        print('Reading kaggle test data for prediction...')
        kaggle_test = pd.read_parquet(
            dataset_path / 'test_data_reduced.parquet')

        print('Predicting output...')
        submit_df = kaggle_test[['customer_ID']]

        submit_df['prediction'] = self.model.predict_proba(
            kaggle_test.drop(['customer_ID'], axis=1))[:, 1]

        print('Saving output to csv...')
        submission_csv = "LGB_%s_%s.csv" % (final_score_str, dt)
        submit_df.to_csv(
            output_path / submission_csv, index=False)

        print("Saved.")
