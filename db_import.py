import pandas as pd
import glob
import re
from pathlib import Path

class SolarForecastProcessor:
    def __init__(self, download_pattern: str, base_dir: str = "blocks_by_day"):
        self.download_pattern = download_pattern
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.generation_cols = []

    def load_and_clean_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep=';', decimal=',', na_values='-')
        df['Datum von'] = pd.to_datetime(df['Datum von'], format='%d.%m.%Y %H:%M')
        df['Datum bis'] = pd.to_datetime(df['Datum bis'], format='%d.%m.%Y %H:%M')
        return df

    def combine(self) -> pd.DataFrame:
        paths = glob.glob(self.download_pattern)
        frames = [self.load_and_clean_csv(p) for p in paths]
        df = pd.concat(frames, ignore_index=True)
        self.generation_cols = [c for c in df.columns if c.startswith('Generation_DE')]
        return df

    def save_daily_files(self, df: pd.DataFrame):
        for _, group in df.groupby(df['Datum von'].dt.date):
            date_str = group['Datum von'].dt.strftime('%Y-%m-%d').iloc[0]
            folder = self.base_dir / date_str
            folder.mkdir(parents=True, exist_ok=True)
            for col in self.generation_cols:
                sub_df = group[['Datum von', 'Datum bis', col]].copy()
                sub_df.rename(columns={col: 'Generation'}, inplace=True)
                sub_df['Datum_von'] = sub_df['Datum von'].dt.strftime('%d.%m.%Y %H:%M')
                sub_df['Datum_bis'] = sub_df['Datum bis'].dt.strftime('%d.%m.%Y %H:%M')
                sub_df = sub_df.drop(['Datum von', 'Datum bis'], axis=1)
                m = re.search(r"Neurath\s+([A-Za-z])", col)
                prefix = m.group(1) if m else 'X'
                safe = col.replace(' ', '_').replace('[', '').replace(']', '')
                filename = f"{prefix}_{safe}.parquet"
                path = folder / filename
                sub_df.to_parquet(path, index=False, engine='pyarrow', compression='snappy')

    def run(self):
        df = self.combine()
        self.save_daily_files(df)

if __name__ == '__main__':
    processor = SolarForecastProcessor(r"C:\Users\jfham\OneDrive\Desktop\Solar_Forecasting\downloads\*.csv")
    processor.run()
