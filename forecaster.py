import pandas as pd
import duckdb
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

class SolarForecastForecaster:
    def __init__(self, data_dir: str = "blocks_by_day", output_csv: str = "aggregated_series.csv"):
        self.data_dir = Path(data_dir)
        self.conn = duckdb.connect(database=':memory:')
        self.output_csv = output_csv

    def load_aggregated_series(self) -> pd.Series:
        pattern = str(self.data_dir / '*' / '*.parquet')
        query = f"""
            SELECT
                STRPTIME(Datum_von, '%d.%m.%Y %H:%M') AS ts,
                SUM(Generation) AS total_output
            FROM read_parquet('{pattern}')
            GROUP BY ts
            ORDER BY ts
        """
        df = self.conn.execute(query).df()
        df['ts'] = pd.to_datetime(df['ts'])
        df.set_index('ts', inplace=True)
        return df['total_output']

    def _prepare_features(self, idx: pd.DatetimeIndex, svr_only: bool = False) -> np.ndarray:
        sec = (idx.astype('int64') - self.global_t0.value) / 1e9
        sec = sec.to_numpy()
        if svr_only:
            return sec.reshape(-1, 1)
        minutes = idx.hour * 60 + idx.minute
        sin_day = np.sin(2 * np.pi * minutes / 1440)
        cos_day = np.cos(2 * np.pi * minutes / 1440)
        doy = idx.dayofyear
        sin_year = np.sin(2 * np.pi * doy / 365)
        cos_year = np.cos(2 * np.pi * doy / 365)
        return np.vstack([sec, sin_day, cos_day, sin_year, cos_year]).T

    def train_svr(self, train: pd.Series, kernel: str = 'poly', C: float = 1.0, epsilon: float = 0.1) -> SVR:
        X = self._prepare_features(train.index, svr_only=True)
        y = train.to_numpy()
        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        model.fit(X, y)
        return model

    def train_xgb(self, train: pd.Series, **kwargs) -> XGBRegressor:
        X = self._prepare_features(train.index)
        y = train.to_numpy()
        model = XGBRegressor(**kwargs)
        model.fit(X, y)
        return model

    def forecast_svr(self, model: SVR, future_idx: pd.DatetimeIndex) -> pd.Series:
        Xf = self._prepare_features(future_idx, svr_only=True)
        return pd.Series(model.predict(Xf), index=future_idx)

    def forecast_xgb(self, model: XGBRegressor, future_idx: pd.DatetimeIndex) -> pd.Series:
        Xf = self._prepare_features(future_idx)
        return pd.Series(model.predict(Xf), index=future_idx)

    def run(
        self,
        train_ratio: float = 0.8,
        svr_kernel: str = 'rbf', svr_C: float = 1.0, svr_epsilon: float = 0.1,
        xgb_params: dict = None,
        horizon_ratio: float = 0.1,
        freq: str = '15T'
    ):
        series = self.load_aggregated_series().dropna()
        series.to_frame('total_output').to_csv(self.output_csv, index_label='ts')

        split = int(len(series) * train_ratio)
        train_full = series.iloc[:split]
        test = series.iloc[split:]

        xgb_params = xgb_params or {'n_estimators': 100, 'learning_rate': 0.1}
        self.global_t0 = train_full.index[0]

        svr_window = max(int(len(train_full) * horizon_ratio), 1)
        train_svr_data = train_full.iloc[-svr_window:]

        svr_model = self.train_svr(train_svr_data, kernel=svr_kernel, C=svr_C, epsilon=svr_epsilon)
        xgb_model = self.train_xgb(train_full, **xgb_params)

        steps = len(test)
        dates = pd.date_range(start=test.index[0], periods=steps, freq=freq)
        split_step = min(max(int(steps * horizon_ratio), 1), steps)
        svr_idx = dates[:split_step]
        xgb_idx = dates[split_step:]

        svr_preds = self.forecast_svr(svr_model, svr_idx) if len(svr_idx) else pd.Series(dtype=float)
        xgb_preds = self.forecast_xgb(xgb_model, xgb_idx) if len(xgb_idx) else pd.Series(dtype=float)
        preds = pd.concat([svr_preds, xgb_preds])

        mse = mean_squared_error(test.values, preds.values)

        plt.figure()
        ax = plt.gca()
        for single_date in pd.date_range(start=train_full.index.min().date(), end=preds.index.max().date(), freq='D'):
            day = pd.Timestamp(single_date)
            ax.axvspan(day, day + pd.Timedelta(hours=6), color='gray', alpha=0.2)
            ax.axvspan(day + pd.Timedelta(hours=18), day + pd.Timedelta(days=1), color='gray', alpha=0.2)
        ax.plot(train_full.index, train_full, label='Train (Full)')
        ax.plot(test.index, test, label='Test')
        ax.plot(preds.index, preds, label='Ensemble Forecast')
        ax.set_xlabel('Time')
        ax.set_ylabel('Total Output')
        ax.set_title(f'Ensemble MSE={mse:.2f}')
        ax.legend()
        plt.show()

        print(f'Mean Squared Error (Ensemble): {mse:.2f}')

if __name__ == '__main__':
    SolarForecastForecaster().run()
