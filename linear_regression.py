import numpy as np
from scipy import stats


class LinearRegression:
    def __init__(self, nan_strategy="drop"):
        """
        nan_strategy:
            "drop"  -> ta bort rader med NaN
            "mean"  -> ersätt NaN med kolumnmedelvärde
        """
        self.nan_strategy = nan_strategy
        self.beta = None
        self.X = None
        self.Y = None
        self.n = None
        self.d = None
        self.sigma2_hat = None
        self.cov_beta = None
        self.SSE = None

    # -------------------------------------------------
    # Intern funktion för att hantera NaN
    # -------------------------------------------------
    def _handle_nan(self, X, Y):
        data = np.column_stack((X, Y))

        if self.nan_strategy == "drop":
            data = data[~np.isnan(data).any(axis=1)]

        elif self.nan_strategy == "mean":
            col_means = np.nanmean(data, axis=0)
            inds = np.where(np.isnan(data))
            data[inds] = np.take(col_means, inds[1])

        else:
            raise ValueError("nan_strategy must be 'drop' or 'mean'")

        X_clean = data[:, :-1]
        Y_clean = data[:, -1]

        return X_clean, Y_clean

    # -------------------------------------------------
    # OLS (stabil version med pseudoinvers)
    # -------------------------------------------------
    def fit(self, X, Y):

        # Hantera NaN
        X, Y = self._handle_nan(X, Y)

        # Lägg till intercept
        X = np.column_stack((np.ones(X.shape[0]), X))

        self.X = X
        self.Y = Y.reshape(-1, 1)

        self.n = X.shape[0]
        self.d = X.shape[1] - 1  # exkl. intercept

        # ===== OLS-SKATTNING =====
        # beta = X^+ Y
        X_pinv = np.linalg.pinv(X)
        self.beta = X_pinv @ self.Y

        # Residualer
        residuals = self.Y - X @ self.beta

        # SSE
        self.SSE = float(residuals.T @ residuals)

        # ===== Variansskattning =====
        self.sigma2_hat = self.SSE / (self.n - self.d - 1)

        # ===== Kovariansmatris =====
        XtX_inv = np.linalg.pinv(X.T @ X)
        self.cov_beta = self.sigma2_hat * XtX_inv

    # -------------------------------------------------
    # Prediktion
    # -------------------------------------------------
    def predict(self, X):
        X = np.column_stack((np.ones(X.shape[0]), X))
        return X @ self.beta

    # -------------------------------------------------
    # Stickprovsvarians
    # -------------------------------------------------
    def sample_variance(self):
        residuals = self.Y - self.X @ self.beta
        return float((residuals.T @ residuals) / (self.n - 1))

    # -------------------------------------------------
    # Standardavvikelse
    # -------------------------------------------------
    def standard_deviation(self):
        return np.sqrt(self.sample_variance())

    # -------------------------------------------------
    # RMSE
    # -------------------------------------------------
    def rmse(self):
        return np.sqrt(self.SSE / self.n)

    # -------------------------------------------------
    # R^2
    # -------------------------------------------------
    def r_squared(self):
        y_mean = np.mean(self.Y)
        Syy = float(((self.Y - y_mean).T @ (self.Y - y_mean)))
        SSR = Syy - self.SSE
        return SSR / Syy

    # -------------------------------------------------
    # F-test (hela modellen)
    # -------------------------------------------------
    def f_test(self):
        y_mean = np.mean(self.Y)
        Syy = float(((self.Y - y_mean).T @ (self.Y - y_mean)))
        SSR = Syy - self.SSE

        F_stat = (SSR / self.d) / self.sigma2_hat
        p_value = stats.f.sf(F_stat, self.d, self.n - self.d - 1)

        return F_stat, p_value

    # -------------------------------------------------
    # t-test (enskilda parametrar)
    # -------------------------------------------------
    def t_tests(self):
        t_values = []
        p_values = []

        df = self.n - self.d - 1

        for i in range(len(self.beta)):
            se = np.sqrt(self.cov_beta[i, i])
            t_stat = float(self.beta[i] / se)

            p = 2 * min(
                stats.t.cdf(t_stat, df),
                stats.t.sf(t_stat, df)
            )

            t_values.append(t_stat)
            p_values.append(p)

        return np.array(t_values), np.array(p_values)

    # -------------------------------------------------
    # Konfidensintervall
    # -------------------------------------------------
    def confidence_intervals(self, alpha=0.05):
        intervals = []
        df = self.n - self.d - 1
        t_crit = stats.t.ppf(1 - alpha / 2, df)

        for i in range(len(self.beta)):
            se = np.sqrt(self.cov_beta[i, i])
            lower = float(self.beta[i] - t_crit * se)
            upper = float(self.beta[i] + t_crit * se)
            intervals.append((lower, upper))

        return intervals
