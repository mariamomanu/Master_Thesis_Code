import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pickle
import os



USE_CALIBRATION = False

VIZ_LOCATION = 'model_results.png'
FEATURE_DATA = 'extracted_features.csv'
BEST_MODEL = 'best_model.pkl'
FEATURE_IMPORTANCE_PLOT = 'feature_importance.png'


class LiquidLevelPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.baseline = None

    def load_data(self, features_file):
        df = pd.read_csv(features_file)

        exclude_cols = ['liquid_level', 'container_type', 'liquid_type', 'timestamp', 'filename']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        df[feature_cols] = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

        # calibration: subtract per-vessel empty baseline
        if USE_CALIBRATION:
            empty = df[df['liquid_level'] == 0]
            self.baseline = empty.groupby('container_type')[feature_cols].mean()
            for vessel in df['container_type'].unique():
                if vessel in self.baseline.index:
                    mask = df['container_type'] == vessel
                    df.loc[mask, feature_cols] -= self.baseline.loc[vessel].values
        else:
            self.baseline = None

        X = df[feature_cols].values
        y = df['liquid_level'].values

        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        for level in sorted(df['liquid_level'].unique()):
            print(f"{level}%: {len(df[df['liquid_level'] == level])} samples")

        return X, y, feature_cols

    def prepare_data(self, X, y, test_size = 0.2, random_state = 93):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = test_size, random_state = random_state, stratify = y
        )
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def cross_validate_models(self, X, y, cv = 5):
        print(f"\nCross-validation ({cv}-fold)")
        print("=" * 50)

        model_defs = {
            'Random Forest': RandomForestRegressor(
                n_estimators = 100, max_depth = 20, min_samples_split = 5,
                random_state = 93, n_jobs = -1
            ),
            'SVM': SVR(kernel = 'rbf', C = 100, gamma = 'scale', epsilon = 0.1),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes = (100, 50, 25), activation = 'relu',
                solver = 'adam', alpha = 0.001, max_iter = 1000,
                random_state = 93, early_stopping = True, validation_fraction = 0.2
            ),
        }

        cv_results = {}
        for name, model in model_defs.items():
            pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])

            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=93)

            mae_scores = -cross_val_score(pipe, X, y, cv=skf, scoring='neg_mean_absolute_error')
            r2_scores  =  cross_val_score(pipe, X, y, cv=skf, scoring='r2')

            cv_results[name] = {
                'mae_mean': mae_scores.mean(),
                'mae_std': mae_scores.std(),
                'r2_mean': r2_scores.mean(),
                'r2_std': r2_scores.std(),
            }

            print(f"\n{name}:")
            print(f"MAE: {mae_scores.mean():.3f}% ± {mae_scores.std():.3f}%")
            print(f"R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

        return cv_results

    def train_random_forest(self, X_train, y_train):
        print("Training Random Forest...")
        model = RandomForestRegressor(
            n_estimators= 100, max_depth = 20, min_samples_split = 5,
            random_state = 93, n_jobs =-1
        )
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        return model

    def train_svm(self, X_train, y_train):
        print("Training SVM...")
        model = SVR(kernel = 'rbf', C = 100, gamma = 'scale', epsilon = 0.1)
        model.fit(X_train, y_train)
        self.models['SVM'] = model
        return model

    def train_neural_network(self, X_train, y_train):
        print("Training Neural Network...")
        model = MLPRegressor(
            hidden_layer_sizes = (100, 50, 25), activation = 'relu',
            solver = 'adam', alpha = 0.001, max_iter = 1000,
            random_state = 93, early_stopping = True, validation_fraction = 0.2
        )
        model.fit(X_train, y_train)
        self.models['Neural Network'] = model
        return model

    def evaluate_model(self, model, X_test, y_test, model_name):
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        errors = np.abs(predictions - y_test)
        within_1 = np.mean(errors <= 1)  * 100
        within_5 = np.mean(errors <= 5)  * 100
        within_10 = np.mean(errors <= 10) * 100

        return {
            'model': model_name,
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'within_1': within_1, 'within_5': within_5, 'within_10': within_10
        }, predictions


    def train_all_models(self, X_train, y_train, X_test, y_test, cv_results=None):
        print("\nTraining all models...")
        self.train_random_forest(X_train, y_train)
        self.train_svm(X_train, y_train)
        self.train_neural_network(X_train, y_train)

        print("\nEvaluating results")
        print("=" * 100)

        all_results = []
        all_predictions = {}

        for model_name, model in self.models.items():
            results, predictions = self.evaluate_model(model, X_test, y_test, model_name)
            all_results.append(results)
            all_predictions[model_name] = predictions

            print(f"\n{model_name}:")
            print(f"{'Metric':<8} {'Test set':>10}  {'CV mean ± std':>22}")
            print(f"{'-'*44}")

            cv = cv_results.get(model_name) if cv_results else None

            mae_cv = (f"{cv['mae_mean']:.3f}% ± {cv['mae_std']:.3f}%" if cv else "n/a")
            r2_cv = (f"{cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}" if cv else "n/a")

            print(f"{'MAE':<8} {results['mae']:>9.3f}%  {mae_cv:>22}")
            print(f"{'R²':<8} {results['r2']:>10.4f}  {r2_cv:>22}")
            print(f"{'RMSE':<8} {results['rmse']:>9.3f}%")
            print(f"Within ±1%: {results['within_1']:.1f}%")
            print(f"Within ±5%: {results['within_5']:.1f}%")
            print(f"Within ±10%: {results['within_10']:.1f}%")

        results_df = pd.DataFrame(all_results)
        best_idx = results_df['mae'].idxmin()
        self.best_model_name = results_df.loc[best_idx, 'model']
        self.best_model = self.models[self.best_model_name]

        print(f"\nBEST MODEL: {self.best_model_name}")
        print(f"MAE: {results_df.loc[best_idx, 'mae']:.3f}%")

        return results_df, all_predictions, y_test


    def plot_results(self, results_df, all_predictions, y_test, cv_results = None):
        print("Plotting results.")

        has_cv = cv_results is not None
        fig = plt.figure(figsize=(18, 10))

        colors = ['#3498db', '#e74c3c', '#2ecc71']
        colors_light = ['#85c1e9', '#f1948a', '#82e0aa']

        # MAE bar
        ax1 = plt.subplot(2, 4, 1)
        ax1.bar(results_df['model'], results_df['mae'], color = colors)
        ax1.set_ylabel('MAE (%)')
        ax1.set_title('Model Comparison - MAE', fontweight = 'bold')
        ax1.set_ylim([0, max(results_df['mae']) * 1.3])
        for i, v in enumerate(results_df['mae']):
            ax1.text(i, v / 2, f'{v:.2f}%', ha = 'center', fontweight = 'bold')

        # R² bar
        ax2 = plt.subplot(2, 4, 2)
        ax2.bar(results_df['model'], results_df['r2'], color = colors)
        ax2.set_ylabel('R² Score')
        ax2.set_title('Model Comparison - R² Score', fontweight = 'bold')
        ax2.set_ylim([0, 1])
        for i, v in enumerate(results_df['r2']):
            ax2.text(i, v / 2, f'{v:.3f}', ha = 'center', fontweight = 'bold')

        # accuracy within tolerance
        ax3 = plt.subplot(2, 4, 3)
        x = np.arange(len(results_df))
        width = 0.25
        ax3.bar(x - width, results_df['within_1'],  width, label = '±1%',  color = '#2ecc71')
        ax3.bar(x, results_df['within_5'],  width, label = '±5%',  color = '#3498db')
        ax3.bar(x + width, results_df['within_10'], width, label = '±10%', color = '#e67e22')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Accuracy Within Tolerance', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(results_df['model'], ha = 'center')
        ax3.legend(fontsize = 8)
        ax3.set_ylim([0, 110])
        for i, v in enumerate(results_df['within_1']):
            ax3.text(i - width, v / 2, f'{v:.1f}', ha = 'center', fontsize = 7)
        for i, v in enumerate(results_df['within_5']):
            ax3.text(i, v / 2, f'{v:.1f}', ha = 'center', fontsize= 7)
        for i, v in enumerate(results_df['within_10']):
            ax3.text(i + width, v / 2, f'{v:.1f}', ha = 'center', fontsize = 7)

        # CV vs test MAE
        ax4 = plt.subplot(2, 4, 4)
        if has_cv:
            cv_means = [cv_results[m]['mae_mean'] for m in results_df['model']]
            cv_stds = [cv_results[m]['mae_std']  for m in results_df['model']]
            w = 0.35
            ax4.bar(x - w/2, results_df['mae'], w,
                    label='Test set', color = colors)
            ax4.bar(x + w/2, cv_means, w, yerr = cv_stds,
                    label='CV mean ± std', color = colors_light, capsize = 5)
            ax4.set_ylabel('MAE (%)')
            ax4.set_title('Test vs Cross-Validation MAE', fontweight = 'bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(results_df['model'])
            ax4.legend(fontsize = 8)
            ax4.grid(True, alpha = 0.3, axis = 'y')
        else:
            ax4.set_visible(False)

        # predicted vs. actual scatter
        for idx, (model_name, predictions) in enumerate(all_predictions.items()):
            ax = plt.subplot(2, 4, 5 + idx)
            ax.scatter(y_test, predictions, alpha=0.6, s=30, color=colors[idx])
            ax.plot([0, 100], [0, 100], 'k--', lw=2, label='Perfect prediction')
            ax.set_xlabel('Actual Level (%)')
            ax.set_ylabel('Predicted Level (%)')
            ax.set_title(model_name, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([-5, 105])
            ax.set_ylim([-5, 105])

        plt.tight_layout()
        plt.savefig(VIZ_LOCATION, dpi = 300, bbox_inches = 'tight')
        print(f"Saved: {VIZ_LOCATION}")
        plt.show()

    def save_best_model(self, filename):
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'baseline': self.baseline,
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Saved: {filename}")

    def save_all_models(self, folder = 'Final_more_data', type = 'water'):
        for name, model in self.models.items():
            fname = os.path.join(
                folder, f"model_{type}_{name.replace(' ', '_').lower()}.pkl"
            )
            with open(fname, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': self.scaler,
                    'model_name': name,
                    'baseline': self.baseline,
                }, f)
            print(f"Saved: {fname}")

    def predict_liquid_level(self, features):
        scaled = self.scaler.transform(features.reshape(1, -1))
        return self.best_model.predict(scaled)[0]
    
    def plot_feature_importance(self, feature_cols, top_n = 15):

        if not hasattr(self.best_model, 'feature_importances_'):
            print("Feature importance only available for Random Forest.")
            return

        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        top_features = [feature_cols[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(10, 6))
        plt.barh(range(top_n), top_importances[::-1], color = '#3498db')
        plt.yticks(range(top_n), top_features[::-1])
        plt.xlabel('Feature Importance (Gini)')
        plt.title(f'Top {top_n} Most Important Features — {self.best_model_name}',
                fontweight='bold')
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_PLOT, dpi = 300, bbox_inches = 'tight')
        plt.show()

        print(f"\nTop {top_n} features:")
        for i, (feat, imp) in enumerate(zip(top_features, top_importances)):
            print(f"  {i+1:>2}. {feat:<40} {imp:.4f}")


def main():
    print("Liquid level model pipeline")
    print("=" * 100)

    predictor = LiquidLevelPredictor()

    # load data
    X, y, feature_cols = predictor.load_data(FEATURE_DATA)
    feature_df = pd.DataFrame(X, columns=feature_cols)
    print(feature_df.max().sort_values(ascending=False).head(10))

    # cross-validate on full dataset (leakage-free via Pipeline)
    cv_results = predictor.cross_validate_models(X, y, cv=5)

    # train/test split and final model training
    X_train, X_test, y_train, y_test = predictor.prepare_data(X, y)
    results_df, all_predictions, y_test = predictor.train_all_models(
        X_train, y_train, X_test, y_test, cv_results=cv_results
    )

    predictor.plot_feature_importance(feature_cols, top_n=15)

    # plot (CV panel included)
    predictor.plot_results(results_df, all_predictions, y_test,
                           cv_results=cv_results)

    # save
    predictor.save_best_model(BEST_MODEL)
    # predictor.save_all_models('Final_more_data', 'water_oil')

    print("\n" + "=" * 100)
    print("TRAINING COMPLETE")
    print("=" * 100)
    print(f"Best model: {predictor.best_model_name}")
    print(f"Model saved: {BEST_MODEL}")
    print(f"Plots saved: {VIZ_LOCATION}")


if __name__ == "__main__":
    main()