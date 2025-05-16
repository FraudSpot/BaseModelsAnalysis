import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class FraudDetectionModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize the Fraud Detection Model with RandomForest classifier
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of trees in the forest
        max_depth : int, default=None
            Maximum depth of the trees
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.feature_importances = None
        
    def fit(self, X_train, y_train):
        """
        Train the model on the training data
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target values
        """
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Store feature importances
        self.feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
    
    def predict(self, X):
        """
        Make binary predictions using the trained model
        
        Parameters:
        -----------
        X : array-like
            Features to predict on
            
        Returns:
        --------
        y_pred : array-like
            Binary predictions (0 or 1)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Make probability predictions using the trained model
        
        Parameters:
        -----------
        X : array-like
            Features to predict on
            
        Returns:
        --------
        y_pred_proba : array-like
            Probability predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data and return metrics
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test target values
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': auc
        }
        
        return metrics
    
    def plot_feature_importances(self, top_n=20):
        """
        Plot the top feature importances
        
        Parameters:
        -----------
        top_n : int, default=20
            Number of top features to display
        """
        if self.feature_importances is None:
            raise ValueError("Model has not been trained yet")
        
        top_features = self.feature_importances.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_confusion_matrix(self, X_test, y_test):
        """
        Plot the confusion matrix
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test target values
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Fraud', 'Fraud'],
                    yticklabels=['Not Fraud', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_roc_curve(self, X_test, y_test):
        """
        Plot the ROC curve
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test target values
        """
        from sklearn.metrics import roc_curve
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        
        return plt.gcf()
    
    def classification_report_df(self, X_test, y_test):
        """
        Generate a DataFrame of the classification report
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test target values
            
        Returns:
        --------
        report_df : DataFrame
            Classification report as a DataFrame
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        return report_df
    
    def tune_hyperparameters(self, X_train, y_train, n_iter=10, cv=3):
        """
        Optimize model hyperparameters using RandomizedSearchCV.
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train : Series
            Training target values
        n_iter : int, default=10
            Number of parameter settings sampled
        cv : int, default=3
            Number of cross-validation folds
            
        Returns:
        --------
        tuple
            (best_params, best_score)
        """
        X_train_scaled = self.scaler.transform(X_train)
        
        # Parameter distribution for RandomForest
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
        
        # Randomized search
        random_search = RandomizedSearchCV(
            self.model, param_distributions=param_dist,
            n_iter=n_iter, cv=cv, scoring='roc_auc',
            n_jobs=-1, random_state=42
        )
        
        random_search.fit(X_train_scaled, y_train)
        
        print(f"Best hyperparameters: {random_search.best_params_}")
        print(f"Best ROC AUC score: {random_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = random_search.best_estimator_
        
        return random_search.best_params_, random_search.best_score_
    
    def evaluate_with_cross_validation(self, X, y, cv=5):
        """
        Evaluate model using stratified cross-validation
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series
            Target values
        cv : int, default=5
            Number of cross-validation folds
            
        Returns:
        --------
        scores : array
            Cross-validation scores
        """
        X_scaled = self.scaler.transform(X)
        
        # Stratified cross-validation
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Calculate various metrics
        roc_auc_scores = cross_val_score(self.model, X_scaled, y, cv=skf, scoring='roc_auc')
        precision_scores = cross_val_score(self.model, X_scaled, y, cv=skf, scoring='precision')
        recall_scores = cross_val_score(self.model, X_scaled, y, cv=skf, scoring='recall')
        f1_scores = cross_val_score(self.model, X_scaled, y, cv=skf, scoring='f1')
        
        print(f"{cv}-fold cross-validation results:")
        print(f"ROC AUC: {roc_auc_scores.mean():.4f} (±{roc_auc_scores.std():.4f})")
        print(f"Precision: {precision_scores.mean():.4f} (±{precision_scores.std():.4f})")
        print(f"Recall: {recall_scores.mean():.4f} (±{recall_scores.std():.4f})")
        print(f"F1 Score: {f1_scores.mean():.4f} (±{f1_scores.std():.4f})")
        
        return {
            'roc_auc': roc_auc_scores,
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores
        }
    
    def save_model(self, model_path="fraud_model.joblib", scaler_path="scaler.joblib"):
        """
        Save the trained model and scaler to disk
        
        Parameters:
        -----------
        model_path : str, default="fraud_model.joblib"
            Path to save model
        scaler_path : str, default="scaler.joblib"
            Path to save scaler
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path="fraud_model.joblib", scaler_path="scaler.joblib"):
        """
        Load a trained model and scaler from disk
        
        Parameters:
        -----------
        model_path : str, default="fraud_model.joblib"
            Path to load model from
        scaler_path : str, default="scaler.joblib"
            Path to load scaler from
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Model and scaler loaded successfully")
            return True
        except FileNotFoundError:
            print(f"Error: Model or scaler file not found")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def select_features(self, X_train, y_train, threshold='median'):
        """
        Select important features using a RandomForest feature selector
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train : Series
            Training target values
        threshold : str or float, default='median'
            Threshold for feature selection
            
        Returns:
        --------
        tuple
            (feature_selector, X_train_selected)
        """
        X_train_scaled = self.scaler.transform(X_train)
        
        # Create feature selector
        feature_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            threshold=threshold
        )
        
        # Fit and transform
        feature_selector.fit(X_train_scaled, y_train)
        X_train_selected = feature_selector.transform(X_train_scaled)
        
        # Get selected feature indices
        selected_features_mask = feature_selector.get_support()
        selected_features = X_train.columns[selected_features_mask].tolist()
        
        # Summary
        n_features_before = X_train_scaled.shape[1]
        n_features_after = X_train_selected.shape[1]
        
        print(f"Feature selection: {n_features_before} -> {n_features_after} features")
        print(f"Selected features: {selected_features}")
        
        # Store for future reference
        self.feature_selector = feature_selector
        self.selected_features = selected_features
        
        return feature_selector, X_train_selected
