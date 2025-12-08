"""
ML Success Model Agent for Dispatch Optimization (with Delta Table Persistence)
================================================================================

This agent trains and exposes models that predict dispatch success metrics
for (dispatch, technician) pairs. Supports persistent model storage in Databricks Delta tables.
"""

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import pickle
import base64
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, falling back to sklearn GradientBoostingClassifier")


class SuccessModelAgent:
    """
    ML agent for predicting dispatch success probabilities.
    
    Trains models on historical data and provides scoring API for candidate pairs.
    Supports saving and loading trained models from Delta tables (persistent).
    """
    
    def __init__(self, spark_session: SparkSession, config: dict = None):
        """
        Initialize the Success Model Agent.
        
        Args:
            spark_session: Active Spark session
            config: Optional configuration dict
        """
        self.spark = spark_session
        self.config = config or {}
        
        # Configuration defaults
        self.test_size = self.config.get('test_size', 0.2)
        self.val_size = self.config.get('val_size', 0.1)
        self.model_type = self.config.get('model_type', 'xgboost')
        self.random_state = self.config.get('random_state', 42)
        self.calibrate = self.config.get('calibrate', True)
        
        # Data location configuration
        self.catalog = self.config.get('catalog', 'hackathon')
        self.schema = self.config.get('schema', 'hackathon_fiber_vault')
        
        # Model and data storage
        self.model_productive = None
        self.model_ftf = None
        self.feature_columns = None
        self.label_encoders = {}
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.feature_stats = {}
        
        print(f"SuccessModelAgent initialized")
        print(f"  Catalog: {self.catalog}")
        print(f"  Schema: {self.schema}")
    
    def load_training_data(self) -> DataFrame:
        """Load and join historical dispatch data with technician information."""
        print("\nLoading training data from Databricks tables...")
        
        history_table = f"{self.catalog}.{self.schema}.dispatch_history_hackathon"
        technicians_table = f"{self.catalog}.{self.schema}.technicians_hackathon"
        
        print(f"  History table: {history_table}")
        print(f"  Technicians table: {technicians_table}")
        
        history_df = self.spark.table(history_table)
        print(f"  Loaded {history_df.count():,} historical dispatches")
        
        technicians_df = self.spark.table(technicians_table)
        print(f"  Loaded {technicians_df.count():,} technicians")
        
        tech_cols_renamed = technicians_df.select(
            F.col("Technician_id").alias("tech_id"),
            F.col("Name").alias("technician_name"),
            F.col("Primary_skill").alias("technician_skill"),
            F.col("City").alias("technician_city"),
            F.col("State").alias("technician_state"),
            F.col("Latitude").alias("technician_latitude"),
            F.col("Longitude").alias("technician_longitude"),
            F.col("Workload_capacity").alias("technician_workload_capacity"),
            F.col("Current_assignments").alias("technician_current_assignments")
        )
        
        joined_df = history_df.join(
            tech_cols_renamed,
            history_df.Assigned_technician_id == tech_cols_renamed.tech_id,
            "inner"
        )
        
        training_df = joined_df.select(
            F.col("Dispatch_id").alias("dispatch_id"),
            F.col("Ticket_type").alias("ticket_type"),
            F.col("Order_type").alias("order_type"),
            F.col("Priority").alias("priority"),
            F.col("Required_skill").alias("required_skill"),
            F.col("City").alias("dispatch_city"),
            F.col("State").alias("dispatch_state"),
            F.col("Customer_latitude").alias("customer_latitude"),
            F.col("Customer_longitude").alias("customer_longitude"),
            F.col("Appointment_start_time").alias("appointment_start_time"),
            F.col("Duration_min").alias("duration_min"),
            F.col("Assigned_technician_id").alias("technician_id"),
            "technician_name",
            "technician_skill",
            "technician_city",
            "technician_state",
            "technician_latitude",
            "technician_longitude",
            "technician_workload_capacity",
            "technician_current_assignments",
            F.col("Productive_dispatch").alias("productive_dispatch"),
            F.col("First_time_fix").alias("first_time_fix"),
            F.col("Distance_km").alias("actual_distance_km"),
            F.col("Actual_duration_min").alias("actual_duration_min")
        )
        
        training_df = training_df.filter(
            F.col("productive_dispatch").isNotNull() & 
            F.col("first_time_fix").isNotNull()
        )
        
        self.train_data = training_df
        print(f"  Training data prepared: {training_df.count():,} records")
        
        return training_df
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points in kilometers."""
        R = 6371.0
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        a = np.sin(delta_lat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    def build_features(self, df, is_training=True):
        """Build feature matrix from raw dispatch-technician pairs."""
        print("Building features...")
        
        if isinstance(df, DataFrame):
            pdf = df.toPandas()
        else:
            pdf = df.copy()
        
        pdf['distance_km'] = self._haversine_distance(
            pdf['technician_latitude'], pdf['technician_longitude'],
            pdf['customer_latitude'], pdf['customer_longitude']
        )
        
        pdf['skill_match'] = (pdf['required_skill'] == pdf['technician_skill']).astype(int)
        pdf['city_match'] = (pdf['dispatch_city'] == pdf['technician_city']).astype(int)
        pdf['state_match'] = (pdf['dispatch_state'] == pdf['technician_state']).astype(int)
        pdf['workload_utilization'] = pdf['technician_current_assignments'] / pdf['technician_workload_capacity']
        pdf['appointment_hour'] = pd.to_datetime(pdf['appointment_start_time']).dt.hour
        pdf['appointment_day_of_week'] = pd.to_datetime(pdf['appointment_start_time']).dt.dayofweek
        
        pdf['duration_category'] = pd.cut(
            pdf['duration_min'], bins=[0, 60, 90, 120, 999],
            labels=['short', 'medium', 'long', 'extra_long']
        ).astype(str)
        
        pdf['distance_category'] = pd.cut(
            pdf['distance_km'], bins=[0, 10, 25, 50, 100, 999],
            labels=['very_close', 'close', 'medium', 'far', 'very_far']
        ).astype(str)
        
        categorical_cols = [
            'ticket_type', 'order_type', 'priority', 'required_skill',
            'technician_skill', 'dispatch_city', 'technician_city',
            'duration_category', 'distance_category'
        ]
        
        for col in categorical_cols:
            if col in pdf.columns:
                if is_training:
                    le = LabelEncoder()
                    pdf[f'{col}_encoded'] = le.fit_transform(pdf[col].fillna('MISSING'))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le is not None:
                        pdf[f'{col}_temp'] = pdf[col].fillna('MISSING')
                        pdf[f'{col}_encoded'] = pdf[f'{col}_temp'].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                        pdf.drop(columns=[f'{col}_temp'], inplace=True)
                    else:
                        pdf[f'{col}_encoded'] = -1
        
        feature_cols = [
            'distance_km', 'skill_match', 'city_match', 'state_match',
            'workload_utilization', 'appointment_hour', 'appointment_day_of_week',
            'duration_min', 'technician_workload_capacity', 'technician_current_assignments',
            'ticket_type_encoded', 'order_type_encoded', 'priority_encoded',
            'required_skill_encoded', 'technician_skill_encoded',
            'dispatch_city_encoded', 'technician_city_encoded',
            'duration_category_encoded', 'distance_category_encoded'
        ]
        
        if is_training:
            self.feature_columns = feature_cols
            X = pdf[feature_cols].fillna(-1)
            self.feature_stats = {
                'mean': X.mean().to_dict(),
                'std': X.std().to_dict(),
                'min': X.min().to_dict(),
                'max': X.max().to_dict()
            }
        
        X = pdf[self.feature_columns].fillna(-1)
        
        y_productive = None
        y_ftf = None
        if is_training and 'productive_dispatch' in pdf.columns:
            y_productive = pdf['productive_dispatch'].values
            y_ftf = pdf['first_time_fix'].values
        
        metadata_cols = ['dispatch_id', 'technician_id']
        metadata_cols.extend([c for c in ['priority', 'dispatch_city', 'required_skill'] if c in pdf.columns])
        metadata_df = pdf[metadata_cols]
        
        print(f"  Features built: {X.shape[0]:,} rows, {X.shape[1]} features")
        
        return X, y_productive, y_ftf, metadata_df
    
    def train_models(self):
        """Train both classification models."""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        if self.train_data is None:
            raise ValueError("No training data loaded. Call load_training_data() first.")
        
        X, y_productive, y_ftf, metadata = self.build_features(self.train_data, is_training=True)
        
        X_temp, X_test, y_prod_temp, y_prod_test, y_ftf_temp, y_ftf_test, meta_temp, meta_test = train_test_split(
            X, y_productive, y_ftf, metadata,
            test_size=self.test_size, random_state=self.random_state, stratify=y_productive
        )
        
        val_proportion = self.val_size / (1 - self.test_size)
        X_train, X_val, y_prod_train, y_prod_val, y_ftf_train, y_ftf_val, meta_train, meta_val = train_test_split(
            X_temp, y_prod_temp, y_ftf_temp, meta_temp,
            test_size=val_proportion, random_state=self.random_state, stratify=y_prod_temp
        )
        
        print(f"\nData split:")
        print(f"  Training:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        self.val_data = (X_val, y_prod_val, y_ftf_val, meta_val)
        self.test_data = (X_test, y_prod_test, y_ftf_test, meta_test)
        
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            base_model_prod = XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, eval_metric='logloss', use_label_encoder=False
            )
            base_model_ftf = XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, eval_metric='logloss', use_label_encoder=False
            )
            print("\nUsing XGBoost models")
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            base_model_prod = GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=self.random_state
            )
            base_model_ftf = GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=self.random_state
            )
            print("\nUsing sklearn GradientBoosting models")
        
        print("\n" + "-"*60)
        print("Training Model A: Productive Dispatch Classifier")
        print("-"*60)
        
        if self.calibrate:
            print("Training with probability calibration...")
            self.model_productive = CalibratedClassifierCV(base_model_prod, method='sigmoid', cv=3)
        else:
            self.model_productive = base_model_prod
        
        self.model_productive.fit(X_train, y_prod_train)
        y_prod_pred = self.model_productive.predict_proba(X_val)[:, 1]
        val_auc_prod = roc_auc_score(y_prod_val, y_prod_pred)
        print(f"Validation ROC-AUC (Productive): {val_auc_prod:.4f}")
        
        print("\n" + "-"*60)
        print("Training Model B: First-Time-Fix Classifier")
        print("-"*60)
        
        if self.calibrate:
            print("Training with probability calibration...")
            self.model_ftf = CalibratedClassifierCV(base_model_ftf, method='sigmoid', cv=3)
        else:
            self.model_ftf = base_model_ftf
        
        self.model_ftf.fit(X_train, y_ftf_train)
        y_ftf_pred = self.model_ftf.predict_proba(X_val)[:, 1]
        val_auc_ftf = roc_auc_score(y_ftf_val, y_ftf_pred)
        print(f"Validation ROC-AUC (First-Time-Fix): {val_auc_ftf:.4f}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        return {
            'model_productive_val_auc': val_auc_prod,
            'model_ftf_val_auc': val_auc_ftf,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'n_features': X_train.shape[1]
        }
    
    def evaluate_models(self) -> dict:
        """Evaluate models on test set."""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        if self.model_productive is None or self.model_ftf is None:
            raise ValueError("Models not trained. Call train_models() first.")
        
        if self.test_data is None:
            raise ValueError("No test data available.")
        
        X_test, y_prod_test, y_ftf_test, meta_test = self.test_data
        
        y_prod_pred_proba = self.model_productive.predict_proba(X_test)[:, 1]
        y_prod_pred = (y_prod_pred_proba >= 0.5).astype(int)
        
        y_ftf_pred_proba = self.model_ftf.predict_proba(X_test)[:, 1]
        y_ftf_pred = (y_ftf_pred_proba >= 0.5).astype(int)
        
        metrics = {}
        metrics['productive_roc_auc'] = roc_auc_score(y_prod_test, y_prod_pred_proba)
        metrics['productive_pr_auc'] = average_precision_score(y_prod_test, y_prod_pred_proba)
        metrics['productive_accuracy'] = accuracy_score(y_prod_test, y_prod_pred)
        
        metrics['ftf_roc_auc'] = roc_auc_score(y_ftf_test, y_ftf_pred_proba)
        metrics['ftf_pr_auc'] = average_precision_score(y_ftf_test, y_ftf_pred_proba)
        metrics['ftf_accuracy'] = accuracy_score(y_ftf_test, y_ftf_pred)
        
        print("\n" + "-"*60)
        print("OVERALL TEST SET PERFORMANCE")
        print("-"*60)
        print("\nProductive Dispatch Model:")
        print(f"  ROC-AUC:  {metrics['productive_roc_auc']:.4f}")
        print(f"  PR-AUC:   {metrics['productive_pr_auc']:.4f}")
        print(f"  Accuracy: {metrics['productive_accuracy']:.4f}")
        
        print("\nFirst-Time-Fix Model:")
        print(f"  ROC-AUC:  {metrics['ftf_roc_auc']:.4f}")
        print(f"  PR-AUC:   {metrics['ftf_pr_auc']:.4f}")
        print(f"  Accuracy: {metrics['ftf_accuracy']:.4f}")
        
        print("\n" + "="*60)
        
        return metrics
    
    def score_candidates(self, candidate_df) -> pd.DataFrame:
        """Score candidate pairs with success probabilities."""
        if self.model_productive is None or self.model_ftf is None:
            raise ValueError("Models not trained. Call train_models() first.")
        
        record_count = candidate_df.count() if isinstance(candidate_df, DataFrame) else len(candidate_df)
        print(f"\nScoring {record_count:,} candidate pairs...")
        
        X, _, _, metadata = self.build_features(candidate_df, is_training=False)
        
        p_productive = self.model_productive.predict_proba(X)[:, 1]
        p_ftf = self.model_ftf.predict_proba(X)[:, 1]
        
        output_df = pd.DataFrame({
            'dispatch_id': metadata['dispatch_id'].values,
            'technician_id': metadata['technician_id'].values,
            'p_productive': p_productive,
            'p_ftf': p_ftf
        })
        
        print(f"  Scoring complete!")
        print(f"  Average p_productive: {p_productive.mean():.3f}")
        print(f"  Average p_ftf: {p_ftf.mean():.3f}")
        
        return output_df
    
    def save_models_to_table(self, table_name: str = None):
        """
        Save models to a Delta table (PERSISTENT across cluster restarts).
        
        Args:
            table_name: Full table name. If None, uses default.
        
        Returns:
            tuple: (table_name, model_id)
        """
        if self.model_productive is None or self.model_ftf is None:
            raise ValueError("No models to save. Train models first.")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if table_name is None:
            table_name = f"{self.catalog}.{self.schema}.ml_models"
        
        print(f"\nSaving models to Delta table: {table_name}")
        
        # Serialize models to bytes and encode as base64 strings
        model_prod_bytes = base64.b64encode(pickle.dumps(self.model_productive)).decode('utf-8')
        model_ftf_bytes = base64.b64encode(pickle.dumps(self.model_ftf)).decode('utf-8')
        encoders_bytes = base64.b64encode(pickle.dumps(self.label_encoders)).decode('utf-8')
        
        metadata = {
            'feature_columns': self.feature_columns,
            'feature_stats': self.feature_stats,
            'config': self.config,
            'catalog': self.catalog,
            'schema': self.schema
        }
        metadata_bytes = base64.b64encode(pickle.dumps(metadata)).decode('utf-8')
        
        # Create DataFrame with model data
        model_data = [{
            'model_id': f'success_agent_{timestamp}',
            'model_version': timestamp,
            'created_timestamp': datetime.now(),
            'model_productive': model_prod_bytes,
            'model_ftf': model_ftf_bytes,
            'label_encoders': encoders_bytes,
            'metadata': metadata_bytes,
            'model_type': 'SuccessModelAgent',
            'config': str(self.config)
        }]
        
        models_df = self.spark.createDataFrame(model_data)
        
        # Save to Delta table (append mode to keep history)
        models_df.write.mode("append").saveAsTable(table_name)
        
        print(f"  ✓ Saved model_productive")
        print(f"  ✓ Saved model_ftf")
        print(f"  ✓ Saved label_encoders")
        print(f"  ✓ Saved metadata")
        
        print(f"\n✓ Models saved to persistent Delta table!")
        print(f"  Table: {table_name}")
        print(f"  Model ID: success_agent_{timestamp}")
        print(f"  Storage: PERSISTENT (survives cluster restarts)")
        
        return table_name, f'success_agent_{timestamp}'
    
    def load_models_from_table(self, table_name: str = None, model_id: str = None):
        """
        Load models from a Delta table.
        
        Args:
            table_name: Full table name
            model_id: Specific model ID to load. If None, loads the latest.
        """
        if table_name is None:
            table_name = f"{self.catalog}.{self.schema}.ml_models"
        
        print(f"\nLoading models from Delta table: {table_name}")
        
        # Read from table
        models_df = self.spark.table(table_name)
        
        if model_id:
            model_row = models_df.filter(F.col("model_id") == model_id).first()
        else:
            model_row = models_df.orderBy(F.desc("created_timestamp")).first()
        
        if model_row is None:
            raise ValueError(f"No models found in table {table_name}")
        
        print(f"  Loading model: {model_row['model_id']}")
        print(f"  Created: {model_row['created_timestamp']}")
        
        # Deserialize models
        self.model_productive = pickle.loads(base64.b64decode(model_row['model_productive']))
        print("  ✓ Loaded model_productive")
        
        self.model_ftf = pickle.loads(base64.b64decode(model_row['model_ftf']))
        print("  ✓ Loaded model_ftf")
        
        self.label_encoders = pickle.loads(base64.b64decode(model_row['label_encoders']))
        print("  ✓ Loaded label_encoders")
        
        metadata = pickle.loads(base64.b64decode(model_row['metadata']))
        self.feature_columns = metadata['feature_columns']
        self.feature_stats = metadata['feature_stats']
        print("  ✓ Loaded metadata")
        
        print(f"\n✓ Models loaded successfully from Delta table!")
        
        return model_row['model_id']
    
    def save_models(self, path: str = None):
        """
        Save models to /tmp (temporary storage).
        
        Note: Use save_models_to_table() for persistent storage.
        """
        if self.model_productive is None or self.model_ftf is None:
            raise ValueError("No models to save. Train models first.")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if path is None:
            path = f"/tmp/models/success_agent_{timestamp}"
        
        os.makedirs(path, exist_ok=True)
        
        print(f"\nSaving models to: {path}")
        print(f"⚠️  Note: /tmp storage is TEMPORARY (lost after cluster restart)")
        
        with open(f"{path}/model_productive.pkl", 'wb') as f:
            pickle.dump(self.model_productive, f)
        print("  ✓ Saved model_productive.pkl")
        
        with open(f"{path}/model_ftf.pkl", 'wb') as f:
            pickle.dump(self.model_ftf, f)
        print("  ✓ Saved model_ftf.pkl")
        
        with open(f"{path}/label_encoders.pkl", 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print("  ✓ Saved label_encoders.pkl")
        
        metadata = {
            'feature_columns': self.feature_columns,
            'feature_stats': self.feature_stats,
            'config': self.config,
            'catalog': self.catalog,
            'schema': self.schema,
            'saved_timestamp': timestamp
        }
        with open(f"{path}/metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        print("  ✓ Saved metadata.pkl")
        
        print(f"\n✓ All models saved!")
        print(f"  Location: {path}")
        
        return path
    
    def load_models(self, path: str):
        """Load models from /tmp."""
        print(f"\nLoading models from: {path}")
        
        with open(f"{path}/model_productive.pkl", 'rb') as f:
            self.model_productive = pickle.load(f)
        print("  ✓ Loaded model_productive.pkl")
        
        with open(f"{path}/model_ftf.pkl", 'rb') as f:
            self.model_ftf = pickle.load(f)
        print("  ✓ Loaded model_ftf.pkl")
        
        with open(f"{path}/label_encoders.pkl", 'rb') as f:
            self.label_encoders = pickle.load(f)
        print("  ✓ Loaded label_encoders.pkl")
        
        with open(f"{path}/metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        self.feature_columns = metadata['feature_columns']
        self.feature_stats = metadata['feature_stats']
        print("  ✓ Loaded metadata.pkl")
        
        print(f"\n✓ All models loaded!")