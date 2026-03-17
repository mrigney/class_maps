"""Random Forest classifier wrapper for landcover classification."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from class_maps.config import RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF


class LandcoverClassifier:
    """Semi-supervised landcover classifier using Random Forest.

    Wraps scikit-learn's RandomForestClassifier with a labeling interface
    for interactive use.
    """

    def __init__(self):
        self._model = None
        self._scaler = StandardScaler()
        self._is_trained = False
        self._oob_score = None
        self._feature_importances = None

    @property
    def is_trained(self):
        return self._is_trained

    @property
    def oob_score(self):
        """Out-of-bag accuracy score, available after training."""
        return self._oob_score

    @property
    def feature_importances(self):
        """Feature importance array, available after training."""
        return self._feature_importances

    def train(self, feature_matrix, segment_ids, labeled_segments):
        """Train the Random Forest on labeled superpixels.

        Parameters
        ----------
        feature_matrix : np.ndarray
            (n_segments, n_features) feature matrix for ALL segments.
        segment_ids : list of int
            Segment IDs corresponding to rows of feature_matrix.
        labeled_segments : dict
            {segment_id: class_id} for user-labeled segments.

        Returns
        -------
        float
            OOB accuracy score.

        Raises
        ------
        ValueError
            If fewer than 2 classes are labeled.
        """
        # Build training arrays
        seg_to_row = {sid: i for i, sid in enumerate(segment_ids)}

        train_indices = []
        train_labels = []
        for seg_id, class_id in labeled_segments.items():
            if seg_id in seg_to_row:
                train_indices.append(seg_to_row[seg_id])
                train_labels.append(class_id)

        if len(train_indices) == 0:
            raise ValueError("No labeled segments found.")

        unique_classes = set(train_labels)
        if len(unique_classes) < 2:
            raise ValueError(
                f"At least 2 classes required for classification, "
                f"but only {len(unique_classes)} found."
            )

        X_train = feature_matrix[train_indices]
        y_train = np.array(train_labels)

        # Scale features
        self._scaler.fit(feature_matrix)  # Fit on all data for consistent scaling
        X_train_scaled = self._scaler.transform(X_train)

        # Train Random Forest
        self._model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            oob_score=True,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X_train_scaled, y_train)

        self._is_trained = True
        self._oob_score = self._model.oob_score_
        self._feature_importances = self._model.feature_importances_

        return self._oob_score

    def predict(self, feature_matrix):
        """Predict class IDs for all segments.

        Parameters
        ----------
        feature_matrix : np.ndarray
            (n_segments, n_features) feature matrix.

        Returns
        -------
        np.ndarray
            (n_segments,) predicted class IDs.
        """
        if not self._is_trained:
            raise RuntimeError("Classifier has not been trained yet.")

        X_scaled = self._scaler.transform(feature_matrix)
        return self._model.predict(X_scaled)

    def predict_proba(self, feature_matrix):
        """Predict class probabilities for all segments.

        Parameters
        ----------
        feature_matrix : np.ndarray
            (n_segments, n_features) feature matrix.

        Returns
        -------
        np.ndarray
            (n_segments, n_classes) probability matrix.
        list
            Class labels corresponding to columns.
        """
        if not self._is_trained:
            raise RuntimeError("Classifier has not been trained yet.")

        X_scaled = self._scaler.transform(feature_matrix)
        proba = self._model.predict_proba(X_scaled)
        classes = list(self._model.classes_)
        return proba, classes

    def get_model_and_scaler(self):
        """Return the trained model and scaler for serialization."""
        return self._model, self._scaler

    def set_model_and_scaler(self, model, scaler):
        """Load a previously trained model and scaler."""
        self._model = model
        self._scaler = scaler
        self._is_trained = True
        if hasattr(model, "oob_score_"):
            self._oob_score = model.oob_score_
        if hasattr(model, "feature_importances_"):
            self._feature_importances = model.feature_importances_
