"""Unit tests for the feature engineering module.

Tests feature extractors, transformers, pipeline composition,
and feature store operations.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.feature_engineering.extractors.horse_features import HorseFeatureExtractor
from src.feature_engineering.extractors.jockey_features import JockeyFeatureExtractor
from src.feature_engineering.extractors.race_features import RaceFeatureExtractor
from src.feature_engineering.extractors.running_style_features import (
    RunningStyleFeatureExtractor,
)
from src.feature_engineering.pipeline import FeaturePipeline
from src.feature_engineering.transformers.encoders import CategoryEncoder
from src.feature_engineering.transformers.scalers import FeatureScaler

# ---------------------------------------------------------------------------
# RaceFeatureExtractor tests
# ---------------------------------------------------------------------------


class TestRaceFeatureExtractor:
    """Tests for RaceFeatureExtractor."""

    def test_extract_all_features(self):
        df = pl.DataFrame(
            {
                "race_id": ["R001"],
                "distance": [1600],
                "track_condition": ["良"],
                "weather": ["晴"],
                "track_type": ["芝"],
                "num_entries": [16],
                "grade": ["G1"],
            }
        )

        extractor = RaceFeatureExtractor()
        result = extractor.extract(df)

        assert "feat_distance" in result.columns
        assert "feat_track_condition" in result.columns
        assert "feat_weather" in result.columns
        assert "feat_track_type" in result.columns
        assert "feat_num_entries" in result.columns
        assert "feat_is_grade_race" in result.columns

        assert result["feat_distance"][0] == 1600
        assert result["feat_track_condition"][0] == 0  # 良 -> 0
        assert result["feat_weather"][0] == 0  # 晴 -> 0
        assert result["feat_track_type"][0] == 0  # 芝 -> 0
        assert result["feat_num_entries"][0] == 16
        assert result["feat_is_grade_race"][0] == 1

    def test_extract_with_missing_columns(self):
        df = pl.DataFrame({"race_id": ["R001"]})

        extractor = RaceFeatureExtractor()
        result = extractor.extract(df)

        # Should produce default values for missing columns
        assert result["feat_distance"][0] == 0
        assert result["feat_track_condition"][0] == -1
        assert result["feat_weather"][0] == -1

    def test_extract_unknown_category(self):
        df = pl.DataFrame(
            {
                "race_id": ["R001"],
                "distance": [2000],
                "track_condition": ["unknown"],
                "weather": ["unknown"],
                "track_type": ["unknown"],
                "num_entries": [12],
                "grade": [""],
            }
        )

        extractor = RaceFeatureExtractor()
        result = extractor.extract(df)

        # Unknown values should map to -1
        assert result["feat_track_condition"][0] == -1
        assert result["feat_weather"][0] == -1
        assert result["feat_track_type"][0] == -1
        assert result["feat_is_grade_race"][0] == 0

    def test_feature_names(self):
        extractor = RaceFeatureExtractor()
        names = extractor.feature_names
        assert len(names) == 6
        assert all(n.startswith("feat_") for n in names)


# ---------------------------------------------------------------------------
# HorseFeatureExtractor tests
# ---------------------------------------------------------------------------


class TestHorseFeatureExtractor:
    """Tests for HorseFeatureExtractor."""

    def test_extract_from_pre_aggregated(self):
        df = pl.DataFrame(
            {
                "horse_id": ["H001"],
                "avg_finish": [2.5],
                "win_rate": [0.3],
                "top3_rate": [0.6],
                "days_since_last_race": [14],
                "horse_age": [4],
                "weight": [480.0],
                "num_past_races": [10],
            }
        )

        extractor = HorseFeatureExtractor()
        result = extractor.extract(df)

        assert result["feat_avg_finish"][0] == pytest.approx(2.5)
        assert result["feat_win_rate"][0] == pytest.approx(0.3)
        assert result["feat_top3_rate"][0] == pytest.approx(0.6)
        assert result["feat_days_since_last_race"][0] == 14
        assert result["feat_horse_age"][0] == 4
        assert result["feat_horse_weight"][0] == pytest.approx(480.0)
        assert result["feat_num_past_races"][0] == 10

    def test_extract_from_single_result(self):
        df = pl.DataFrame(
            {
                "horse_id": ["H001"],
                "finish_position": [1],
            }
        )

        extractor = HorseFeatureExtractor()
        result = extractor.extract(df)

        assert result["feat_avg_finish"][0] == pytest.approx(1.0)
        assert result["feat_win_rate"][0] == pytest.approx(1.0)
        assert result["feat_top3_rate"][0] == pytest.approx(1.0)

    def test_aggregate_history(self):
        history = pl.DataFrame(
            {
                "horse_id": ["H001"] * 5 + ["H002"] * 3,
                "race_date": [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-04-01",
                    "2024-05-01",
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                ],
                "finish_position": [1, 2, 3, 1, 4, 5, 1, 2],
                "weight": [480, 482, 481, 483, 480, 460, 462, 461],
            }
        )

        agg = HorseFeatureExtractor.aggregate_history(history, n_past=5)

        assert len(agg) == 2
        assert "avg_finish" in agg.columns
        assert "win_rate" in agg.columns
        assert "top3_rate" in agg.columns
        assert "num_past_races" in agg.columns

    def test_extract_new_features(self):
        df = pl.DataFrame(
            {
                "horse_id": ["H001"],
                "finish_position": [1],
                "win_odds": [5.5],
                "win_favorite": [2],
                "bracket_number": [3],
                "post_position": [5],
                "carried_weight": [55.0],
                "sex": ["牡"],
                "horse_weight_change": [4.0],
            }
        )

        extractor = HorseFeatureExtractor()
        result = extractor.extract(df)

        assert result["feat_win_odds"][0] == pytest.approx(5.5)
        assert result["feat_win_favorite"][0] == 2
        assert result["feat_bracket_number"][0] == 3
        assert result["feat_post_position"][0] == 5
        assert result["feat_carried_weight"][0] == pytest.approx(55.0)
        assert result["feat_sex"][0] == 0  # 牡 -> 0
        assert result["feat_horse_weight_change"][0] == pytest.approx(4.0)

    def test_sex_encoding(self):
        df = pl.DataFrame(
            {
                "horse_id": ["H001", "H002", "H003", "H004"],
                "sex": ["牡", "牝", "騸", "unknown"],
            }
        )

        extractor = HorseFeatureExtractor()
        result = extractor.extract(df)

        assert result["feat_sex"].to_list() == [0, 1, 2, -1]

    def test_new_features_missing_columns(self):
        df = pl.DataFrame({"horse_id": ["H001"]})

        extractor = HorseFeatureExtractor()
        result = extractor.extract(df)

        assert result["feat_win_odds"][0] is None
        assert result["feat_win_favorite"][0] is None
        assert result["feat_bracket_number"][0] is None
        assert result["feat_post_position"][0] is None
        assert result["feat_carried_weight"][0] is None
        assert result["feat_sex"][0] is None
        assert result["feat_horse_weight_change"][0] is None

    def test_feature_names(self):
        extractor = HorseFeatureExtractor()
        names = extractor.feature_names
        assert len(names) == 14
        assert all(n.startswith("feat_") for n in names)


# ---------------------------------------------------------------------------
# RunningStyleFeatureExtractor tests
# ---------------------------------------------------------------------------


class TestRunningStyleFeatureExtractor:
    """Tests for RunningStyleFeatureExtractor."""

    def test_extract_from_pre_aggregated(self):
        df = pl.DataFrame(
            {
                "horse_id": ["H001"],
                "avg_corner_pos_4": [3.2],
                "avg_last_3f_time": [35.5],
            }
        )

        extractor = RunningStyleFeatureExtractor()
        result = extractor.extract(df)

        assert result["feat_avg_corner_pos_4"][0] == pytest.approx(3.2)
        assert result["feat_avg_last_3f_time"][0] == pytest.approx(35.5)

    def test_extract_with_missing_columns(self):
        df = pl.DataFrame({"horse_id": ["H001"]})

        extractor = RunningStyleFeatureExtractor()
        result = extractor.extract(df)

        assert result["feat_avg_corner_pos_4"][0] is None
        assert result["feat_avg_last_3f_time"][0] is None

    def test_feature_names(self):
        extractor = RunningStyleFeatureExtractor()
        names = extractor.feature_names
        assert len(names) == 2
        assert all(n.startswith("feat_") for n in names)


# ---------------------------------------------------------------------------
# JockeyFeatureExtractor tests
# ---------------------------------------------------------------------------


class TestJockeyFeatureExtractor:
    """Tests for JockeyFeatureExtractor."""

    def test_extract_from_pre_aggregated(self):
        df = pl.DataFrame(
            {
                "jockey_id": ["J001"],
                "jockey_win_rate": [0.15],
                "jockey_top3_rate": [0.35],
                "jockey_course_win_rate": [0.20],
                "jockey_experience": [500],
            }
        )

        extractor = JockeyFeatureExtractor()
        result = extractor.extract(df)

        assert result["feat_jockey_win_rate"][0] == pytest.approx(0.15)
        assert result["feat_jockey_top3_rate"][0] == pytest.approx(0.35)
        assert result["feat_jockey_course_win_rate"][0] == pytest.approx(0.20)
        assert result["feat_jockey_experience"][0] == 500

    def test_aggregate_history(self):
        history = pl.DataFrame(
            {
                "jockey_id": ["J001"] * 10,
                "race_id": [f"R{i:03d}" for i in range(10)],
                "course": ["Tokyo"] * 5 + ["Kyoto"] * 5,
                "finish_position": [1, 2, 3, 1, 5, 1, 2, 4, 1, 3],
            }
        )

        agg = JockeyFeatureExtractor.aggregate_history(history, course="Tokyo")

        assert len(agg) == 1
        assert "jockey_win_rate" in agg.columns
        assert "jockey_course_win_rate" in agg.columns

    def test_aggregate_history_without_course(self):
        history = pl.DataFrame(
            {
                "jockey_id": ["J001"] * 3,
                "race_id": ["R001", "R002", "R003"],
                "course": ["Tokyo", "Kyoto", "Tokyo"],
                "finish_position": [1, 2, 3],
            }
        )

        agg = JockeyFeatureExtractor.aggregate_history(history)
        assert "jockey_course_win_rate" in agg.columns

    def test_feature_names(self):
        extractor = JockeyFeatureExtractor()
        names = extractor.feature_names
        assert len(names) == 4
        assert all(n.startswith("feat_") for n in names)


# ---------------------------------------------------------------------------
# CategoryEncoder tests
# ---------------------------------------------------------------------------


class TestCategoryEncoder:
    """Tests for CategoryEncoder."""

    def test_label_encoding(self):
        df = pl.DataFrame(
            {
                "course": ["Tokyo", "Kyoto", "Tokyo", "Hanshin"],
            }
        )

        encoder = CategoryEncoder(columns=["course"], strategy="label")
        result = encoder.fit_transform(df)

        assert "course_encoded" in result.columns
        # Sorted unique: Hanshin=0, Kyoto=1, Tokyo=2
        encoded_vals = result["course_encoded"].to_list()
        assert encoded_vals[0] == 2  # Tokyo
        assert encoded_vals[1] == 1  # Kyoto
        assert encoded_vals[2] == 2  # Tokyo
        assert encoded_vals[3] == 0  # Hanshin

    def test_onehot_encoding(self):
        df = pl.DataFrame(
            {
                "weather": ["Sunny", "Rain", "Sunny"],
            }
        )

        encoder = CategoryEncoder(columns=["weather"], strategy="onehot")
        result = encoder.fit_transform(df)

        assert "weather_Rain" in result.columns
        assert "weather_Sunny" in result.columns
        assert result["weather_Sunny"].to_list() == [1, 0, 1]
        assert result["weather_Rain"].to_list() == [0, 1, 0]

    def test_transform_before_fit_raises(self):
        encoder = CategoryEncoder(columns=["course"])
        df = pl.DataFrame({"course": ["Tokyo"]})
        with pytest.raises(RuntimeError):
            encoder.transform(df)

    def test_unknown_value_gets_default(self):
        train = pl.DataFrame({"course": ["Tokyo", "Kyoto"]})
        test = pl.DataFrame({"course": ["Hanshin"]})

        encoder = CategoryEncoder(columns=["course"], strategy="label")
        encoder.fit(train)
        result = encoder.transform(test)

        assert result["course_encoded"][0] == -1


# ---------------------------------------------------------------------------
# FeatureScaler tests
# ---------------------------------------------------------------------------


class TestFeatureScaler:
    """Tests for FeatureScaler."""

    def test_standard_scaling(self):
        df = pl.DataFrame({"feat_distance": [1000, 1500, 2000, 2500, 3000]})

        scaler = FeatureScaler(columns=["feat_distance"], strategy="standard")
        result = scaler.fit_transform(df)

        assert "feat_distance_scaled" in result.columns
        scaled = result["feat_distance_scaled"]
        # Mean should be approximately 0
        assert abs(scaled.mean()) < 0.01  # type: ignore[operator]

    def test_minmax_scaling(self):
        df = pl.DataFrame({"feat_distance": [1000, 2000, 3000]})

        scaler = FeatureScaler(columns=["feat_distance"], strategy="minmax")
        result = scaler.fit_transform(df)

        assert "feat_distance_scaled" in result.columns
        scaled = result["feat_distance_scaled"].to_list()
        assert scaled[0] == pytest.approx(0.0)
        assert scaled[2] == pytest.approx(1.0)

    def test_transform_before_fit_raises(self):
        scaler = FeatureScaler(columns=["feat_distance"])
        df = pl.DataFrame({"feat_distance": [1000]})
        with pytest.raises(RuntimeError):
            scaler.transform(df)

    def test_zero_std_handled(self):
        df = pl.DataFrame({"feat_distance": [1500, 1500, 1500]})

        scaler = FeatureScaler(columns=["feat_distance"], strategy="standard")
        result = scaler.fit_transform(df)

        # Should not produce NaN/inf
        scaled = result["feat_distance_scaled"].to_list()
        assert all(v is not None for v in scaled)


# ---------------------------------------------------------------------------
# FeaturePipeline tests
# ---------------------------------------------------------------------------


class TestFeaturePipeline:
    """Tests for FeaturePipeline."""

    def test_pipeline_with_single_extractor(self):
        df = pl.DataFrame(
            {
                "race_id": ["R001"],
                "distance": [1600],
                "track_condition": ["良"],
                "weather": ["晴"],
                "track_type": ["芝"],
                "num_entries": [16],
                "grade": ["G1"],
            }
        )

        pipeline = FeaturePipeline(
            extractors=[RaceFeatureExtractor()],
        )
        result = pipeline.fit_transform(df)

        assert "feat_distance" in result.columns
        assert "feat_track_condition" in result.columns

    def test_pipeline_with_multiple_extractors(self):
        df = pl.DataFrame(
            {
                "race_id": ["R001"],
                "distance": [1600],
                "track_condition": ["良"],
                "weather": ["晴"],
                "track_type": ["芝"],
                "num_entries": [16],
                "grade": ["G1"],
                "avg_finish": [2.5],
                "win_rate": [0.3],
                "top3_rate": [0.6],
                "days_since_last_race": [14],
                "horse_age": [4],
                "weight": [480.0],
                "num_past_races": [10],
                "win_odds": [5.5],
                "win_favorite": [2],
                "bracket_number": [3],
                "post_position": [5],
                "carried_weight": [55.0],
                "sex": ["牡"],
                "horse_weight_change": [4.0],
                "jockey_win_rate": [0.15],
                "jockey_top3_rate": [0.35],
                "jockey_course_win_rate": [0.20],
                "jockey_experience": [500],
                "avg_corner_pos_4": [3.2],
                "avg_last_3f_time": [35.5],
            }
        )

        pipeline = FeaturePipeline(
            extractors=[
                RaceFeatureExtractor(),
                HorseFeatureExtractor(),
                JockeyFeatureExtractor(),
                RunningStyleFeatureExtractor(),
            ],
        )
        result = pipeline.fit_transform(df)

        # All feature names from all extractors
        all_features = pipeline.feature_names
        assert len(all_features) == 6 + 14 + 4 + 2  # race + horse + jockey + running
        for feat in all_features:
            assert feat in result.columns

    def test_pipeline_from_config(self):
        config = {
            "extractors": ["race", "horse"],
            "scaler": {
                "columns": ["feat_distance"],
                "strategy": "standard",
            },
        }

        pipeline = FeaturePipeline.from_config(config)
        assert len(pipeline._extractors) == 2

    def test_pipeline_from_config_unknown_extractor_raises(self):
        config = {"extractors": ["race", "unknown_extractor"]}
        with pytest.raises(ValueError, match="Unknown extractor"):
            FeaturePipeline.from_config(config)

    def test_pipeline_get_feature_columns(self):
        df = pl.DataFrame(
            {
                "race_id": ["R001"],
                "distance": [1600],
                "track_condition": ["良"],
                "weather": ["晴"],
                "track_type": ["芝"],
                "num_entries": [16],
                "grade": ["G1"],
            }
        )

        pipeline = FeaturePipeline(extractors=[RaceFeatureExtractor()])
        result = pipeline.fit_transform(df)
        features_only = pipeline.get_feature_columns(result)

        # Should only contain feat_ columns
        assert all(c.startswith("feat_") for c in features_only.columns)
        assert "race_id" not in features_only.columns

    def test_pipeline_transform_without_fit_raises_with_scaler(self):
        pipeline = FeaturePipeline(
            extractors=[RaceFeatureExtractor()],
            scaler=FeatureScaler(columns=["feat_distance"]),
        )

        df = pl.DataFrame(
            {
                "distance": [1600],
                "track_condition": ["良"],
                "weather": ["晴"],
                "track_type": ["芝"],
                "num_entries": [16],
                "grade": ["G1"],
            }
        )

        with pytest.raises(RuntimeError):
            pipeline.transform(df)

    def test_pipeline_with_encoder_and_scaler(self):
        df = pl.DataFrame(
            {
                "race_id": ["R001", "R002", "R003"],
                "distance": [1600, 2000, 1200],
                "track_condition": ["良", "重", "良"],
                "weather": ["晴", "雨", "曇"],
                "track_type": ["芝", "ダート", "芝"],
                "num_entries": [16, 14, 18],
                "grade": ["G1", "", "G3"],
            }
        )

        pipeline = FeaturePipeline(
            extractors=[RaceFeatureExtractor()],
            scaler=FeatureScaler(
                columns=["feat_distance", "feat_num_entries"],
                strategy="standard",
            ),
        )
        result = pipeline.fit_transform(df)

        assert "feat_distance_scaled" in result.columns
        assert "feat_num_entries_scaled" in result.columns

    def test_add_extractor(self):
        pipeline = FeaturePipeline()
        pipeline.add_extractor(RaceFeatureExtractor())
        assert len(pipeline._extractors) == 1


# ---------------------------------------------------------------------------
# FeaturePipeline fit/transform boundary tests
# ---------------------------------------------------------------------------


class TestFeaturePipelineFitTransformBoundary:
    """Verify that fit learns only from train data and transform uses fitted stats."""

    def test_encoder_fitted_on_train_only(self):
        """Encoder mappings must contain only categories seen in training data."""
        train = pl.DataFrame({"course": ["Tokyo", "Kyoto"]})
        test = pl.DataFrame({"course": ["Hanshin"]})

        encoder = CategoryEncoder(columns=["course"], strategy="label")
        pipeline = FeaturePipeline(
            extractors=[],
            encoder=encoder,
        )
        pipeline.fit(train)

        # Mappings should only contain train categories
        assert set(encoder._mappings["course"].keys()) == {"Kyoto", "Tokyo"}
        assert "Hanshin" not in encoder._mappings["course"]

        # Transforming test data should map unseen category to -1
        result = pipeline.transform(test)
        assert result["course_encoded"][0] == -1

    def test_scaler_fitted_on_train_only(self):
        """Scaler stats must reflect only the training data distribution."""
        train = pl.DataFrame({"feat_val": [10.0, 20.0, 30.0]})

        scaler = FeatureScaler(columns=["feat_val"], strategy="standard")
        pipeline = FeaturePipeline(
            extractors=[],
            scaler=scaler,
        )
        pipeline.fit(train)

        stats = scaler._stats["feat_val"]
        assert stats.mean == pytest.approx(20.0)
        # Polars std uses ddof=1 by default: std([10,20,30]) = 10.0
        assert stats.std == pytest.approx(10.0)

    def test_transform_before_fit_raises(self):
        """Calling transform without fit must raise RuntimeError."""
        pipeline = FeaturePipeline(
            extractors=[],
            encoder=CategoryEncoder(columns=["course"]),
            scaler=FeatureScaler(columns=["feat_val"]),
        )
        df = pl.DataFrame({"course": ["Tokyo"], "feat_val": [1.0]})

        with pytest.raises(RuntimeError):
            pipeline.transform(df)

    def test_transform_uses_fitted_stats_not_new_data(self):
        """Scaled output must use train statistics, not test data's own stats."""
        train = pl.DataFrame({"feat_val": [10.0, 20.0, 30.0]})
        # Test data with a very different distribution
        test = pl.DataFrame({"feat_val": [10.0, 100.0, 1000.0]})

        scaler = FeatureScaler(columns=["feat_val"], strategy="standard")
        pipeline = FeaturePipeline(extractors=[], scaler=scaler)
        pipeline.fit(train)

        result = pipeline.transform(test)
        scaled = result["feat_val_scaled"].to_list()

        # Train: mean=20.0, std=10.0
        # test value 10.0 -> (10-20)/10 = -1.0
        assert scaled[0] == pytest.approx(-1.0)
        # test value 100.0 -> (100-20)/10 = 8.0
        assert scaled[1] == pytest.approx(8.0)
        # test value 1000.0 -> (1000-20)/10 = 98.0
        assert scaled[2] == pytest.approx(98.0)
