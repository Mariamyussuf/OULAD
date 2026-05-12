import pandas as pd
import requests
import zipfile
import io
import logging
from pathlib import Path
from typing import Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Class responsible for loading and preparing educational datasets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # UCI Student Performance (Regression: predict final grade G3)
    # G1 and G2 (period grades) are valid predictors — they represent
    # mid-year assessments, not the final result.
    # ------------------------------------------------------------------
    def load_uci_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads the UCI Student Performance dataset (Regression task)."""
        filepath = self.data_dir / "student-mat.csv"
        if not filepath.exists():
            logger.info("Downloading UCI dataset...")
            url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
            r = requests.get(url, timeout=30)
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(self.data_dir)
            student_zip_path = self.data_dir / "student.zip"
            if student_zip_path.exists():
                with zipfile.ZipFile(student_zip_path) as z2:
                    z2.extract("student-mat.csv", self.data_dir)
                    z2.extract("student-por.csv", self.data_dir)

        df = pd.read_csv(filepath, sep=";")

        # Keep G1 and G2 — they are period (mid-year) grades, not the
        # final grade. Including them reflects realistic early-prediction
        # scenarios where previous assessment results are available.
        X = df.drop(columns=['G3'])
        y = df['G3']

        logger.info(f"Loaded UCI dataset: {X.shape[0]} students, {X.shape[1]} features")
        return X, y

    # ------------------------------------------------------------------
    # xAPI Educational Data (Classification: Low / Medium / High)
    # ------------------------------------------------------------------
    def load_xapi_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads the xAPI-Edu-Data dataset (Classification task)."""
        filepath = self.data_dir / "xAPI-Edu-Data.csv"
        if not filepath.exists():
            logger.info("Downloading xAPI dataset...")
            url = (
                "https://raw.githubusercontent.com/basilatawneh/"
                "Students-Academic-Performance-Dataset-xAPI-Edu-Data-"
                "/master/xAPI-Edu-Data.csv"
            )
            try:
                r = requests.get(url, timeout=30)
                filepath.write_bytes(r.content)
            except Exception as e:
                logger.error(f"Failed to download xAPI: {e}")
                return self._mock_xapi_data()

        df = pd.read_csv(filepath)
        X = df.drop(columns=['Class'])
        y = df['Class'].map({'L': 0, 'M': 1, 'H': 2})
        logger.info(f"Loaded xAPI dataset: {X.shape[0]} students, {X.shape[1]} features")
        return X, y

    # ------------------------------------------------------------------
    # OULAD (Classification: Fail / Withdrawn / Pass / Distinction)
    # Enriched with aggregated assessment scores from studentAssessment.csv
    # ------------------------------------------------------------------
    def load_oulad_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads the OULAD dataset, enriched with assessment score aggregations."""
        filepath = self.data_dir / "studentInfo.csv"
        assessment_path = self.data_dir / "studentAssessment.csv"

        if not filepath.exists():
            logger.warning("OULAD studentInfo.csv not found. Using mock data.")
            return self._mock_oulad_data()

        df = pd.read_csv(filepath)

        # ---- Merge assessment features --------------------------------
        if assessment_path.exists():
            logger.info("Merging studentAssessment.csv aggregations into OULAD features...")
            assess = pd.read_csv(assessment_path)

            # Drop unsubmitted assessments (score = NaN)
            assess = assess.dropna(subset=['score'])

            # Aggregate per student
            agg = assess.groupby('id_student').agg(
                mean_score        = ('score', 'mean'),
                median_score      = ('score', 'median'),
                max_score         = ('score', 'max'),
                min_score         = ('score', 'min'),
                score_std         = ('score', 'std'),
                num_assessments   = ('id_assessment', 'count'),
                mean_submit_day   = ('date_submitted', 'mean'),   # avg submission timing
            ).reset_index()

            agg['score_std'] = agg['score_std'].fillna(0)

            df = df.merge(agg, on='id_student', how='left')

            # Students with no assessment records get 0-filled columns
            for col in ['mean_score', 'median_score', 'max_score', 'min_score',
                        'score_std', 'num_assessments', 'mean_submit_day']:
                df[col] = df[col].fillna(0)

            logger.info(
                f"Assessment features merged. Dataset shape: {df.shape}"
            )
        else:
            logger.warning(
                "studentAssessment.csv not found — model will use demographic features only."
            )

        # ---- Merge VLE clicks ----------------------------------------
        vle_path = self.data_dir / "vle_aggregated.csv"
        if vle_path.exists():
            logger.info("Merging VLE clickstream aggregations...")
            vle = pd.read_csv(vle_path)
            df = df.merge(vle, on='id_student', how='left')
            df['total_clicks'] = df['total_clicks'].fillna(0)
        else:
            logger.warning("vle_aggregated.csv not found. Clickstream data omitted.")

        X = df.drop(columns=['final_result', 'id_student'])
        target_map = {'Distinction': 3, 'Pass': 2, 'Withdrawn': 1, 'Fail': 0}
        y = df['final_result'].map(target_map)

        logger.info(f"Loaded OULAD dataset: {X.shape[0]} students, {X.shape[1]} features")
        return X, y

    # ------------------------------------------------------------------
    # Mock fallbacks
    # ------------------------------------------------------------------
    def _mock_xapi_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Using Mock xAPI data for demonstration.")
        np.random.seed(42)
        n = 480
        data = {
            'gender': np.random.choice(['M', 'F'], n),
            'NationalITy': np.random.choice(['KW', 'lebanon', 'Egypt', 'SaudiArabia'], n),
            'PlaceofBirth': np.random.choice(['KuwaIT', 'lebanon', 'Egypt', 'SaudiArabia'], n),
            'StageID': np.random.choice(['lowerlevel', 'MiddleSchool', 'HighSchool'], n),
            'GradeID': np.random.choice(['G-04', 'G-07', 'G-08'], n),
            'SectionID': np.random.choice(['A', 'B', 'C'], n),
            'Topic': np.random.choice(['IT', 'Math', 'Arabic', 'Science'], n),
            'Semester': np.random.choice(['F', 'S'], n),
            'Relation': np.random.choice(['Father', 'Mum'], n),
            'raisedhands': np.random.randint(0, 100, n),
            'VisITedResources': np.random.randint(0, 100, n),
            'AnnouncementsView': np.random.randint(0, 100, n),
            'Discussion': np.random.randint(0, 100, n),
            'ParentAnsweringSurvey': np.random.choice(['Yes', 'No'], n),
            'ParentschoolSatisfaction': np.random.choice(['Good', 'Bad'], n),
            'StudentAbsenceDays': np.random.choice(['Under-7', 'Above-7'], n),
        }
        y = pd.Series(np.random.choice([0, 1, 2], n))
        return pd.DataFrame(data), y

    def _mock_oulad_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Using Mock OULAD data for demonstration.")
        np.random.seed(42)
        n = 200
        data = {
            'code_module': np.random.choice(['AAA', 'BBB', 'CCC'], n),
            'code_presentation': np.random.choice(['2013J', '2014B'], n),
            'gender': np.random.choice(['M', 'F'], n),
            'region': np.random.choice(['Scotland', 'East Anglian Region', 'London'], n),
            'highest_education': np.random.choice(['A Level', 'HE Qualification', 'Lower Than A Level'], n),
            'imd_band': np.random.choice(['20-30%', '90-100%', '0-10%'], n),
            'age_band': np.random.choice(['0-35', '35-55', '55<='], n),
            'num_of_prev_attempts': np.random.randint(0, 3, n),
            'studied_credits': np.random.randint(30, 120, n),
            'disability': np.random.choice(['Y', 'N'], n),
            'mean_score': np.random.uniform(40, 95, n),
            'num_assessments': np.random.randint(1, 8, n),
        }
        y = pd.Series(np.random.choice([0, 1, 2, 3], n))
        return pd.DataFrame(data), y
