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
    # 1. UCI Student Performance (Regression: predict final grade G3)
    #    G1 and G2 (period grades) are valid predictors — they represent
    #    mid-year assessments, not the final result.
    # ------------------------------------------------------------------
    def load_uci_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads the UCI Student Performance dataset (Regression task)."""
        filepath = self.data_dir / "student-mat.csv"
        if not filepath.exists():
            logger.info("Downloading UCI Student Performance dataset...")
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
        X = df.drop(columns=['G3'])
        y = df['G3']
        logger.info(f"Loaded UCI dataset: {X.shape[0]} students, {X.shape[1]} features")
        return X, y

    # ------------------------------------------------------------------
    # 2. xAPI Educational Data (Classification: Low / Medium / High)
    # ------------------------------------------------------------------
    def load_xapi_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads the xAPI-Edu-Data dataset (3-class Classification task)."""
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
    # 3. UCI Predict Students' Dropout and Academic Success (UCI #697)
    #    3-class: 0=Dropout, 1=Enrolled, 2=Graduate
    #    4,424 students · 36 numeric features · no missing values
    #    Published benchmark — papers report 88–93% accuracy.
    #    Citation: Realinho et al. (2022), Data, MDPI. DOI:10.3390/data7110146
    # ------------------------------------------------------------------
    def load_dropout_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads the UCI Dropout & Academic Success dataset (3-class Classification)."""
        filepath = self.data_dir / "dropout_data.csv"
        if not filepath.exists():
            logger.info("Downloading UCI Dropout & Academic Success dataset (UCI #697)...")
            url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
            try:
                r = requests.get(url, timeout=60)
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    # The zip contains a file named 'data.csv'
                    names = z.namelist()
                    csv_name = next((n for n in names if n.endswith('.csv')), None)
                    if csv_name:
                        with z.open(csv_name) as f:
                            content = f.read()
                        filepath.write_bytes(content)
                        logger.info(f"Extracted '{csv_name}' → dropout_data.csv")
                    else:
                        logger.error(f"No CSV found in zip. Contents: {names}")
                        return self._mock_dropout_data()
            except Exception as e:
                logger.error(f"Failed to download Dropout dataset: {e}")
                return self._mock_dropout_data()

        df = pd.read_csv(filepath, sep=';')

        # Normalise column names (strip extra whitespace)
        df.columns = df.columns.str.strip()

        # Target column is named 'Target' with string labels
        if 'Target' not in df.columns:
            logger.error(f"Expected 'Target' column. Found: {df.columns.tolist()}")
            return self._mock_dropout_data()

        target_map = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
        y = df['Target'].map(target_map)

        # Drop any rows where target didn't map (shouldn't happen)
        valid = y.notna()
        df = df[valid].copy()
        y  = y[valid].copy()

        X = df.drop(columns=['Target'])

        logger.info(
            f"Loaded Dropout dataset: {X.shape[0]} students, {X.shape[1]} features  "
            f"| Class dist: {y.value_counts().sort_index().to_dict()}"
        )
        return X, y

    # ------------------------------------------------------------------
    # Legacy OULAD loader (kept for reference, not used in main pipeline)
    # ------------------------------------------------------------------
    def load_oulad_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads the OULAD dataset (kept for reference)."""
        filepath = self.data_dir / "studentInfo.csv"
        assessment_path = self.data_dir / "studentAssessment.csv"

        if not filepath.exists():
            logger.warning("OULAD studentInfo.csv not found. Using mock data.")
            return self._mock_oulad_data()

        df = pd.read_csv(filepath)

        if assessment_path.exists():
            assess = pd.read_csv(assessment_path)
            assess = assess.dropna(subset=['score'])
            agg = assess.groupby('id_student').agg(
                mean_score       =('score', 'mean'),
                median_score     =('score', 'median'),
                max_score        =('score', 'max'),
                min_score        =('score', 'min'),
                score_std        =('score', 'std'),
                num_assessments  =('id_assessment', 'count'),
                mean_submit_day  =('date_submitted', 'mean'),
            ).reset_index()
            agg['score_std'] = agg['score_std'].fillna(0)
            df = df.merge(agg, on='id_student', how='left')
            for col in ['mean_score','median_score','max_score','min_score',
                        'score_std','num_assessments','mean_submit_day']:
                df[col] = df[col].fillna(0)

        vle_path = self.data_dir / "vle_aggregated.csv"
        if vle_path.exists():
            vle = pd.read_csv(vle_path)
            df = df.merge(vle, on='id_student', how='left')
            df['total_clicks'] = df['total_clicks'].fillna(0)

        X = df.drop(columns=['final_result', 'id_student'])
        target_map = {'Distinction': 3, 'Pass': 2, 'Withdrawn': 1, 'Fail': 0}
        y = df['final_result'].map(target_map)
        logger.info(f"Loaded OULAD dataset: {X.shape[0]} students, {X.shape[1]} features")
        return X, y

    # ------------------------------------------------------------------
    # Mock fallbacks
    # ------------------------------------------------------------------
    def _mock_dropout_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Using Mock Dropout data for demonstration.")
        np.random.seed(42)
        n = 400
        cols = [
            'Marital status', 'Application mode', 'Application order',
            'Course', 'Daytime/evening attendance', 'Previous qualification',
            'Previous qualification (grade)', 'Nacionality',
            "Mother's qualification", "Father's qualification",
            "Mother's occupation", "Father's occupation",
            'Admission grade', 'Displaced', 'Educational special needs',
            'Debtor', 'Tuition fees up to date', 'Gender',
            'Scholarship holder', 'Age at enrollment',
            'International', 'Curricular units 1st sem (credited)',
            'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
            'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
            'Curricular units 1st sem (without evaluations)',
            'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
            'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
            'Unemployment rate', 'Inflation rate', 'GDP'
        ]
        data = {c: np.random.randn(n) for c in cols}
        y = pd.Series(np.random.choice([0, 1, 2], n, p=[0.33, 0.18, 0.49]))
        return pd.DataFrame(data), y

    def _mock_xapi_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Using Mock xAPI data for demonstration.")
        np.random.seed(42)
        n = 480
        data = {
            'gender':                  np.random.choice(['M', 'F'], n),
            'NationalITy':             np.random.choice(['KW', 'lebanon', 'Egypt', 'SaudiArabia'], n),
            'PlaceofBirth':            np.random.choice(['KuwaIT', 'lebanon', 'Egypt', 'SaudiArabia'], n),
            'StageID':                 np.random.choice(['lowerlevel', 'MiddleSchool', 'HighSchool'], n),
            'GradeID':                 np.random.choice(['G-04', 'G-07', 'G-08'], n),
            'SectionID':               np.random.choice(['A', 'B', 'C'], n),
            'Topic':                   np.random.choice(['IT', 'Math', 'Arabic', 'Science'], n),
            'Semester':                np.random.choice(['F', 'S'], n),
            'Relation':                np.random.choice(['Father', 'Mum'], n),
            'raisedhands':             np.random.randint(0, 100, n),
            'VisITedResources':        np.random.randint(0, 100, n),
            'AnnouncementsView':       np.random.randint(0, 100, n),
            'Discussion':              np.random.randint(0, 100, n),
            'ParentAnsweringSurvey':   np.random.choice(['Yes', 'No'], n),
            'ParentschoolSatisfaction':np.random.choice(['Good', 'Bad'], n),
            'StudentAbsenceDays':      np.random.choice(['Under-7', 'Above-7'], n),
        }
        y = pd.Series(np.random.choice([0, 1, 2], n))
        return pd.DataFrame(data), y

    def _mock_oulad_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Using Mock OULAD data for demonstration.")
        np.random.seed(42)
        n = 200
        data = {
            'code_module':         np.random.choice(['AAA', 'BBB', 'CCC'], n),
            'code_presentation':   np.random.choice(['2013J', '2014B'], n),
            'gender':              np.random.choice(['M', 'F'], n),
            'region':              np.random.choice(['Scotland', 'East Anglian Region', 'London'], n),
            'highest_education':   np.random.choice(['A Level', 'HE Qualification', 'Lower Than A Level'], n),
            'imd_band':            np.random.choice(['20-30%', '90-100%', '0-10%'], n),
            'age_band':            np.random.choice(['0-35', '35-55', '55<='], n),
            'num_of_prev_attempts':np.random.randint(0, 3, n),
            'studied_credits':     np.random.randint(30, 120, n),
            'disability':          np.random.choice(['Y', 'N'], n),
            'mean_score':          np.random.uniform(40, 95, n),
            'num_assessments':     np.random.randint(1, 8, n),
        }
        y = pd.Series(np.random.choice([0, 1, 2, 3], n))
        return pd.DataFrame(data), y
