# subset_analysis.py

import sqlite3
from typing import Dict

def summarize_baseline_subset_db(conn: sqlite3.Connection, 
                                 condition: str = "melanoma", 
                                 treatment: str = "miraclib") -> Dict:
    """
    Query the database to get baseline PBMC samples for melanoma patients treated with miraclib
    and return summary counts:
        - samples per project
        - unique subjects responders/non-responders
        - unique subjects males/females
    """
    cur = conn.cursor()
    
    # Correct SQL query: sample_type is in subjects, not samples
    query = """
        SELECT s.sample_id, s.subject_id, subj.sex, subj.response, proj.project_name
        FROM samples s
        JOIN subjects subj ON s.subject_id = subj.subject_id
        JOIN projects proj ON subj.project_id = proj.project_id
        WHERE subj.sample_type = 'PBMC'
          AND s.time_from_treatment_start = 0
          AND subj.condition = ?
          AND subj.treatment = ?
    """
    
    cur.execute(query, (condition, treatment))
    rows = cur.fetchall()

    # Initialize counters
    samples_per_project = {}
    subjects_response = {"responders": 0, "non_responders": 0}
    subjects_sex = {"males": 0, "females": 0}
    seen_subjects = set()
    
    for sample_id, subject_id, sex, response, project_name in rows:
        # Count samples per project
        samples_per_project[project_name] = samples_per_project.get(project_name, 0) + 1

        # Count unique subjects
        if subject_id not in seen_subjects:
            seen_subjects.add(subject_id)
            # Response
            if response.lower() == "yes":
                subjects_response["responders"] += 1
            elif response.lower() == "no":
                subjects_response["non_responders"] += 1
            # Sex
            if sex.upper() == "M":
                subjects_sex["males"] += 1
            elif sex.upper() == "F":
                subjects_sex["females"] += 1

    return {
        "samples_per_project": samples_per_project,
        "subjects_response": subjects_response,
        "subjects_sex": subjects_sex
    }
