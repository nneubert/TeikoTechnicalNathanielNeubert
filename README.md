To run this analysis program simply assure that all required packages are installed by running

pip install -r requirements.txt

After this is finished, run

python -m streamlit run dashboard.py

in the terminal. This will also provide the link to the dashboard. The results will take a minute to load. 

I divided the csv into four main tables.
projects -> many subjects
subject -> samples
saple -> cell_counts
1. projects
    Columns: project_id, project_name
    Purpose: Stores the projects uniquely
2. subjects
    Columns: project_id, subject_id (internal for joins), subject, treatment, response, sample_type, condition, sex
    Purpose: Stores metadata for individual subjects. The project_id links each subject to a project.
3. samples
    Columns: sample_id, sample, subject_id, time_from_treatment_start, total_count
    Purpose: Stores information about each sample taken from a subject.
4. cell_counts
    Columns: cell_count_id, sample_id, cell_type, count, percentage
    Purpose: Stores the immune cell counts for each sample on a cell type basis.

By organizing the data this way we avoid repeating any metadata at lower levels.
Adding new cells can be done by modifying the cell_counts tables rather than requiring a more extensive reformating.
Use of additional keys that use integers allow for easier joining.
Use of several layers in the dataframe allows for easy expansion of the number of projects while minimizing data storage requirements.

I started out with many more files based on modifying a pandas dataframe to accomplish the desired tasks. Once I was satisfied, I altered my approach to focus more exclusively on using sqlite3 to directly generate the dataframes I wanted to analyze. 

The real backbone of this project is the structure of the dataset, it allows for queries of robust variety. From it I was able to quickly sample desired samples, projects, or various conditions and treatments for subjects, etc. 

analysis.py houses the key analysis functions of the program. This provides the tools we might want to use when analyzing a generated dataframe. 

dashboard.py is like the frame that puts the overall picture together. In it I placed the majority of sqlite3 queries to generate the desired dataframes and run the desired analysis for the assignment. With additional time, I would like to have made the dashboard more generalizable, but it succeeds for the current assignment.
