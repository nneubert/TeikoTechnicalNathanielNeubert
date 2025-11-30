To run this analysis program simply assure that all required packages are installed by running

pip install -r requirements.txt

After this is finished, run

streamlit run dashboard.py

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

At my core, I am a mathematician. I actually designed a completely different set of files when I first began working on this project based purely on the csv file using pandas for my analysis. After I determined which statistical tests I liked, I began to code four five new python files.
Each python file is divided up based on function. data_loader.py creates the dataset from the cell-count.csv file. analysis.py holds the analysis and visualization functions used to determine if any of the data is statistically significant. feature_engineering.py helps with the generation of summary tables. subset_analysis.py performs the queries necessary for the final part of the assignment. dashboard.py is the glue that holds it all together and presents the results in a visual format. 
