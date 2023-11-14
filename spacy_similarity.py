import os
import json
import logging
import spacy
import numpy as np

nlp = spacy.load("en_core_web_lg")  # make sure to use larger package!

def read_doc(path):
    with open(path) as f:
        try:
            data = json.load(f)
        except Exception as e:
            logging.error(f'Error reading JSON file: {e}')
            data = {}
    return data

def find_path(folder_name):
    curr_dir = os.getcwd()
    while True:
        if folder_name in os.listdir(curr_dir):
            return os.path.join(curr_dir, folder_name)
        else:
            parent_dir = os.path.dirname(curr_dir)
            if parent_dir == '/':
                break
            curr_dir = parent_dir
    raise ValueError(f"Folder '{folder_name}' not found.")

# Algorithm for matching job keywords against extracted resume keywords using cosine similarty forumla
def similarity_match(job_keywords, resume_keywords):
    results = []
    resume_keywords_lower = set([word.lower() for word in resume_keywords])
    job_keywords_lower = set([word.lower() for word in job_keywords])
    for jd_word in job_keywords_lower:
        jd_word_results = []
        jd_word_nlp = nlp(jd_word)
        if jd_word_nlp.has_vector:
            for resume_word in resume_keywords_lower:
                resume_word_nlp = nlp(resume_word)
                if resume_word_nlp.has_vector:
                    similarity = np.dot(jd_word_nlp.vector, resume_word_nlp.vector) / (np.linalg.norm(jd_word_nlp.vector) * np.linalg.norm(resume_word_nlp.vector))
                    jd_word_results.append(similarity)
                else:
                    jd_word_results.append(0)
            results.append(max(jd_word_results))
        else:
            results.append(0)
    return round((sum(results) / len(results)) * 100, 2)


cwd = find_path('Resume-Matcher')
READ_RESUME_FROM = os.path.join(cwd, 'Data', 'Processed', 'Resumes')
READ_JOB_DESCRIPTION_FROM = os.path.join(cwd, 'Data', 'Processed', 'JobDescription')

if __name__ == "__main__":
    # Example: MERN Full Stack Developer Skills
    job_keywords = ('Nodejs', "TypeScript", "JavaScript", "React", "Nextjs", "MongoDB", "Expressjs")

    resume_file_match = "/Resume-bruce_wayne_fullstack.pdf.json"
    resume_dict_match = read_doc(READ_RESUME_FROM + resume_file_match)["extracted_keywords"]
    resume_file_mismatch = "/Resume-alfred_pennyworth_pm.pdf.json"
    resume_dict_mismatch = read_doc(READ_RESUME_FROM + resume_file_mismatch)["extracted_keywords"]

    #### Proof of concept
    match_percentage_match = similarity_match(job_keywords, resume_dict_match)
    print( f'\n## Resume with Match: {resume_file_match}')
    print( f'\n## Match percentage based on Keywords: {match_percentage_match}%\n')

    match_percentage_mismatch = similarity_match(job_keywords, resume_dict_mismatch)
    print( f'\n## Resume with Mismatch: {resume_file_mismatch}')
    print( f'\n## Match percentage based on Keywords: {match_percentage_mismatch}%\n')