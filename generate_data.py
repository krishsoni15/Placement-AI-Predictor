import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 600
cgpa = np.round(rng.uniform(5.0, 10.0, n), 2)
aptitude = np.clip(np.round(rng.normal(65 + (cgpa-7)*7, 13), 0), 25, 100).astype(int)
coding = np.clip(np.round(rng.normal(62 + (cgpa-7)*8, 15), 0), 20, 100).astype(int)
communication = np.clip(np.round(rng.normal(65 + (cgpa-7)*5, 12), 0), 25, 100).astype(int)
internships = np.clip(rng.poisson(1.2, n), 0, 4).astype(int)
projects = np.clip(rng.poisson(2.2, n), 0, 6).astype(int)
attendance = np.clip(np.round(rng.normal(78 + (cgpa-7)*4, 9), 0), 55, 100).astype(int)
certifications = np.clip(rng.poisson(1.5, n), 0, 5).astype(int)
backlogs = np.clip(rng.poisson(0.7, n), 0, 5).astype(int)
score = (
    1.35*cgpa + 0.025*aptitude + 0.035*coding + 0.018*communication +
    0.45*internships + 0.32*projects + 0.012*attendance +
    0.18*certifications - 0.65*backlogs + rng.normal(0, 0.8, n)
)
threshold = np.quantile(score, 0.52)
placed = (score >= threshold).astype(int)

df = pd.DataFrame({
    'CGPA': cgpa,
    'Aptitude_Score': aptitude,
    'Coding_Score': coding,
    'Communication_Score': communication,
    'Internships': internships,
    'Projects': projects,
    'Attendance': attendance,
    'Certifications': certifications,
    'Backlogs': backlogs,
    'Placed': placed
})
df.to_csv('student_placement.csv', index=False)
print(df.head())
print(df['Placed'].value_counts())
