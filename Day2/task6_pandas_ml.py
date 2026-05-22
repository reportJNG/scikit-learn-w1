'''Goal

Predict if a student will pass or fail using:

study_hours
attendance
previous_score

Output:

0 = fail
1 = pass

So use:

LogisticRegression
Step 1 — Create CSV file

Create file:

student_pass.csv

Put this inside:

study_hours,attendance,previous_score,passed
1,40,25,0
2,50,35,0
3,55,40,0
4,60,45,0
5,70,55,1
6,75,65,1
7,85,80,1
8,90,88,1
4,65,50,0
6,80,70,1
Step 2 — Create Python file
task6_pandas_ml.py
Your task ✅

Your program must:

1. Import pandas
2. Read student_pass.csv
3. Print first rows using df.head()
4. Print columns
5. Set X = study_hours, attendance, previous_score
6. Set y = passed
7. Split train/test
8. Train LogisticRegression
9. Predict this student: [6.5, 78, 72]
10. Calculate accuracy
11. Print prediction + accuracy'''